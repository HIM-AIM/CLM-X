# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import torch
import torch.nn.functional as F
from torch import nn



from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input

from models.multiway_network import MultiwayWrapper, set_split_position


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "swish":
        return F.silu
    else:
        raise NotImplementedError


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        dropout=0.0,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({embed_dim}) is not a multiple of the number of attention "
                f"heads ({num_heads})"
            )
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout_module = torch.nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def attention_ops(self, q, k, v, key_padding_mask=None, attn_mask=None, rel_pos=None, is_causal=False):
        batch_size = q.shape[0]
        seqlen = q.shape[1]
        nheads = self.num_heads
        qkv = torch.stack([q, k, v], dim=2)
        x = rearrange(qkv, "b s three h d -> b s (three h d)")
        key_padding_mask = 1 - key_padding_mask
        x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
        output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad,
            cu_seqlens,
            max_s,
            self.dropout_p if self.training else 0.0,
            softmax_scale=self.scaling,
            causal=False,
        )
        output_unpad = rearrange(output_unpad, "nnz h d -> nnz (h d)")
        context_layer = pad_input(output_unpad, indices, batch_size, seqlen)
        return context_layer

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
    ):
        bsz, src_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = rearrange(q, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.num_heads)

        if not self.training:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        context_layer = self.attention_ops(q, k, v, key_padding_mask, attn_mask)

        if not self.training:
            context_layer = context_layer.to(torch.float32)

        attn_out = self.dropout_module(self.out_proj(context_layer))
        return attn_out


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout=None,
        layernorm_eps=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)

    def forward(self, x):
        x_intermediate = self.dropout_module(self.activation_fn(self.fc1(x)))
        ffn_out = self.dropout_module(self.fc2(x_intermediate))
        return ffn_out

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.ffn_dim = args.encoder_ffn_embed_dim
        self.pre_norm = args.pre_norm

        self.ffn = MultiwayWrapper(
            args,
            self.build_ffn(
                self.embed_dim,
                self.args,
            ),
        )

        self.self_attn = MultiheadAttention(
            args=args,
            embed_dim=self.embed_dim,
            num_heads=args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )

        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def build_ffn(self, embed_dim, args):
        return FeedForwardNetwork(
            embed_dim,
            self.ffn_dim,
            args.activation_fn,
            args.dropout,
            args.activation_dropout,
            args.layernorm_eps,
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            args,
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )

    def forward(self, x, encoder_padding_mask, attn_mask=None, multiway_split_position=None):
        if multiway_split_position is not None:
            assert self.args.multiway
            self.apply(set_split_position(multiway_split_position))

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        if self.pre_norm:
            x_norm = self.layer_norm(x)
            x_attn = self.self_attn(
                query=x_norm,
                key=x_norm,
                value=x_norm,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            x = x + x_attn
            x = x + self.ffn(self.layer_norm(x))
        else:
            x_attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            x = self.layer_norm(x + x_attn)
            x = self.layer_norm(x + self.ffn(x))

        l_aux = None
        return x, l_aux


class Encoder(nn.Module):
    def __init__(
        self,
        args,
        **kwargs,
    ):
        self.args = args
        super().__init__(**kwargs)

        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.encoder_layers)])
        self.num_layers = len(self.layers)
        self.gradient_checkpointing = False

    def forward(
        self,
        token_embeddings,
        encoder_padding_mask,
        attn_mask=None,
        return_all_hiddens=False,
        multiway_split_position=None,
        **kwargs,
    ):
        if multiway_split_position is not None:
            assert self.args.multiway
            self.apply(set_split_position(multiway_split_position))

        encoder_states = []

        x = token_embeddings
        if return_all_hiddens:
            encoder_states.append(x)

        l_aux = []
        for _, layer in enumerate(self.layers):
            x, l_aux_i = layer(x, encoder_padding_mask, attn_mask, multiway_split_position)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)
            l_aux.append(l_aux_i)

        return {
            "encoder_out": x,
            "encoder_embedding": token_embeddings,
            "encoder_padding_mask": encoder_padding_mask,
            "encoder_states": encoder_states,
            "l_aux": l_aux,
        }