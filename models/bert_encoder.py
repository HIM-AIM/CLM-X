# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn import checkpoint_wrapper, wrap
try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm

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


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout,
        layernorm_eps,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = LayerNorm(ffn_dim, eps=layernorm_eps)
    
    def forward(self, x):
        x_intermediate = self.activation_fn(self.fc1(x))
        x_intermediate = self.fc2(x_intermediate)
        x_intermediate = self.dropout_module(x_intermediate)
        ffn_out = self.ffn_layernorm(x + x_intermediate)
        return ffn_out
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()



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

        self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))        
        self.dropout_module = torch.nn.Dropout(dropout)
    

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

        # bsz, src_len, self.encoder_dim
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        # q *= self.scaling
        # bsz, self.num_heads, src_len, self.head_dim
        q = q.view(bsz, src_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        # bsz, self.num_heads, src_len, self.src_len
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        # need prepare the mask value like key_padding_mask
        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        # raw key_padding_mask: bsz, src_len, 1 for the padding position
        if key_padding_mask is not None:
            # Fills elements of self tensor with value where mask is True
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
        
        # Normalize the attention scores to probabilities.
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout_module(attn_probs)

        # bsz, self.num_heads, src_len, self.src_len
        context_layer = torch.matmul(attn_probs, v)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        context_layer = context_layer.view(bsz, src_len, embed_dim)
        
        return context_layer


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.ffn_dim = args.encoder_ffn_embed_dim

        self.self_attn = self.build_self_attention(self.embed_dim, args)

        self.ffn = MultiwayWrapper(
            args,
            self.build_ffn(
                self.embed_dim,
                self.args,
            ),
        )
            
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
        
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        # have a residual connection and layernorm
        x = self.ffn(x)
        l_aux = None

        return x, l_aux


class Encoder(nn.Module):
    def __init__(
        self,
        args,
        **kwargs
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
            **kwargs
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

