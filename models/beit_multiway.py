# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math
import torch
import torch.nn as nn

from models.encoder_flash import Encoder


class BEiT3(nn.Module):
    def __init__(self, args, **kwargs):
        """_summary_
        input: atac, rna
        Args:
            args (_type_): _description_
        """
        super().__init__()
        self.args = args

        self.atac_embed = nn.Embedding(args.atac_vocab_size, args.encoder_embed_dim)
        self.rna_embed = nn.Embedding(args.rna_vocab_size, args.encoder_embed_dim)
        self.atac_value_embed = Continuous_atac_ValueEncoder(args.encoder_embed_dim, args.peak_length)
        self.rna_value_embed = Continuous_rna_ValueEncoder(args.encoder_embed_dim, args.dropout)
        self.atac_norm = nn.LayerNorm(args.encoder_embed_dim)
        self.rna_norm = nn.LayerNorm(args.encoder_embed_dim)
        if args.perturbation:
            self.perturbation_embed = nn.Embedding(3, args.encoder_embed_dim,padding_idx=2)
        self.encoder = Encoder(args)

    def forward(
            self,
            atac_tokens=None,
            rna_tokens=None,
            values_atac=None,
            values_rna=None,
            atac_padding_position=None,
            rna_padding_position=None,
            attn_mask=None,
            pert_flag=None
    ):
        assert atac_tokens is not None or rna_tokens is not None
        assert atac_padding_position is not None or rna_padding_position is not None

        if atac_tokens is None:
            x_r = self.rna_norm(self.rna_embed(rna_tokens))
            x_v = self.rna_value_embed(values_rna)
            if pert_flag is not None:
                x_p = self.perturbation_embed(pert_flag)
                x_v = x_v + x_p
            # option 1: loss to zero with position scaling
            scale_num = 8
            x = x_r + x_v
            encoder_padding_mask = rna_padding_position
            multiway_split_position = 0
        elif rna_tokens is None:
            x_a = self.atac_norm(self.atac_embed(atac_tokens))
            x_v = self.atac_value_embed(values_atac)
            if pert_flag is not None:
                x_p = self.perturbation_embed(pert_flag)
                x_v = x_v + x_p
            scale_num = 8
            x = x_a + x_v
            encoder_padding_mask = atac_padding_position
            multiway_split_position = -1
        else:
            x1_a = self.atac_norm(self.atac_embed(atac_tokens))
            _values_atac = values_atac.to(dtype=self.atac_value_embed.linear1.weight.dtype)
            x1_v = self.atac_value_embed(_values_atac)
            x1 = x1_a + x1_v

            multiway_split_position = x1.size(1)
            x2_r = self.rna_norm(self.rna_embed(rna_tokens))
            _values_rna = values_rna.to(dtype=self.rna_value_embed.linear1.weight.dtype)
            x2_v = self.rna_value_embed(_values_rna)
            x2 = x2_r + x2_v
            x = torch.cat([x1, x2], dim=1)

            if rna_padding_position is not None:
                encoder_padding_mask = torch.cat(
                    [
                        atac_padding_position,
                        rna_padding_position,
                    ],
                    dim=1,
                )
            else:
                encoder_padding_mask = None

        encoder_out = self.encoder(
            token_embeddings=x,
            encoder_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            multiway_split_position=multiway_split_position,
        )
        encoder_out["multiway_split_position"] = multiway_split_position

        return encoder_out

class Continuous_atac_ValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, peak_length: int,dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(peak_length, d_model)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # expand last dimension
        # x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class Continuous_rna_ValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)