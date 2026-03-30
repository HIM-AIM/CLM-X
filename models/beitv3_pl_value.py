import math
import torch
import os
import numpy as np
from torch import Tensor, nn
import torch.nn.functional as F
from transformers.optimization import get_cosine_schedule_with_warmup
import lightning as pl
from lightning.pytorch.utilities import grad_norm
from models.beit_multiway import BEiT3
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import pearsonr
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_optimizers_for_lightning(
    params,
    learning_rate: float,
    adam_weight_decay: float,
    warmup_steps: int,
    max_steps: int,
):
    optimizer = torch.optim.AdamW(
        params,
        lr=learning_rate,
        weight_decay=adam_weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

def random_mask_value_without_cls(
    target_values: torch.Tensor,
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = 0,
):
    values = target_values.clone().detach()
    non_padding_mask = values.ne(pad_value)
    cls_mask = non_padding_mask[:, 0].to(torch.long)
    n_mask = ((non_padding_mask.sum(dim=1) - cls_mask) * mask_ratio).long()
    mask_pos = torch.zeros_like(values, dtype=torch.bool)

    for i, num in enumerate(n_mask):
        if num == 0:
            continue
        non_pad_indices = non_padding_mask[i].nonzero(as_tuple=False).squeeze()
        non_pad_indices = non_pad_indices[non_pad_indices != 0]
        if num > non_pad_indices.numel():
            mask_indices = non_pad_indices
        else:
            mask_indices = non_pad_indices[torch.randperm(non_pad_indices.numel())[:num]]

        mask_pos[i, mask_indices] = True

    masked_values = values.masked_fill(mask_pos, mask_value)

    return masked_values, mask_pos.float()
def compute_auroc(y_true, y_score):
    """
    计算 AUROC，此函数接受 PyTorch Tensor 或者 numpy 数组作为输入。
    """
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy()
    else:
        y_true_np = y_true

    if isinstance(y_score, torch.Tensor):
        y_score_np = y_score.detach().cpu().numpy()
    else:
        y_score_np = y_score

    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true_np, y_score_np)

def compute_rmse(pred, target):
    """
    计算均方根误差（RMSE）
    """
    return np.sqrt(np.mean((pred - target) ** 2))

class ExprHead_atac(nn.Module):
    def __init__(self, d_model: int,peak_length: int):
        super().__init__()
        self.expr_fc1 = nn.Linear(d_model, d_model)
        self.expr_fc2 = nn.Linear(d_model, d_model)
        self.expr_fc3 = nn.Linear(d_model, peak_length)
        self.act = nn.LeakyReLU()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.expr_fc1.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.expr_fc2.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.expr_fc3.weight, gain=1 / math.sqrt(2))

    def forward(self, x: Tensor):
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        x = self.act(self.expr_fc1(x))
        x = self.act(self.expr_fc2(x))
        pred_value = self.expr_fc3(x) # (batch, seq_len)
        return pred_value

class ExprHead_rna(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.expr_fc1 = nn.Linear(d_model, d_model)
        self.expr_fc2 = nn.Linear(d_model, d_model)
        self.expr_fc3 = nn.Linear(d_model, 1)
        self.act = nn.LeakyReLU()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.expr_fc1.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.expr_fc2.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.expr_fc3.weight, gain=1 / math.sqrt(2))

    def forward(self, x: Tensor):
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        x = self.act(self.expr_fc1(x))
        x = self.act(self.expr_fc2(x))
        pred_value = self.expr_fc3(x).squeeze(-1)
        return pred_value



class GatedFusion(nn.Module):
    """
    learned gate: alpha = sigmoid(gate([e_rna,e_atac]))
    fused = alpha*e_rna + (1-alpha)*e_atac

    """
    def __init__(self, d_model: int, gate_type: str = "scalar", dropout: float = 0.1):
        super().__init__()
        assert gate_type in {"scalar", "vector"}
        self.gate_type = gate_type
        gate_out = 1 if gate_type == "scalar" else d_model

        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, gate_out),
        )

    def forward(self, e_rna: torch.Tensor, e_atac: torch.Tensor):
        h = torch.cat([e_rna, e_atac], dim=-1)          # [B, 2D]
        alpha = torch.sigmoid(self.gate(h))
        fused = alpha * e_rna + (1.0 - alpha) * e_atac
        return fused, alpha

class RNA_Decoder(nn.Module):
    """
    A neural network classifier with multiple layers for cell type classification.

    Args:
        embedding_dim (int): Dimension of the input embeddings.
        num_classes (int): Number of cell type classes to predict.
    """

    def __init__(self, embedding_dim, num_classes):
        super(RNA_Decoder, self).__init__()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.layer_norm1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 1024)
        self.layer_norm2 = nn.LayerNorm(1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, X):
        """
        Forward pass for the classifier.

        Args:
            X (tensor): Input features.
            calculate_loss (bool): Whether to calculate loss (default: False).
            target (tensor): Ground truth labels (required if calculate_loss is True).

        Returns:
            tensor: Predicted labels.
            tensor (optional): Computed loss if calculate_loss is True.
        """
        predicted_label = self.fc3(
            self.layer_norm2(self.gelu(self.fc2(self.dropout(self.layer_norm1(self.fc1(X))))))
        )

        return predicted_label
class BeitForPretrain(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.test_step_metrics = []

        self.test_step_outputs = []
        self.config = config

        self.task = config.task
        self.atac_mask_ratio = config.atac_mask_ratio
        self.rna_mask_ratio = config.rna_mask_ratio
        self.phase_change_epoch = config.phase_change_epoch

        self.beit3 = BEiT3(config)
        self.atac_mlm_scorer = ExprHead_atac(config.encoder_embed_dim,config.peak_length)
        self.rna_mlm_scorer = ExprHead_rna(config.encoder_embed_dim)
        self.mix_pred_atac_mlm_scorer = ExprHead_atac(config.encoder_embed_dim,config.peak_length)
        self.mix_pred_rna_mlm_scorer = ExprHead_rna(config.encoder_embed_dim)
        if config.translation_to_rna:
            self.RNA_prediction = RNA_Decoder(config.encoder_embed_dim, config.features_dim)
        if config.cell_type_annotation:
            self.gated_fuse = GatedFusion(d_model=config.encoder_embed_dim, gate_type="scalar", dropout=0.1)
            self.classifier = nn.Linear(config.encoder_embed_dim, config.num_classes)
            self.test_preds = []
            self.test_labels = []
            self.cell_type_to_idx = {label: idx for idx, label in enumerate(self.config.cell_types)}
            self.idx_to_cell_type = {idx: label for label, idx in self.cell_type_to_idx.items()}

    def infer_rna_mlm(self, gene_tokens, values, target_values, values_masked_pos, \
                      padding_mask):
        """
        """
        masked_values, mask_pos = random_mask_value_without_cls(
            target_values,
            mask_ratio=self.rna_mask_ratio,
            mask_value=-1,
            pad_value=-2,
        )
        # as the no, the length is fixed, no vision padding token
        outputs = self.beit3(
            atac_tokens=None,
            rna_tokens=gene_tokens,
            values_atac=None,
            values_rna=masked_values,
            atac_padding_position=None,
            rna_padding_position=padding_mask,
            attn_mask=None,
        )
        rna_feats = outputs["encoder_out"]
        rna_feats = rna_feats[mask_pos.bool()]

        return self.rna_mlm_scorer(rna_feats), target_values[mask_pos.bool()]

    def infer_atac_mlm(self, batch):

        gene_tokens = torch.from_numpy(self.config.gene_tokens).repeat(len(batch), 1, 1).squeeze().to(batch.device)
        padding_mask = gene_tokens.eq(self.config.pad_id)
        padding_mask = padding_mask.int()
        original_tokenized_batch = batch.clone()
        n_mask = int(self.config.context_length * self.atac_mask_ratio)
        random_integers = torch.randperm(self.config.context_length)[:n_mask]

        batch[:, random_integers] = -1

        outputs = self.beit3(
            atac_tokens= gene_tokens,
            rna_tokens=None,
            values_atac=batch.float(),
            values_rna=None,
            atac_padding_position=padding_mask,
            rna_padding_position=None,
            attn_mask=None,
        )
        atac_feats = outputs["encoder_out"]

        atac_feats = self.atac_mlm_scorer(atac_feats)
        atac_feats = atac_feats[:,random_integers]
        
        return atac_feats, original_tokenized_batch.float()[:,random_integers]

    def infer_rna_predict_atac_mlm(self, batch):
        """

        """
        gene_tokens_rna = batch['rna']['gene_tokens']
        target_values_rna = batch['rna']['target_values']
        padding_mask_rna = batch['rna']['padding_mask']

        masked_values_rna, mask_pos_rna = random_mask_value_without_cls(
            target_values_rna,
            mask_ratio=0,
            mask_value=-1,
            pad_value=-2,
        )
        gene_tokens_atac = torch.from_numpy(self.config.gene_tokens).repeat(len(batch['atac']), 1, 1).squeeze().to(batch['atac'].device)
        padding_mask_atac = gene_tokens_atac.eq(self.config.pad_id)
        padding_mask_atac = padding_mask_atac.int()
        original_tokenized_batch = batch['atac'].clone()
        n_mask = int(self.config.context_length * self.atac_mask_ratio)
        random_integers = torch.randperm(self.config.context_length)[:n_mask]

        batch['atac'][:, random_integers] = -1
        mask_pos_atac = torch.zeros_like(batch['atac'], dtype=torch.int)
        mask_pos_atac[:, random_integers] = 1
        outputs = self.beit3(
            atac_tokens=gene_tokens_atac,
            rna_tokens=gene_tokens_rna,
            values_atac=batch['atac'].float(),
            values_rna=masked_values_rna,
            atac_padding_position=padding_mask_atac,
            rna_padding_position=padding_mask_rna,
            attn_mask=None,
        )
        atac_feats = outputs["encoder_out"][:, :len(gene_tokens_atac[1])]
        rna_feats = outputs["encoder_out"][:, len(gene_tokens_atac[1]):]
        atac_feats = self.mix_pred_atac_mlm_scorer(atac_feats)
        atac_feats = atac_feats[:,random_integers]
        return atac_feats, original_tokenized_batch.float()[:,random_integers]
    def infer_atac_predict_rna_mlm(self, batch):

        gene_tokens_rna = batch['rna']['gene_tokens']
        target_values_rna = batch['rna']['target_values']
        padding_mask_rna = batch['rna']['padding_mask']

        masked_values_rna, mask_pos_rna = random_mask_value_without_cls(
            target_values_rna,
            mask_ratio=self.rna_mask_ratio,
            mask_value=-1,
            pad_value=-2,
        )
        gene_tokens_atac = torch.from_numpy(self.config.gene_tokens).repeat(len(batch['atac']), 1, 1).squeeze().to(batch['atac'].device)
        padding_mask_atac = gene_tokens_atac.eq(self.config.pad_id)
        padding_mask_atac = padding_mask_atac.int()

        n_mask = int(self.config.context_length * 0)
        random_integers = torch.randperm(self.config.context_length)[:n_mask]

        batch['atac'][:, random_integers] = -1
        mask_pos_atac = torch.zeros_like(batch['atac'], dtype=torch.int)
        mask_pos_atac[:, random_integers] = 1
        outputs = self.beit3(
            atac_tokens=gene_tokens_atac,
            rna_tokens=gene_tokens_rna,
            values_atac=batch['atac'].float(),
            values_rna=masked_values_rna,
            atac_padding_position=padding_mask_atac,
            rna_padding_position=padding_mask_rna,
            attn_mask=None,
        )

        rna_feats = outputs["encoder_out"][:, len(gene_tokens_atac[1]):]
        rna_feats = rna_feats[mask_pos_rna.bool()]
        return self.mix_pred_rna_mlm_scorer(rna_feats), target_values_rna[mask_pos_rna.bool()]

    def rna_translation_in_atac(self, batch):

        gene_tokens_rna = batch['rna']['gene_tokens']
        target_values_rna = batch['rna']['target_values']
        padding_mask_rna = batch['rna']['padding_mask']

        masked_values_rna, mask_pos_rna = random_mask_value_without_cls(
            target_values_rna,
            mask_ratio=0,
            mask_value=-1,
            pad_value=-2,
        )
        gene_tokens_atac = torch.from_numpy(self.config.gene_tokens).repeat(len(batch['atac']), 1, 1).squeeze().to(batch['atac'].device)
        padding_mask_atac = gene_tokens_atac.eq(self.config.pad_id)
        padding_mask_atac = padding_mask_atac.int()
        original_tokenized_batch = batch['atac'].clone()
        n_mask = int(self.config.context_length * self.atac_mask_ratio)
        random_integers = torch.randperm(self.config.context_length)[:n_mask]
        random_integers = torch.sort(random_integers)[0]
        batch['atac'][:, random_integers] = -1

        outputs = self.beit3(
            atac_tokens=gene_tokens_atac,
            rna_tokens=gene_tokens_rna,
            values_atac=batch['atac'].float(),
            values_rna=masked_values_rna,
            atac_padding_position=padding_mask_atac,
            rna_padding_position=padding_mask_rna,
            attn_mask=None,
        )

        if self.config.atac_feats:
            atac_feats = outputs["encoder_out"][:, :len(gene_tokens_atac[1])]
            atac_features = self.mix_pred_atac_mlm_scorer(atac_feats)
            atac_features = atac_features[:, random_integers]
            return atac_features, original_tokenized_batch[:, random_integers]
        elif self.config.rna_feats:
            rna_feats = outputs["encoder_out"][:, len(gene_tokens_atac[1]):]
            atac_features = self.mix_pred_atac_mlm_scorer(rna_feats)
            atac_features = atac_features[:, random_integers]
            return atac_features, original_tokenized_batch[:, random_integers]

    def atac_translation_in_rna_full(self, batch):

        gene_tokens_rna = batch['rna']['gene_tokens']
        target_values_rna = batch['rna']['target_values']
        target_values_rna_full = batch['rna']['target_values_rna_full']
        padding_mask_rna = batch['rna']['padding_mask']

        masked_values_rna, mask_pos_rna = random_mask_value_without_cls(
            target_values_rna,
            mask_ratio=self.rna_mask_ratio,
            mask_value=-1,
            pad_value=-2,
        )
        gene_tokens_atac = torch.from_numpy(self.config.gene_tokens).repeat(len(batch['atac']), 1, 1).squeeze().to(batch['atac'].device)
        padding_mask_atac = gene_tokens_atac.eq(self.config.pad_id)
        padding_mask_atac = padding_mask_atac.int()

        n_mask = int(self.config.context_length * 0)
        random_integers = torch.randperm(self.config.context_length)[:n_mask]
        random_integers = torch.sort(random_integers)[0]
        batch['atac'][:, random_integers] = -1
        mask_pos_atac = torch.zeros_like(batch['atac'], dtype=torch.int)
        mask_pos_atac[:, random_integers] = 1
        outputs = self.beit3(
            atac_tokens=gene_tokens_atac,
            rna_tokens=gene_tokens_rna,
            values_atac=batch['atac'].float(),
            values_rna=masked_values_rna,
            atac_padding_position=padding_mask_atac,
            rna_padding_position=padding_mask_rna,
            attn_mask=None,
        )
        if self.config.rna_cls:
            rna_feats = outputs["encoder_out"][:, len(gene_tokens_atac[1]):]
            cls_rna = rna_feats[:, 0, :]
            rna_features = self.RNA_prediction(cls_rna)  # (10, 36601)
            return rna_features, target_values_rna_full
        elif self.config.atac_cls:
            atac_feats = outputs["encoder_out"][:, :len(gene_tokens_atac[1])]
            cls_atac = atac_feats[:, 0, :]
            rna_features = self.RNA_prediction(cls_atac)  # (10, 36601)
            return rna_features, target_values_rna_full

    def atac_translation_in_rna(self, batch):
        gene_tokens_rna = batch['rna']['gene_tokens']
        target_values_rna = batch['rna']['target_values']
        padding_mask_rna = batch['rna']['padding_mask']

        masked_values_rna, mask_pos_rna = random_mask_value_without_cls(
            target_values_rna,
            mask_ratio=self.rna_mask_ratio,
            mask_value=-1,
            pad_value=-2,
        )
        gene_tokens_atac = torch.from_numpy(self.config.gene_tokens).repeat(len(batch['atac']), 1, 1).squeeze().to(batch['atac'].device)
        padding_mask_atac = gene_tokens_atac.eq(self.config.pad_id)
        padding_mask_atac = padding_mask_atac.int()
        n_mask = int(self.config.context_length * 0)
        random_integers = torch.randperm(self.config.context_length)[:n_mask]
        batch['atac'][:, random_integers] = -1
        mask_pos_atac = torch.zeros_like(batch['atac'], dtype=torch.int)
        mask_pos_atac[:, random_integers] = 1

        outputs = self.beit3(
            atac_tokens=gene_tokens_atac,
            rna_tokens=gene_tokens_rna,
            values_atac=batch['atac'].float(),
            values_rna=masked_values_rna,
            atac_padding_position=padding_mask_atac,
            rna_padding_position=padding_mask_rna,
            attn_mask=None,
        )
        if self.config.atac_cls:
            atac_feats = outputs["encoder_out"][:, :len(gene_tokens_atac[1])]
            cls_atac = atac_feats[:, 0, :]
            rna_features = self.RNA_prediction(cls_atac)  # (10, 36601)
            return rna_features, target_values_rna
        else:
            rna_feats = outputs["encoder_out"][:, len(gene_tokens_atac[1]):]

            return self.mix_pred_rna_mlm_scorer(rna_feats), target_values_rna

    def modality_fusion(self, batch):

        mod = str(self.config.embedding_modality).lower()
        if mod not in {"mix", "atac", "rna"}:
            raise ValueError(f"Unknown embedding_modality: {self.config.embedding_modality}")

        gene_tokens_rna = None
        target_values_rna = None
        padding_mask_rna = None
        masked_values_rna = None
        mask_pos_rna = None

        gene_tokens_atac = None
        padding_mask_atac = None
        values_atac_masked = None
        original_atac_values = None
        random_integers = None
        if mod in {"mix", "rna"}:
            gene_tokens_rna = batch["rna"]["gene_tokens"]
            target_values_rna = batch["rna"]["target_values"]
            padding_mask_rna = batch["rna"]["padding_mask"]
            if not isinstance(padding_mask_rna, torch.Tensor):
                padding_mask_rna = torch.tensor(padding_mask_rna, device=gene_tokens_rna.device)
            padding_mask_rna = padding_mask_rna.to(torch.long)

            masked_values_rna, mask_pos_rna = random_mask_value_without_cls(
                target_values_rna,
                mask_ratio=self.rna_mask_ratio,
                mask_value=-1,
                pad_value=-2,
            )

        if mod in {"mix", "atac"}:
            atac_raw = batch["atac"]  # [B, L_atac]
            if not isinstance(atac_raw, torch.Tensor):
                atac_raw = torch.tensor(atac_raw)

            original_atac_values = atac_raw.clone().float()
            values_atac_masked = original_atac_values.clone()

            B, L_atac = values_atac_masked.size(0), values_atac_masked.size(1)

            # gene_tokens_atac: [B, L_atac]
            gene_tokens_atac = torch.from_numpy(self.config.gene_tokens).to(values_atac_masked.device)
            if gene_tokens_atac.dim() == 1:
                gene_tokens_atac = gene_tokens_atac.unsqueeze(0)  # [1, L]
            if gene_tokens_atac.size(0) == 1:
                gene_tokens_atac = gene_tokens_atac.repeat(B, 1)  # [B, L]
            if gene_tokens_atac.size(0) != B:
                raise ValueError(
                    f"ATAC batch mismatch: atac B={B}, gene_tokens_atac B={gene_tokens_atac.size(0)}"
                )
            padding_mask_atac = gene_tokens_atac.eq(self.config.pad_id).to(torch.long)

            n_mask = int(L_atac * self.atac_mask_ratio)
            if n_mask > 0:
                random_integers = torch.randperm(L_atac, device=values_atac_masked.device)[:n_mask]
                random_integers = torch.sort(random_integers)[0]
                values_atac_masked[:, random_integers] = -1.0
            else:
                random_integers = torch.empty(0, dtype=torch.long, device=values_atac_masked.device)

        outputs = self.beit3(
            atac_tokens=gene_tokens_atac,
            rna_tokens=gene_tokens_rna,
            values_atac=values_atac_masked,
            values_rna=masked_values_rna,
            atac_padding_position=padding_mask_atac,
            rna_padding_position=padding_mask_rna,
            attn_mask=None,
        )
        if self.config.embedding_modality == "mix":
            atac_rna_feats = outputs["encoder_out"]

            cls_emb = atac_rna_feats[:, 0, :]

            rna_features = self.RNA_prediction(cls_emb)

            return rna_features, target_values_rna
        if self.config.embedding_modality == "atac":
            atac_feats = outputs["encoder_out"]

            atac_features = self.mix_pred_atac_mlm_scorer(atac_feats)

            atac_feats = atac_features[:, random_integers]

            return atac_feats, original_atac_values[:,random_integers]
        if self.config.embedding_modality == "rna":
            rna_feats = outputs["encoder_out"]

            rna_feats = rna_feats[mask_pos_rna.bool()]

            return self.mix_pred_rna_mlm_scorer(rna_feats), target_values_rna[mask_pos_rna.bool()]


    def batch_correction(self, batch):
        gene_tokens_rna = batch['rna']['gene_tokens']
        target_values_rna = batch['rna']['target_values']
        padding_mask_rna = batch['rna']['padding_mask']

        if getattr(self.config, "rna_cls", False):
            outputs = self.beit3(
                atac_tokens=None,
                rna_tokens=gene_tokens_rna,
                values_atac=None,
                values_rna=target_values_rna,
                atac_padding_position=None,
                rna_padding_position=padding_mask_rna,
                attn_mask=None,
            )

            rna_feats = outputs["encoder_out"]  # [B, L_rna(+CLS), D]
            cls_emb = rna_feats[:, 0, :]  # [B, D]
            rna_features = self.RNA_prediction(cls_emb)
            return rna_features, target_values_rna

        batch_value_atac = batch['atac'][:, :-1, :].clone()
        batch_label_atac = batch['atac'][:, -1, 0].reshape(-1, 1).clone()
        gene_tokens_atac = (
            torch.from_numpy(self.config.gene_tokens)
            .repeat(len(batch_value_atac), 1, 1)
            .squeeze()
            .to(self.device)
        )
        padding_mask_atac = gene_tokens_atac.eq(self.config.pad_id)

        outputs = self.beit3(
            atac_tokens=gene_tokens_atac.long(),
            rna_tokens=gene_tokens_rna,
            values_atac=batch_value_atac.float().to(self.device),
            values_rna=target_values_rna,
            atac_padding_position=padding_mask_atac,
            rna_padding_position=padding_mask_rna,
            attn_mask=None,
        )

        atac_rna_feats = outputs["encoder_out"]
        cls_emb = atac_rna_feats[:, 0, :]
        rna_features = self.RNA_prediction(cls_emb)
        return rna_features, target_values_rna

    def batch_correction_rna(self, batch):
        gene_tokens_rna = batch['rna']['gene_tokens']
        target_values_rna = batch['rna']['target_values']
        padding_mask_rna = batch['rna']['padding_mask']
        masked_values, mask_pos = random_mask_value_without_cls(
            target_values_rna,
            mask_ratio=self.rna_mask_ratio,
            mask_value=-1,
            pad_value=-2,
        )
        outputs = self.beit3(
            atac_tokens=None,
            rna_tokens=gene_tokens_rna,
            values_atac=None,
            values_rna=masked_values,
            atac_padding_position=None,
            rna_padding_position=padding_mask_rna,
            attn_mask=None,
        )

        rna_feats = outputs["encoder_out"]

        rna_feats = rna_feats[mask_pos.bool()]

        return self.mix_pred_rna_mlm_scorer(rna_feats), target_values_rna[mask_pos.bool()]

        return rna_features, target_values_rna

    def _repeat_gene_tokens(self, gene_tokens_np, B, device):
        t = torch.from_numpy(gene_tokens_np).to(device)
        # squeeze trailing singleton dims
        while t.dim() > 2 and t.size(-1) == 1:
            t = t.squeeze(-1)
        if t.dim() == 2 and t.size(1) == 1:  # [L,1] -> [L]
            t = t.squeeze(1)
        if t.dim() == 1:  # [L] -> [1,L]
            t = t.unsqueeze(0)
        if t.size(0) == 1:
            t = t.repeat(B, 1)
        if t.size(0) != B:
            raise ValueError(f"gene_tokens batch mismatch: expect B={B}, got {t.size(0)}")
        return t.long()

    def _slice_modal_feats(self, encoder_out, L_atac, L_rna):
        """
        兼容 encoder_out 里是否额外插了 global CLS：
        extra = L_total - (L_atac + L_rna)
        start_rna = extra + L_atac
        """
        L_total = encoder_out.size(1)
        extra = L_total - (L_atac + L_rna)
        if extra < 0:
            raise RuntimeError(f"encoder_out length mismatch: {L_total=} < {L_atac+L_rna=}")
        start_rna = extra + L_atac
        rna_feats = encoder_out[:, start_rna:start_rna + L_rna, :]
        return rna_feats

    def predict_rna_from_atac(self, batch, no_grad=True, detach=True):
        """
        输入 batch（至少要有 batch['atac']、batch['rna']['gene_tokens']、batch['rna']['padding_mask']）
        输出 pseudo RNA values（shape 与 rna_tokens 对齐）。
        """
        atac_values = batch["atac"]
        if not isinstance(atac_values, torch.Tensor):
            atac_values = torch.tensor(atac_values)

        device = atac_values.device
        B = atac_values.size(0)
        L_atac = atac_values.size(1)

        # ATAC tokens / pad
        atac_tokens = self._repeat_gene_tokens(self.config.gene_tokens, B, device)
        atac_padding_position = atac_tokens.eq(self.config.pad_id).to(torch.long)
        values_atac = atac_values.clone().float()

        # RNA tokens / pad
        rna_tokens = batch["rna"]["gene_tokens"]
        rna_padding_position = batch["rna"]["padding_mask"].to(torch.long)
        if not isinstance(rna_tokens, torch.Tensor):
            rna_tokens = torch.tensor(rna_tokens, device=device)
        if not isinstance(rna_padding_position, torch.Tensor):
            rna_padding_position = torch.tensor(rna_padding_position, device=device)

        rna_tokens = rna_tokens.to(device)
        rna_padding_position = rna_padding_position.to(device)

        values_rna = torch.full(
            (B, rna_tokens.size(1)),
            fill_value=float(self.config.rna_mask_value),
            device=device,
            dtype=torch.float,
        )
        values_rna = values_rna.masked_fill(rna_padding_position.bool(), float(self.config.rna_pad_value))

        def _forward():
            outputs = self.beit3(
                atac_tokens=atac_tokens,
                rna_tokens=rna_tokens,
                values_atac=values_atac,
                values_rna=values_rna,
                atac_padding_position=atac_padding_position,
                rna_padding_position=rna_padding_position,
                attn_mask=None,
            )
            encoder_out = outputs["encoder_out"]  # [B, L_total, D]
            rna_feats = self._slice_modal_feats(encoder_out, L_atac=atac_tokens.size(1), L_rna=rna_tokens.size(1))
            pred_rna = self.mix_pred_rna_mlm_scorer(rna_feats)

            if pred_rna.dim() == 2:
                pred_rna = pred_rna.masked_fill(rna_padding_position.bool(), float(self.config.rna_pad_value))
            return pred_rna

        if no_grad:
            with torch.no_grad():
                pred = _forward()
        else:
            pred = _forward()

        return pred.detach() if detach else pred

    def cell_type_annotation(self, batch):

        mod = str(self.config.embedding_modality).lower()
        if mod not in {"mix", "atac", "rna"}:
            raise ValueError(f"Unknown embedding_modality: {self.config.embedding_modality}")

        # ===== label: str -> int =====
        cell_type = batch["rna"]["cell_type"]
        int_cell_type_list = []
        for label in cell_type:
            if label in self.cell_type_to_idx:
                int_cell_type_list.append(self.cell_type_to_idx[label])
            else:
                raise ValueError(f"unknown cell type: {label}")

        atac_tokens = None
        values_atac = None
        atac_padding_position = None

        rna_tokens = None
        values_rna = None
        rna_padding_position = None

        # ===== build ATAC inputs (mix / atac) =====
        if mod in {"mix", "atac"}:
            atac_values_raw = batch["atac"]  # [B, L_atac]
            if not isinstance(atac_values_raw, torch.Tensor):
                atac_values_raw = torch.tensor(atac_values_raw)

            B = atac_values_raw.size(0)
            L_atac = atac_values_raw.size(1)

            gene_tokens_atac = torch.from_numpy(self.config.gene_tokens).to(atac_values_raw.device)
            if gene_tokens_atac.dim() == 1:
                gene_tokens_atac = gene_tokens_atac.unsqueeze(0)  # [1, L_atac]
            if gene_tokens_atac.size(0) == 1:
                gene_tokens_atac = gene_tokens_atac.repeat(B, 1)  # [B, L_atac]
            if gene_tokens_atac.size(0) != B:
                raise ValueError(f"ATAC batch mismatch: atac B={B}, gene_tokens_atac B={gene_tokens_atac.size(0)}")

            atac_tokens = gene_tokens_atac
            atac_padding_position = atac_tokens.eq(self.config.pad_id).to(torch.long)  # 1=pad,0=valid

            values_atac = atac_values_raw.clone().float()

            mask_ratio = 0.0
            n_mask = int(L_atac * mask_ratio)
            if n_mask > 0:
                idx = torch.randperm(L_atac, device=values_atac.device)[:n_mask]
                idx = torch.sort(idx)[0]
                values_atac[:, idx] = -1.0

        # ===== build RNA inputs (mix / rna) =====
        if mod in {"mix", "rna"}:
            rna_tokens = batch["rna"]["gene_tokens"]  # [B, L_rna]
            target_values_rna = batch["rna"]["target_values"]
            rna_padding_position = batch["rna"]["padding_mask"]  # [B, L_rna]
            rna_padding_position = rna_padding_position.to(torch.long)

            if not isinstance(rna_padding_position, torch.Tensor):
                rna_padding_position = torch.tensor(rna_padding_position, device=rna_tokens.device)

            masked_values_rna, _ = random_mask_value_without_cls(
                target_values_rna,
                mask_ratio=0,
                mask_value=-1,
                pad_value=-2,
            )
            values_rna = masked_values_rna

        if values_atac is not None:
            device = values_atac.device
        elif rna_tokens is not None:
            device = rna_tokens.device
        else:
            raise RuntimeError("No modality is provided to beit3 (both ATAC and RNA are None).")

        int_cell_type_tensor = torch.tensor(int_cell_type_list, dtype=torch.long, device=device)

        outputs = self.beit3(
            atac_tokens=atac_tokens,
            rna_tokens=rna_tokens,
            values_atac=values_atac,
            values_rna=values_rna,
            atac_padding_position=atac_padding_position,
            rna_padding_position=rna_padding_position,
            attn_mask=None,
        )
        encoder_out = outputs["encoder_out"]

        def masked_mean_without_cls(feats, pad_mask_bool):
            """
            feats: [B, L, D]，pad_mask_bool: [B, L]，True=padding
            return: [B, D]
            """
            pad_mask_bool = pad_mask_bool.bool()
            feats_ = feats[:, 1:, :]
            valid = (~pad_mask_bool[:, 1:]).unsqueeze(-1)  # [B, L-1, 1]
            denom = (~pad_mask_bool[:, 1:]).sum(dim=1).clamp(min=1).unsqueeze(-1).type_as(feats_)
            return (feats_ * valid.type_as(feats_)).sum(dim=1) / denom

        if mod == "mix":
            L_atac = atac_tokens.size(1)
            atac_seq = encoder_out[:, :L_atac, :]
            rna_seq = encoder_out[:, L_atac:, :]

            e_atac = masked_mean_without_cls(atac_seq, atac_padding_position)  # [B,D]
            e_rna = masked_mean_without_cls(rna_seq, rna_padding_position)  # [B,D]

            cell_emb_gate, alpha = self.gated_fuse(e_rna, e_atac)
            cell_emb = cell_emb_gate
        elif mod == "atac":
            if atac_padding_position is None:
                raise RuntimeError("mod=atac but atac_padding_position is None.")
            cell_emb = masked_mean_without_cls(encoder_out, atac_padding_position)
        elif mod == "rna":
            if rna_padding_position is None:
                raise RuntimeError("mod=rna but rna_padding_position is None.")
            cell_emb = masked_mean_without_cls(encoder_out, rna_padding_position)

        return int_cell_type_tensor, cell_emb

    def infer_rna_perturbation(self, perturb_value, gene_id, ctrl_value, pert_flag, index):

        gene_id = gene_id.long()
        ctrl_value = ctrl_value.float()
        pert_flag = pert_flag.long()
        perturb_value = perturb_value.float()
        padding_mask = torch.zeros_like(gene_id).long()

        outputs = self.beit3(
            atac_tokens=None,
            rna_tokens=gene_id,
            values_atac=None,
            values_rna=ctrl_value,
            atac_padding_position=None,
            rna_padding_position=padding_mask,
            attn_mask=None,
            pert_flag=pert_flag
        )
        rna_feats = outputs["encoder_out"]

        # 返回视觉MLM预测器的输出
        return self.rna_mlm_scorer(rna_feats), perturb_value


    def _get_split_epoch(self):

        if getattr(self.config, "two_stage_split_epoch", None) is not None:
            return int(self.config.two_stage_split_epoch)

        max_epochs = getattr(self.trainer, "max_epochs", None)
        if max_epochs is None or max_epochs <= 0:
            return 0
        ratio = float(getattr(self.config, "two_stage_split_ratio", 0.5))
        ratio = max(0.0, min(1.0, ratio))
        return int(max_epochs * ratio)
    def training_step(self, batch, batch_idx):
        """
        定义训练步骤，包括前向计算和损失计算
        """
        loss = None
        if self.task == "rnamlm":
            mlm_logits, mlm_labels = self.infer_rna_mlm(**batch)
            loss = F.mse_loss(mlm_logits, mlm_labels, reduction="mean")
        if self.task == "atacmlm":
            mlm_logits, mlm_labels= self.infer_atac_mlm(batch)
            BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction = 'mean')
            mlm_labels = mlm_labels.view(-1)
            non_negative_two_indices = torch.where(mlm_labels != -2)[0]
            loss = BCEWithLogitsLoss(mlm_logits.view(-1)[non_negative_two_indices], mlm_labels[non_negative_two_indices])

        if self.task == "rnaatacmlm" and self.config.pretrain and self.config.both_pretrain:
                current_epoch = self.current_epoch

                if current_epoch < self.phase_change_epoch:
                    mlm_logits_atac, mlm_labels_atac = self.infer_rna_predict_atac_mlm(batch)
                    BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='mean')
                    mlm_labels_atac = mlm_labels_atac.view(-1)
                    non_negative_two_indices = torch.where(mlm_labels_atac != -2)[0]
                    loss1 = BCEWithLogitsLoss(mlm_logits_atac.view(-1)[non_negative_two_indices],
                                              mlm_labels_atac[non_negative_two_indices])
                    total_loss = loss1
                else:
                    mlm_logits_rna, mlm_labels_rna = self.infer_atac_predict_rna_mlm(batch)
                    loss2 = F.mse_loss(mlm_logits_rna, mlm_labels_rna, reduction="mean")
                    total_loss = loss2
                self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                torch.cuda.empty_cache()
                return total_loss

        if self.task == "rnaatacmlm" and self.config.translation_to_atac:
            mlm_logits_atac, mlm_labels_atac = self.rna_translation_in_atac(batch)
            BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='mean')
            mlm_labels_atac = mlm_labels_atac.view(-1)
            non_negative_two_indices = torch.where(mlm_labels_atac != -2)[0]
            loss = BCEWithLogitsLoss(mlm_logits_atac.view(-1)[non_negative_two_indices],
                                      mlm_labels_atac[non_negative_two_indices])
        if self.task == "rnaatacmlm" and self.config.translation_to_rna:
            if self.config.pred_full:
                mlm_logits_rna, mlm_labels_rna = self.atac_translation_in_rna_full(batch)
                loss = F.mse_loss(mlm_logits_rna, mlm_labels_rna, reduction="mean")
            else:
                mlm_logits_rna, mlm_labels_rna = self.atac_translation_in_rna(batch)
                loss = F.mse_loss(mlm_logits_rna, mlm_labels_rna, reduction="mean")

        if self.task == "rnaatacmlm" and self.config.batch_correction:
            mlm_logits_rna, mlm_labels_rna = self.batch_correction(batch)
            loss = F.mse_loss(mlm_logits_rna, mlm_labels_rna, reduction="mean")

        if self.task == "rnaatacmlm" and self.config.cell_type_annotation:

            cell_type, cell_emb = self.cell_type_annotation(batch)
            cell_type_logits = self.classifier(cell_emb)
            ls = float(getattr(self.config, "label_smoothing", 0.0))
            loss = F.cross_entropy(cell_type_logits, cell_type, label_smoothing=ls)
            self.log("train_cls_ce", loss, prog_bar=True, on_step=True, on_epoch=True)
            return loss

        if self.task == "rnaatacmlm" and self.config.modality_fusion:
                mlm_logits, mlm_labels = self.modality_fusion(batch)
                if self.config.embedding_modality == "atac":
                    BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='mean')
                    mlm_labels_atac = mlm_labels.view(-1)
                    non_negative_two_indices = torch.where(mlm_labels_atac != -2)[0]
                    loss = BCEWithLogitsLoss(mlm_logits.view(-1)[non_negative_two_indices],
                                             mlm_labels_atac[non_negative_two_indices])
                else:
                    loss = F.mse_loss(mlm_logits, mlm_labels, reduction="mean")
        if self.config.perturbation:
            mlm_logits, mlm_labels = self.infer_rna_perturbation(**batch)
            loss = F.mse_loss(mlm_logits, mlm_labels, reduction="mean")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        """
        定义验证步骤，包括前向计算和验证损失计算
        """
        loss = None

        if self.task == "rnamlm":
            mlm_logits, mlm_labels = self.infer_rna_mlm(**batch)
            loss = F.mse_loss(mlm_logits, mlm_labels, reduction="mean")
        if self.task == "atacmlm":
            mlm_logits, mlm_labels = self.infer_atac_mlm(batch)
            BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='mean')
            mlm_labels = mlm_labels.view(-1)
            non_negative_two_indices = torch.where(mlm_labels != -2)[0]
            loss = BCEWithLogitsLoss(mlm_logits.view(-1)[non_negative_two_indices],
                                     mlm_labels[non_negative_two_indices])
        if self.task == "rnaatacmlm" and self.config.pretrain and self.config.both_pretrain:
                current_epoch = self.current_epoch

                if current_epoch < self.phase_change_epoch:
                    mlm_logits_atac, mlm_labels_atac = self.infer_rna_predict_atac_mlm(batch)
                    BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='mean')
                    mlm_labels_atac = mlm_labels_atac.view(-1)
                    non_negative_two_indices = torch.where(mlm_labels_atac != -2)[0]
                    loss1 = BCEWithLogitsLoss(mlm_logits_atac.view(-1)[non_negative_two_indices],
                                              mlm_labels_atac[non_negative_two_indices])
                    total_loss = loss1
                else:
                    mlm_logits_rna, mlm_labels_rna = self.infer_atac_predict_rna_mlm(batch)
                    loss2 = F.mse_loss(mlm_logits_rna, mlm_labels_rna, reduction="mean")
                    total_loss = loss2
                self.log("val_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                torch.cuda.empty_cache()
                return total_loss

        if self.task == "rnaatacmlm" and self.config.translation_to_atac:
            mlm_logits_atac, mlm_labels_atac = self.rna_translation_in_atac(batch)

            BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='mean')
            mlm_labels_flat = mlm_labels_atac.view(-1)
            valid_idx = torch.where(mlm_labels_flat != -2)[0]
            loss = BCEWithLogitsLoss(mlm_logits_atac.view(-1)[valid_idx],
                                     mlm_labels_flat[valid_idx])

        if self.task == "rnaatacmlm" and self.config.translation_to_rna:
            if self.config.pred_full:
                mlm_logits_rna, mlm_labels_rna = self.atac_translation_in_rna_full(batch)
                loss = F.mse_loss(mlm_logits_rna, mlm_labels_rna, reduction="mean")
                mlm_logits_np = mlm_logits_rna.detach().cpu().numpy()
                mlm_labels_np = mlm_labels_rna.detach().cpu().numpy()
                rmse_list = []
                pcc_list = []
                for cell_logits, cell_labels in zip(mlm_logits_np, mlm_labels_np):
                    cell_logits_flat = cell_logits.flatten()
                    cell_labels_flat = cell_labels.flatten()

                    cell_rmse = compute_rmse(cell_logits_flat, cell_labels_flat)
                    rmse_list.append(cell_rmse)

                    try:
                        cell_pcc, _ = pearsonr(cell_logits_flat, cell_labels_flat)
                    except Exception as e:
                        cell_pcc = float("nan")
                    pcc_list.append(cell_pcc)

                avg_rmse = np.mean(rmse_list)
                avg_pcc = np.nanmean(pcc_list)

            else:
                mlm_logits_rna, mlm_labels_rna = self.atac_translation_in_rna(batch)
                loss = F.mse_loss(mlm_logits_rna, mlm_labels_rna, reduction="mean")
                mlm_logits_np = mlm_logits_rna.detach().cpu().numpy()
                mlm_labels_np = mlm_labels_rna.detach().cpu().numpy()

                rmse_list = []
                pcc_list = []

                for cell_logits, cell_labels in zip(mlm_logits_np, mlm_labels_np):
                    cell_logits_flat = cell_logits.flatten()
                    cell_labels_flat = cell_labels.flatten()

                    cell_rmse = compute_rmse(cell_logits_flat, cell_labels_flat)
                    rmse_list.append(cell_rmse)
                    try:
                        cell_pcc, _ = pearsonr(cell_logits_flat, cell_labels_flat)
                    except Exception as e:
                        cell_pcc = float("nan")
                    pcc_list.append(cell_pcc)

                avg_rmse = np.mean(rmse_list)
                avg_pcc = np.nanmean(pcc_list)
        if self.task == "rnaatacmlm" and self.config.batch_correction:
            mlm_logits_rna, mlm_labels_rna = self.batch_correction(batch)
            loss = F.mse_loss(mlm_logits_rna, mlm_labels_rna, reduction="mean")

        if self.task == "rnaatacmlm" and self.config.cell_type_annotation:

                cell_type, cell_emb = self.cell_type_annotation(batch)

                logits = self.classifier(cell_emb)
                ce_loss = F.cross_entropy(logits, cell_type)

                preds = torch.argmax(logits, dim=1)
                acc = accuracy_score(cell_type.detach().cpu().numpy(), preds.detach().cpu().numpy())
                f1 = f1_score(cell_type.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro")
                acc_f1_sum = acc + f1
                self.log("val_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log("val_acc_f1_sum", acc_f1_sum, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                return {"loss": ce_loss, "val_acc": acc, "val_f1": f1}
        if self.task == "rnaatacmlm" and self.config.modality_fusion:
                mlm_logits, mlm_labels = self.modality_fusion(batch)
                if self.config.embedding_modality == "atac":
                    BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='mean')
                    mlm_labels_atac = mlm_labels.view(-1)
                    non_negative_two_indices = torch.where(mlm_labels_atac != -2)[0]
                    loss = BCEWithLogitsLoss(mlm_logits.view(-1)[non_negative_two_indices],
                                             mlm_labels_atac[non_negative_two_indices])
                else:
                    loss = F.mse_loss(mlm_logits, mlm_labels, reduction="mean")
        if self.config.perturbation:
            mlm_logits, mlm_labels = self.infer_rna_perturbation(**batch)
            loss = F.mse_loss(mlm_logits, mlm_labels, reduction="mean")
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.task == "rnaatacmlm" and self.config.translation_to_rna:
            self.log("val_pcc", avg_pcc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val_rmse", avg_rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        定义测试步骤，包括前向计算和测试损失计算
        """
        loss = None

        if self.task == "rnaatacmlm" and self.config.translation_to_atac:
            mlm_logits_atac, mlm_labels_atac = self.rna_translation_in_atac(batch)

            bce_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
            mlm_labels_flat = mlm_labels_atac.view(-1)
            valid_idx = torch.where(mlm_labels_flat != -2)[0]
            loss = bce_loss_fn(
                mlm_logits_atac.view(-1)[valid_idx],
                mlm_labels_flat[valid_idx]
            )


            preds_sigmoid = mlm_logits_atac.sigmoid().view(-1)[valid_idx].detach().cpu().numpy()
            labels_np = mlm_labels_flat[valid_idx].detach().cpu().numpy()


            batch_rmse = compute_rmse(preds_sigmoid, labels_np)
            try:
                batch_pcc, _ = pearsonr(preds_sigmoid, labels_np)
            except Exception:
                batch_pcc = float('nan')

            try:
                batch_auroc = compute_auroc(labels_np, preds_sigmoid)
            except Exception:
                batch_auroc = float('nan')

            self.test_step_metrics.append({
                "rmse": batch_rmse,
                "pcc": batch_pcc,
                "auroc": batch_auroc
            })

            result = {
                "loss": loss,
                "pred_logits": mlm_logits_atac.detach(),
                "labels": mlm_labels_atac.detach()
            }
            self.test_step_outputs.append(result)
            return result

            # ========== ATAC -> RNA ==========
        if self.task == "rnaatacmlm" and self.config.translation_to_rna:
            if self.config.pred_full:
                mlm_logits_rna, mlm_labels_rna = self.atac_translation_in_rna_full(batch)
            else:
                mlm_logits_rna, mlm_labels_rna = self.atac_translation_in_rna(batch)


            loss = F.mse_loss(mlm_logits_rna, mlm_labels_rna, reduction="mean")
            mlm_logits_np = mlm_logits_rna.detach().cpu().numpy()
            mlm_labels_np = mlm_labels_rna.detach().cpu().numpy()

            rmse_list = []
            pcc_list = []

            for cell_logits, cell_labels in zip(mlm_logits_np, mlm_labels_np):
                cell_logits_flat = cell_logits.flatten()
                cell_labels_flat = cell_labels.flatten()

                cell_rmse = compute_rmse(cell_logits_flat, cell_labels_flat)
                rmse_list.append(cell_rmse)

                try:
                    cell_pcc, _ = pearsonr(cell_logits_flat, cell_labels_flat)
                except Exception as e:
                    cell_pcc = float("nan")
                pcc_list.append(cell_pcc)

            avg_rmse = np.mean(rmse_list)
            avg_pcc = np.nanmean(pcc_list)

            self.test_step_metrics.append({
                "rmse": avg_rmse,
                "pcc": avg_pcc
            })

            result = {
                "loss": loss,
                "logits": mlm_logits_rna.detach(),
                "labels": mlm_labels_rna.detach()
            }
            self.test_step_outputs.append(result)
            return result

        if self.task == "rnaatacmlm" and self.config.cell_type_annotation:

                cell_type, cell_emb = self.cell_type_annotation(batch)

                logits = self.classifier(cell_emb)
                ce_loss = F.cross_entropy(logits, cell_type)

                preds = torch.argmax(logits, dim=1)
                acc = accuracy_score(cell_type.detach().cpu().numpy(), preds.detach().cpu().numpy())
                f1 = f1_score(cell_type.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro")

                self.log("test_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log("test_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

                if hasattr(self, "test_preds") and hasattr(self, "test_labels"):
                    self.test_preds.extend(preds.detach().cpu().tolist())
                    self.test_labels.extend(cell_type.detach().cpu().tolist())

                return {"loss": ce_loss, "test_acc": acc, "test_f1": f1}

    def on_test_epoch_end(self):


        if self.config.translation_to_atac:
            rmse_list = [m["rmse"] for m in self.test_step_metrics if "rmse" in m]
            pcc_list = [m["pcc"] for m in self.test_step_metrics if "pcc" in m]
            auroc_list = [m["auroc"] for m in self.test_step_metrics if "auroc" in m]

            avg_rmse = float(np.nanmean(rmse_list)) if rmse_list else float('nan')
            avg_pcc = float(np.nanmean(pcc_list)) if pcc_list else float('nan')
            avg_auroc = float(np.nanmean(auroc_list)) if auroc_list else float('nan')

            self.log("test_rmse", avg_rmse, on_epoch=True, prog_bar=True, logger=True)
            self.log("test_pcc", avg_pcc, on_epoch=True, prog_bar=True, logger=True)
            self.log("test_auroc", avg_auroc, on_epoch=True, prog_bar=True, logger=True)

            if self.trainer.is_global_zero:

                output_dir = self.config.output_dir if hasattr(self.config, "output_dir") else "."
                os.makedirs(output_dir, exist_ok=True)


                metrics_str = (
                    f"test_rmse: {avg_rmse:.4f}\n"
                    f"test_auroc: {avg_auroc:.4f}\n"
                    f"test_pcc: {avg_pcc:.4f}\n"
                    # f"predicted_atac.npy: {save_path}\n"
                )
                metrics_save_path = os.path.join(output_dir, "CLM-X_test_atac_metrics.txt")
                with open(metrics_save_path, "w") as f:
                    f.write(metrics_str)

        elif self.config.translation_to_rna:
            rmse_list = [m["rmse"] for m in self.test_step_metrics if "rmse" in m]
            pcc_list = [m["pcc"] for m in self.test_step_metrics if "pcc" in m]

            avg_rmse = float(np.nanmean(rmse_list)) if rmse_list else float('nan')
            avg_pcc = float(np.nanmean(pcc_list)) if pcc_list else float('nan')

            self.log("test_rmse", avg_rmse, on_epoch=True, prog_bar=True, logger=True)
            self.log("test_pcc", avg_pcc, on_epoch=True, prog_bar=True, logger=True)

            local_logits = torch.cat([x["logits"] for x in self.test_step_outputs], dim=0)

            local_num_samples = local_logits.shape[0]
            local_count_tensor = torch.tensor([local_num_samples], device=local_logits.device)

            gathered_logits = self.all_gather(local_logits)
            gathered_counts = self.all_gather(local_count_tensor)


            all_logits = gathered_logits.reshape(-1, gathered_logits.shape[-1])

            total_samples = int(gathered_counts.sum().item())

            all_logits = all_logits[:total_samples]

            if self.trainer.is_global_zero:
                output_dir = self.config.output_dir if hasattr(self.config, "output_dir") else "."
                os.makedirs(output_dir, exist_ok=True)

                metrics_str = (
                    f"test_rmse: {avg_rmse:.4f}\n"
                    f"test_pcc: {avg_pcc:.4f}\n"
                )
                metrics_save_path = os.path.join(output_dir, "CLM-X_test_rna_metrics.txt")
                with open(metrics_save_path, "w") as f:
                    f.write(metrics_str)

                # np.save(os.path.join(output_dir, "CLM-X_predicted_rna.npy"), all_logits.cpu().numpy())

        if self.task == "rnaatacmlm" and self.config.cell_type_annotation:
            if len(self.test_labels) == 0 or len(self.test_preds) == 0:
                return

            preds_tensor = torch.tensor(self.test_preds, dtype=torch.long, device=self.device)
            labels_tensor = torch.tensor(self.test_labels, dtype=torch.long, device=self.device)

            gathered_preds = self.all_gather(preds_tensor)
            gathered_labels = self.all_gather(labels_tensor)

            test_preds = gathered_preds.flatten().cpu().numpy().tolist()
            test_labels = gathered_labels.flatten().cpu().numpy().tolist()

            labels_order = list(range(self.config.num_classes))
            cm = confusion_matrix(test_labels, test_preds, labels=labels_order)

            if hasattr(self, "idx_to_cell_type"):
                tick_labels = [self.idx_to_cell_type[i] for i in labels_order]
            else:
                tick_labels = self.config.cell_types

            col_sums = np.sum(cm, axis=0)
            col_sums_reshaped = col_sums.reshape(1, -1)
            prob_matrix = np.divide(
                cm.astype(np.float64),
                col_sums_reshaped,
                out=np.zeros_like(cm, dtype=np.float64),
                where=(col_sums_reshaped != 0)
            )

            with plt.rc_context({
                "font.size": 14,
                "svg.fonttype": "none",
                "font.family": "Arial",
                "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                "axes.unicode_minus": False
            }):
                fig, ax = plt.subplots(figsize=(12, 12))

                heatmap = sns.heatmap(
                    prob_matrix,
                    square=True,
                    cmap="OrRd",
                    annot=False,
                    xticklabels=tick_labels,
                    yticklabels=tick_labels,
                    cbar=False,
                    ax=ax
                )

                ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=18)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)

                cbar = fig.colorbar(heatmap.collections[0], cax=cax, ticks=np.linspace(0, 1, 5))
                cbar.ax.set_yticklabels([f"{tick:.2f}" for tick in np.linspace(0, 1, 5)])

                ax.set_title("Confusion Matrix", fontsize=18)

                fig.tight_layout()
                from pathlib import Path

                if self.trainer.is_global_zero:
                    os.makedirs(self.config.output_dir, exist_ok=True)
                    tag = getattr(self, "test_tag", "test")
                    out_dir = Path(getattr(self, "test_out_dir", "./test_outputs"))
                    os.makedirs(out_dir, exist_ok=True)
                    cm_filepath = os.path.join(out_dir, "confusion_matrix.svg")
                    fig.savefig(cm_filepath, format="svg", bbox_inches="tight")

                    cm_data_filepath = os.path.join(out_dir, "confusion_matrix_data.csv")
                    with open(cm_data_filepath, mode="w", newline="") as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow([""] + tick_labels)
                        for i, row in enumerate(cm):
                            writer.writerow([tick_labels[i]] + row.tolist())

                    overall_acc = accuracy_score(test_labels, test_preds)
                    overall_f1 = f1_score(test_labels, test_preds, average="macro")

                    metrics_filepath = os.path.join(out_dir, "test_metrics.csv")
                    with open(metrics_filepath, mode="w", newline="") as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(["metric", "value"])
                        writer.writerow(["accuracy", overall_acc])
                        writer.writerow(["f1_score", overall_f1])

                plt.close(fig)


            self.test_step_outputs.clear()
            self.test_step_metrics.clear()

    def configure_optimizers(self):
        """
        configure optimizers
        """
        return get_optimizers_for_lightning(
                self.parameters(),
                self.config.learning_rate,
                self.config.adam_weight_decay,
                self.config.num_warmup_steps,
                self.config.max_steps,
            )
    
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        beit_norms = grad_norm(self.beit3, norm_type=2)
        self.log_dict(beit_norms)
        # mlm_norms = grad_norm(self.mlm_scorer, norm_type=2)
        # self.log_dict(mlm_norms)

