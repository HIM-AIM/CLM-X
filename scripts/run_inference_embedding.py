import os
import sys
import dotmap

# set gpu number
os.environ["WANDB_MODE"] = "disabled"
import pickle
import lightning as pl
from cellstory.utils import get_obs
import scanpy as sc
from tqdm import tqdm
import re
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import Dataset as HFDataset
import numpy as np
from pathlib import Path
# add code to pythonpath
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.beitv3_pl_value import BeitForPretrain as BeitForPretrain_Value
from cellstory.inference.inference_rna import append_to_obsm, generate_rna_metrics
from cellstory.utils import convert_to_path
from cellstory.logger import init_logger
# import experiment
from configs.config_eval import ex


def model_infer_rna_atac(model, dataloader, gene_tokens, rna_mask_ratio, atac_mask_ratio, pad_id,
                         context_length, embedding_type):
    rna_atac_reprs = []
    atac_reprs = []
    rna_reprs = []

    device = next(model.parameters()).device

    gene_tokens_tensor = torch.from_numpy(gene_tokens).to(device)

    for batch in tqdm(dataloader):
        gene_tokens_rna = batch['rna']['gene_tokens'].to(device)
        target_values_rna = batch['rna']['target_values'].to(device)
        padding_mask_rna = batch['rna']['padding_mask'].to(device)

        gene_tokens_atac = gene_tokens_tensor.repeat(len(batch['atac']), 1, 1).squeeze()
        padding_mask_atac = gene_tokens_atac.eq(pad_id).int()

        original_tokenized_batch = batch['atac'].clone().to(device)
        context_length = context_length
        n_mask = int(context_length * atac_mask_ratio)
        random_integers = torch.randint(0, context_length, (n_mask,)).to(device)
        original_tokenized_batch[:, random_integers] = -1
        mask_pos_atac = torch.zeros_like(original_tokenized_batch, dtype=torch.int).to(device)
        mask_pos_atac[:, random_integers] = 1

        with torch.no_grad():
            outputs = model.beit3(
                atac_tokens=gene_tokens_atac,
                rna_tokens=gene_tokens_rna,
                values_atac=original_tokenized_batch.float(),
                values_rna=target_values_rna,
                atac_padding_position=padding_mask_atac,
                rna_padding_position=padding_mask_rna,
                attn_mask=None,
            )
            rna_atac_feats = outputs["encoder_out"]
            atac_feats = outputs["encoder_out"][:, :len(gene_tokens_atac[1])]
            rna_feats = outputs["encoder_out"][:, len(gene_tokens_atac[1]):]

            if embedding_type == "cls":
                # Process ATAC and RNA features
                atac_features = atac_feats[:, 0, :]
                rna_features = rna_feats[:, 0, :]

                combined_features = atac_features
                rna_atac_reprs.extend(combined_features.cpu())
                atac_reprs.extend(atac_features.cpu())
                rna_reprs.extend(rna_features.cpu())
            elif embedding_type == "avgpool":
                # Function to calculate average pooling
                def avg_pool(feats, padding_mask):
                    feats_without_cls = feats[:, 1:, :]
                    mask_without_cls = padding_mask[:, 1:]
                    mask_without_cls_ = 1 - mask_without_cls
                    mask_without_cls_ = torch.unsqueeze(mask_without_cls_, dim=2)
                    repr_wopadding = feats_without_cls * mask_without_cls_
                    avg_repr = torch.sum(repr_wopadding, dim=1) / torch.unsqueeze(
                        torch.sum(1 - mask_without_cls, dim=1), dim=1
                    )

                    return avg_repr / avg_repr.norm(dim=-1, keepdim=True)

                rna_atac_features = avg_pool(rna_atac_feats, torch.cat([padding_mask_atac, padding_mask_rna], dim=1))
                atac_features = avg_pool(atac_feats, padding_mask_atac)
                rna_features = avg_pool(rna_feats, padding_mask_rna)

                rna_atac_reprs.extend(rna_atac_features.cpu())
                atac_reprs.extend(atac_features.cpu())
                rna_reprs.extend(rna_features.cpu())

        stacked_rna_atac_embeddings = torch.stack(rna_atac_reprs, dim=0).numpy()
        stacked_atac_embeddings = torch.stack(atac_reprs, dim=0).numpy()
        stacked_rna_embeddings = torch.stack(rna_reprs, dim=0).numpy()

    return stacked_rna_atac_embeddings, stacked_atac_embeddings, stacked_rna_embeddings


def rna_atac_inference(args):
    # init logger
    logger = init_logger(args)

    class CombinedRNATACDataset(Dataset):
        def __init__(self, rna_dataset: HFDataset, atac_mm: np.memmap):
            if len(rna_dataset) != atac_mm.shape[0]:
                raise ValueError(
                    f"RNA dataset length ({len(rna_dataset)}) does not match ATAC dataset length ({atac_mm.shape[0]})."
                )
            self.rna_dataset = rna_dataset
            self.atac_mm = atac_mm

        def __len__(self):
            return len(self.rna_dataset)

        def __getitem__(self, idx):
            rna_sample = self.rna_dataset[idx]
            atac_sample = self.atac_mm[idx]
            atac_tensor = torch.from_numpy(atac_sample).type(torch.float32)
            return {
                "rna": rna_sample,
                "atac": atac_tensor,
            }

    def load_rna_dataset_new(args):
        rna_dataset_path = args.rna_dataset_path
        logger.info(f"Loading RNA dataset: {rna_dataset_path}")
        rna_dataset = HFDataset.load_from_disk(rna_dataset_path)

        if "text" in rna_dataset.features and "vocabulary" in rna_dataset.features["text"].feature:
            rna_vocab_size = len(rna_dataset.features["text"].feature["vocabulary"])
        else:
            rna_vocab_size = None

        atac_vocab_size = None

        logger.info("RNA dataset loaded.")
        return rna_dataset, rna_vocab_size, atac_vocab_size

    def sort_by_index(file_path: str):
        match = re.search(r"(\d+)_(\d+)\.npy$", file_path)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            return (start, end)
        else:
            return (float("inf"), float("inf"))

    def load_numpy_dataset_new(args):
        dataset_path = args.atac_dataset_path
        context_length = args.context_length
        peak_length = args.peak_length

        logger.info(f"Starting to load ATAC dataset: {dataset_path}")

        all_npy_files = [str(f) for f in Path(dataset_path).glob("*.npy")]
        data_npy_files = [f for f in all_npy_files if os.path.basename(f) != "gene_tokens.npy"]

        if not data_npy_files:
            raise ValueError("No data chunk files found matching the expected pattern (e.g., '0_1000.npy').")

        sorted_npy_files = sorted(data_npy_files, key=sort_by_index)

        last_file = sorted_npy_files[-1]
        match = re.search(r"(\d+)_(\d+)\.npy$", last_file)
        if not match:
            raise ValueError(f"Invalid filename format: {last_file}")
        end = int(match.group(2))

        large_data_path = os.path.join(dataset_path, "large_data.bin")

        if not os.path.exists(large_data_path):
            logger.info(f"Creating memmap file: {large_data_path}")
            mm = np.memmap(
                large_data_path,
                dtype=np.int8,
                mode="w+",
                shape=(end, context_length, peak_length),
            )
            current_idx = 0
            for file in sorted_npy_files:
                data_chunk = np.load(file)
                num_samples = data_chunk.shape[0]
                mm[current_idx : current_idx + num_samples] = data_chunk
                current_idx += num_samples
            del mm
            logger.info("Memmap file created.")
        else:
            logger.info(f"Memmap file already exists: {large_data_path}")

        mm = np.memmap(
            large_data_path,
            dtype=np.int8,
            mode="r",
            shape=(end, context_length, peak_length),
        )

        gene_tokens_path = os.path.join(dataset_path, "gene_tokens.npy")
        if not os.path.exists(gene_tokens_path):
            raise FileNotFoundError(f"gene_tokens.npy not found: {gene_tokens_path}")

        gene_tokens = np.load(gene_tokens_path)

        if args.cell_type_annotation:
            with open(f"{dataset_path}/id2type.pkl", "rb") as file:
                cell_type_label = pickle.load(file)
            args.cell_type_number = len(cell_type_label.keys())

        if args.batch_correction:
            with open(f"{dataset_path}/id2type.pkl", "rb") as file:
                batch_label = pickle.load(file)
            args.batch_number = len(batch_label.keys())

        logger.info("ATAC dataset loaded.")
        return mm, gene_tokens

    def create_dataloader_new(dataset, is_train, batch_size, num_workers, pin_mem, dist_eval):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=is_train,
        )

    def create_dataset_by_split_new(args, is_train=True):
        logger.info("Preparing to load data")

        rna_dataset, rna_vocab_size, atac_vocab_size = load_rna_dataset_new(args)
        atac_mm, gene_tokens = load_numpy_dataset_new(args)

        combined_dataset = CombinedRNATACDataset(rna_dataset, atac_mm)

        logger.info("Creating DataLoader")
        dataloader = create_dataloader_new(
            combined_dataset,
            is_train=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_mem=args.pin_mem,
            dist_eval=args.dist_eval,
        )

        atac_vocab_size = len(gene_tokens)
        adata_obs = get_obs(args.rna_h5ad)
        return adata_obs, dataloader, gene_tokens, rna_vocab_size, atac_vocab_size

    adata_obs, dataloader, gene_tokens, rna_vocab_size, atac_vocab_size = create_dataset_by_split_new(
        args, is_train=False
    )

    args.rna_vocab_size = 60668
    args.atac_vocab_size = 2002
    logger.info(f"vocab size: RNA: {args.rna_vocab_size}, ATAC: {args.atac_vocab_size}")

    # load model checkpoint
    logger.info("loading the model parameters")
    model = BeitForPretrain_Value.load_from_checkpoint(args.model_load_path, map_location="cpu", config=args, strict=False)
    model = model.cuda()
    model.eval()
    # inference from dataloader
    if args.atac_rna_cls:
        rna_atac_model_embed, atac_model_embed, rna_model_embed = model_infer_rna_atac(
            model,
            dataloader,
            gene_tokens,
            rna_mask_ratio=args.rna_mask_ratio,
            atac_mask_ratio=args.atac_mask_ratio,
            pad_id=args.pad_id,
            context_length=args.context_length,
            embedding_type=args.embedding_type
        )

        output_dir = args.output_dir

        os.makedirs(output_dir, exist_ok=True)

        file_data = {
            "CLM-X_embeddings.csv": rna_atac_model_embed,
        }

        for filename, data in file_data.items():
            save_path = os.path.join(output_dir, filename)
            np.savetxt(save_path, data, delimiter=',')

        print("finish")

    # adata = sc.read_h5ad(args.rna_h5ad)
    # adata.layers[args.raw_layer_key] = adata.X.copy()


@ex.automain
def main(_config):
    args_ = dotmap.DotMap(_config)
    pl.seed_everything(args_.seed)

    logger = init_logger(args_)

    args_.dirpath = convert_to_path(args_.dirpath)

    if not os.path.exists(args_.dirpath):
        os.makedirs(args_.dirpath)
    if not os.path.exists(args_.log_dir):
        os.makedirs(args_.log_dir, exist_ok=True)

    if args_.task == "rnaatacmlm":
        args_.rna_h5ad = convert_to_path(args_.rna_h5ad)
        logger.info("Start inference for RNA-ATAC")
        inferred_adata = rna_atac_inference(args_)
        logger.info("Finish inference for RNA-ATAC")
        logger.info("Start calculating metrics for RNA-ATAC")
        fig, metric_df = generate_rna_metrics(args_, inferred_adata)
        logger.info("Finish calculating metrics for RNA-ATAC")
