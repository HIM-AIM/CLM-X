import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import dotmap
import lightning.pytorch as pl
import numpy as np
import torch
from datasets import Dataset as HFDataset
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, random_split

os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "36000"

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cellstory.logger import init_logger
from configs.config_pretrain import ex
from datamodules.datasets_pt import create_dataset_by_split
from models.beitv3_pl_value import BeitForPretrain as BeitForPretrainValue


CHUNK_RE = re.compile(r"^(?P<start>\d+)_(?P<end>\d+)\.npy$")


class CombinedRNATACDataset(Dataset):
    def __init__(self, rna_dataset: HFDataset, atac_mm: np.memmap):
        if len(rna_dataset) != atac_mm.shape[0]:
            raise ValueError(
                f"RNA length ({len(rna_dataset)}) != ATAC length ({atac_mm.shape[0]})."
            )
        self.rna_dataset = rna_dataset
        self.atac_mm = atac_mm

    def __len__(self):
        return len(self.rna_dataset)

    def __getitem__(self, idx):
        rna_sample = self.rna_dataset[idx]
        atac_tensor = torch.from_numpy(self.atac_mm[idx]).to(torch.float32)
        return {"rna": rna_sample, "atac": atac_tensor}


def _infer_hf_vocab_size(ds: HFDataset, field: str = "text", vocab_key: str = "vocabulary"):
    try:
        return len(ds.features[field].feature[vocab_key])
    except Exception:
        return None


def load_rna_dataset(path: str, logger):
    logger.info(f"Loading RNA dataset: {path}")
    ds = HFDataset.load_from_disk(path)
    vocab_size = _infer_hf_vocab_size(ds)
    logger.info("RNA dataset loaded.")
    return ds, vocab_size


def _list_atac_chunks(dataset_path: Path):
    chunks = []
    for p in dataset_path.glob("*.npy"):
        m = CHUNK_RE.match(p.name)
        if not m:
            continue
        start, end = int(m.group("start")), int(m.group("end"))
        if end <= start:
            raise ValueError(f"Invalid chunk range in filename: {p.name}")
        chunks.append((start, end, p))
    chunks.sort(key=lambda x: (x[0], x[1]))
    return chunks


def _try_acquire_lock(lock_path: Path):
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        return fd
    except FileExistsError:
        return None


def load_atac_memmap(dataset_path: str, context_length: int, peak_length: int, logger):
    dataset_path = Path(dataset_path)
    logger.info(f"Loading ATAC dataset: {dataset_path}")

    chunks = _list_atac_chunks(dataset_path)
    if not chunks:
        raise ValueError("No ATAC chunk files found (expected filenames like '0_1000.npy').")

    total = max(end for _, end, _ in chunks)
    bin_path = dataset_path / "large_data.bin"
    lock_path = dataset_path / "large_data.bin.lock"

    if not bin_path.exists():
        fd = _try_acquire_lock(lock_path)
        if fd is not None:
            try:
                logger.info(f"Building memmap: {bin_path} (n={total})")
                mm = np.memmap(
                    bin_path,
                    dtype=np.int8,
                    mode="w+",
                    shape=(total, context_length, peak_length),
                )
                for start, end, p in chunks:
                    arr = np.load(p, mmap_mode=None)
                    if arr.shape[0] != (end - start):
                        raise ValueError(
                            f"Chunk {p.name} has {arr.shape[0]} rows, expected {end - start}."
                        )
                    mm[start:end] = arr
                mm.flush()
                del mm
                logger.info("Memmap built.")
            finally:
                os.close(fd)
                try:
                    lock_path.unlink(missing_ok=True)
                except TypeError:
                    if lock_path.exists():
                        lock_path.unlink()
        else:
            logger.info("Memmap build in progress by another process; waiting...")
            while not bin_path.exists():
                time.sleep(1.0)

    mm = np.memmap(
        bin_path,
        dtype=np.int8,
        mode="r",
        shape=(total, context_length, peak_length),
    )

    gene_tokens_path = dataset_path / "gene_tokens.npy"
    if not gene_tokens_path.exists():
        raise FileNotFoundError(f"Missing gene_tokens.npy: {gene_tokens_path}")
    gene_tokens = np.load(gene_tokens_path)

    logger.info("ATAC dataset loaded.")
    return mm, gene_tokens


def make_dataloader(dataset, *, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,
    )


def create_rna_atac_dataloaders(args, logger, val_ratio: float = 0.1):
    rna_ds, rna_vocab_size = load_rna_dataset(args.rna_dataset_path, logger)
    atac_mm, gene_tokens = load_atac_memmap(
        args.atac_dataset_path, args.context_length, args.peak_length, logger
    )

    ds = CombinedRNATACDataset(rna_ds, atac_mm)

    n_total = len(ds)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    logger.info(f"Split: train={n_train}, val={n_val}")

    gen = torch.Generator().manual_seed(int(getattr(args, "seed", 42)))
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

    train_loader = make_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    return train_loader, val_loader, gene_tokens, rna_vocab_size, len(gene_tokens)


def _unwrap_state_dict(ckpt_obj):
    return ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj


def filter_and_rename_keys_atac(state_dict):
    out = {}
    for k, v in state_dict.items():
        if (
            "beit3.atac_embed" in k
            or "beit3.atac_value_embed" in k
            or "beit3.atac_norm" in k
        ):
            out[k] = v
            continue

        if "encoder.layers" in k and "ffn" in k:
            if ".ffn.rna." not in k:
                out[k] = v
            continue

        if "encoder.layers" in k and (".self_attn." in k or ".layer_norm." in k):
            out[k] = v
            continue

        if k.startswith("atac_mlm_scorer"):
            out[k.replace("atac_mlm_scorer", "mix_pred_atac_mlm_scorer")] = v
            continue

    return out


def filter_and_rename_keys_rna(state_dict):
    out = {}
    for k, v in state_dict.items():
        if "beit3.rna_embed" in k:
            out[k] = v
            continue

        if "beit3.value_embed" in k:
            out[k.replace("beit3.value_embed", "beit3.rna_value_embed")] = v
            continue

        if "beit3.norm" in k and "encoder.layers" not in k:
            out[k.replace("beit3.norm", "beit3.rna_norm")] = v
            continue

        if "encoder.layers" in k and ".ffn.rna." in k:
            out[k] = v
            continue

        if k.startswith("mlm_scorer"):
            out[k.replace("mlm_scorer", "mix_pred_rna_mlm_scorer")] = v
            continue

    return out


def load_fusion_model(model, args, logger):
    model_dict = model.state_dict()

    atac_path = getattr(args, "model_load_path_atac", None)
    if atac_path and os.path.exists(atac_path):
        logger.info(f"Loading ATAC checkpoint: {atac_path}")
        ckpt = _unwrap_state_dict(torch.load(atac_path, map_location="cpu"))
        model_dict.update(filter_and_rename_keys_atac(ckpt))

    rna_path = getattr(args, "model_load_path_rna", None)
    if rna_path and os.path.exists(rna_path):
        logger.info(f"Loading RNA checkpoint: {rna_path}")
        ckpt = _unwrap_state_dict(torch.load(rna_path, map_location="cpu"))
        model_dict.update(filter_and_rename_keys_rna(ckpt))

    missing, unexpected = model.load_state_dict(model_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")
    logger.info("Fusion weights loaded.")
    return model


@ex.automain
def main(_config):
    args_ = dotmap.DotMap(_config)
    pl.seed_everything(args_.seed, workers=True)

    os.makedirs(args_.log_dir, exist_ok=True)
    os.makedirs(args_.dirpath, exist_ok=True)

    logger = init_logger(args_)

    now = datetime.now()
    run_name = f"{args_.exp_name}-{now:%Y%m%d-%H%M}"

    wandb_logger = WandbLogger(
        name=run_name,
        save_dir=args_.log_dir,
        project=args_.project_name,
        log_model=True,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=args_.dirpath,
        filename=f"{args_.exp_name}-epoch{{epoch:02d}}-val{{val_loss:.3f}}",
        every_n_epochs=1,
        save_top_k=-1,
        monitor="val_loss",
        mode="min",
        save_last=True,
        verbose=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_cb, lr_cb]

    grad_steps = args_.grad_steps
    max_steps = args_.max_steps if args_.max_steps is not None else None

    logger.info("Loading dataset")
    is_train = args_.model_task in {"for_finetune", "finetune"}

    if args_.task == "rnaatacmlm":
        train_dataloader, val_dataloader, gene_tokens, rna_vocab_size, atac_vocab_size = (
            create_rna_atac_dataloaders(args_, logger)
        )
    else:
        train_dataloader, val_dataloader, gene_tokens, rna_vocab_size, atac_vocab_size = (
            create_dataset_by_split(args_, is_train=is_train)
        )

    args_.gene_tokens = gene_tokens
    args_.rna_vocab_size = getattr(args_, "rna_vocab_size", None) or rna_vocab_size or 60668
    args_.atac_vocab_size = getattr(args_, "atac_vocab_size", None) or atac_vocab_size or 2002

    logger.info(f"Vocab size: RNA={args_.rna_vocab_size}, ATAC={args_.atac_vocab_size}")

    logger.info("Loading model")
    MODEL_CLS = BeitForPretrainValue

    if (
        args_.task == "rnamlm"
        and args_.model_load_path_rna
        and os.path.exists(args_.model_load_path_rna)
    ):
        model = MODEL_CLS.load_from_checkpoint(
            args_.model_load_path_rna, map_location="cpu", config=args_
        )
    elif (
        args_.task == "atacmlm"
        and args_.model_load_path_atac
        and os.path.exists(args_.model_load_path_atac)
    ):
        model = MODEL_CLS.load_from_checkpoint(
            args_.model_load_path_atac, map_location="cpu", config=args_
        )
    elif args_.task == "rnaatacmlm":
        model = load_fusion_model(MODEL_CLS(args_), args_, logger)
    else:
        model = MODEL_CLS(args_)

    strategy = "fsdp" if args_.use_sharded_training else "ddp_find_unused_parameters_true"
    logger.info(f"Strategy: {strategy}")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args_.num_gpus,
        num_nodes=args_.num_nodes,
        precision=args_.precision,
        strategy=strategy,
        benchmark=True,
        deterministic=True,
        max_epochs=args_.max_epoch,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=wandb_logger,
        use_distributed_sampler=True,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        fast_dev_run=args_.fast_dev_run,
        check_val_every_n_epoch=1,
        val_check_interval=None,
        gradient_clip_val=0.02,
    )

    logger.info("Training started")
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args_.resume_from_checkpoint,
    )
    logger.info("Training finished")
