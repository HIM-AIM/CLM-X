import os

# Environment knobs (must be set before importing wandb/lightning in some setups)
os.environ.setdefault("TORCH_DISTRIBUTED_TIMEOUT", "36000")
# os.environ.setdefault("WANDB_MODE", "disabled")

import re
import sys
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import Dataset as HFDataset

import lightning as pl
from dotmap import DotMap
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

# Make project imports work when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.beitv3_pl_value import BeitForPretrain as BeitForPretrain_Value
from cellstory.logger import init_logger
from configs.config_finetune import ex


_CHUNK_RE = re.compile(r"(?P<start>\d+)_(?P<end>\d+)\.npy$")


class CombinedRNATACDataset(Dataset):
    """Pairs one RNA sample (HF Dataset row) with one ATAC sample (memmap row)."""

    def __init__(self, rna_dataset: HFDataset, atac_mm: np.memmap):
        if len(rna_dataset) != atac_mm.shape[0]:
            raise ValueError(
                f"RNA length ({len(rna_dataset)}) != ATAC length ({atac_mm.shape[0]})."
            )
        self.rna_dataset = rna_dataset
        self.atac_mm = atac_mm

    def __len__(self) -> int:
        return len(self.rna_dataset)

    def __getitem__(self, idx: int):
        rna_sample = self.rna_dataset[idx]
        atac = torch.as_tensor(self.atac_mm[idx], dtype=torch.float32)
        return {"rna": rna_sample, "atac": atac}


def _list_atac_chunks(dataset_dir: Path) -> list[tuple[int, int, Path]]:
    chunks: list[tuple[int, int, Path]] = []
    for p in dataset_dir.glob("*.npy"):
        if p.name == "gene_tokens.npy":
            continue
        m = _CHUNK_RE.search(p.name)
        if not m:
            continue
        chunks.append((int(m["start"]), int(m["end"]), p))

    if not chunks:
        raise ValueError(
            f"No ATAC chunk files found under {dataset_dir} "
            f"(expected names ending with like '0_1000.npy')."
        )

    chunks.sort(key=lambda x: (x[0], x[1]))
    return chunks


def load_atac_memmap_and_tokens(args: DotMap, logger):
    """
    Build a single consolidated memmap once (large_data.bin) to avoid loading many .npy files during training.
    """
    dataset_dir = Path(args.atac_dataset_path)
    context_length = int(args.context_length) + 1
    peak_length = int(args.peak_length)

    chunks = _list_atac_chunks(dataset_dir)
    end_max = max(end for _, end, _ in chunks)

    bin_path = dataset_dir / "large_data.bin"
    shape = (end_max, context_length, peak_length)

    if not bin_path.exists():
        logger.info(f"Creating ATAC memmap: {bin_path} (shape={shape}, dtype=int8)")
        mm = np.memmap(bin_path, dtype=np.int8, mode="w+", shape=shape)

        for start, end, p in chunks:
            arr = np.load(p, mmap_mode="r")
            expected = end - start
            if arr.shape[0] != expected:
                raise ValueError(
                    f"Chunk {p.name} rows={arr.shape[0]} but expected {expected} from filename range."
                )
            mm[start:end] = arr

        mm.flush()
        del mm
        logger.info("ATAC memmap created.")
    else:
        logger.info(f"Using existing ATAC memmap: {bin_path}")

    atac_mm = np.memmap(bin_path, dtype=np.int8, mode="r", shape=shape)

    gene_tokens_path = dataset_dir / "gene_tokens.npy"
    if not gene_tokens_path.exists():
        raise FileNotFoundError(f"Missing gene_tokens.npy: {gene_tokens_path}")
    gene_tokens = np.load(gene_tokens_path)

    # Optional label metadata
    if getattr(args, "cell_type_annotation", False):
        id2type_path = dataset_dir / "id2type.pkl"
        with open(id2type_path, "rb") as f:
            id2type = pickle.load(f)
        args.cell_type_number = len(id2type)

    if getattr(args, "batch_correction", False):
        id2batch_path = dataset_dir / "id2type.pkl"  # keep original behavior
        with open(id2batch_path, "rb") as f:
            id2batch = pickle.load(f)
        args.batch_number = len(id2batch)

    return atac_mm, gene_tokens


def build_dataloaders(args: DotMap, logger):
    """Load data, create a paired dataset, and split into train/val."""
    logger.info(f"Loading RNA dataset from: {args.rna_dataset_path}")
    rna_ds = HFDataset.load_from_disk(args.rna_dataset_path)

    logger.info(f"Loading ATAC dataset from: {args.atac_dataset_path}")
    atac_mm, gene_tokens = load_atac_memmap_and_tokens(args, logger)

    dataset = CombinedRNATACDataset(rna_ds, atac_mm)

    train_ratio = float(0.95)
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    logger.info(f"Split: train={train_size}, val={val_size}")

    gen = torch.Generator().manual_seed(int(42))
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return train_loader, val_loader, gene_tokens


def build_callbacks(args: DotMap):
    """Checkpointing + LR logging (only for finetune to match your original logic)."""
    if getattr(args, "model_task", None) != "finetune":
        return []

    best_ckpt = ModelCheckpoint(
        dirpath=args.dirpath,
        filename=f"{args.exp_name}_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        verbose=True,
    )


    return [best_ckpt, LearningRateMonitor(logging_interval="step")]


def set_trainable_params(model: torch.nn.Module, task: str, report_path: Path):
    """Freeze/unfreeze parameters based on task and write a report to disk."""
    for name, p in model.named_parameters():
        if task == "rnamlm":
            p.requires_grad = ".atac" not in name
        elif task == "atacmlm":
            p.requires_grad = ".rna" not in name
        else:  # rnaatacmlm (and any other task): train everything
            p.requires_grad = True

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        for name, p in model.named_parameters():
            f.write(f"{name}, {p.requires_grad}\n")


def build_strategy(args: DotMap):
    """Select FSDP or DDP (find_unused_parameters=True) with a configurable timeout."""
    if getattr(args, "use_sharded_training", False):
        return "fsdp"

    timeout_s = int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT", "36000"))
    return DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=timeout_s))


@ex.automain
def main(_config):
    args = DotMap(_config)
    pl.seed_everything(args.seed)

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.dirpath).mkdir(parents=True, exist_ok=True)

    logger = init_logger(args)

    run_name = f"{args.exp_name}-{datetime.now():%Y%m%d-%H%M}"
    wandb_logger = WandbLogger(
        name=run_name,
        save_dir=args.log_dir,
        project=args.project_name,
        log_model=True,
    )

    callbacks = build_callbacks(args)

    logger.info("Building dataloaders...")
    train_loader, val_loader, gene_tokens = build_dataloaders(args, logger)

    # Model expects these fields in config
    args.rna_vocab_size = int(60668)
    args.atac_vocab_size = int(2002)
    args.gene_tokens = np.asarray(gene_tokens, dtype=np.int32)

    logger.info(f"Vocab size: RNA={args.rna_vocab_size}, ATAC={args.atac_vocab_size}")

    if getattr(args, "model_load_path", None) and Path(args.model_load_path).is_file():
        logger.info(f"Loading model checkpoint: {args.model_load_path}")
        model = BeitForPretrain_Value.load_from_checkpoint(
            args.model_load_path,
            map_location="cpu",
            config=args,
            strict=False,
        )
    else:
        model = BeitForPretrain_Value(args)

    set_trainable_params(
        model,
        task=str(getattr(args, "task", "")),
        report_path=Path(args.dirpath) / "training_params_name_pair.txt",
    )

    strategy = build_strategy(args)
    logger.info(f"Distributed strategy: {strategy}")

    max_steps = args.max_steps if getattr(args, "max_steps", None) is not None else -1

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        precision=args.precision,
        strategy=strategy,
        benchmark=True,
        deterministic=True,
        max_epochs=args.max_epoch,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=wandb_logger,
        use_distributed_sampler=True,
        accumulate_grad_batches=args.grad_steps,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
        check_val_every_n_epoch=1,
        gradient_clip_val=0.05,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=getattr(args, "resume_from_checkpoint", None),
    )

    logger.info("Training finished.")