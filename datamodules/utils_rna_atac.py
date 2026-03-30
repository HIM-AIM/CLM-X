from __future__ import annotations

import os
import re
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader, random_split
from dotmap import DotMap
from lightning.pytorch.strategies import DDPStrategy


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


def make_run_name(exp_name: str, now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
    return f"{exp_name}-{now:%Y%m%d-%H%M}"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _log(logger, msg: str):
    if logger is None:
        return
    logger.info(msg)



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


def load_gene_tokens(dataset_dir: str | Path) -> np.ndarray:
    dataset_dir = Path(dataset_dir)
    p = dataset_dir / "gene_tokens.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing gene_tokens.npy: {p}")
    return np.load(p)


def build_or_load_atac_memmap(
    dataset_dir: str | Path,
    *,
    context_length: int,
    peak_length: int,
    logger=None,
    memmap_name: str = "large_data.bin",
    dtype=np.int8,
) -> np.memmap:
    """
    Consolidate chunk .npy files into one memmap file once to avoid repeatedly loading many files.
    """
    dataset_dir = Path(dataset_dir)
    chunks = _list_atac_chunks(dataset_dir)

    end_max = max(end for _, end, _ in chunks)
    shape = (end_max, int(context_length), int(peak_length))

    bin_path = dataset_dir / memmap_name
    if not bin_path.exists():
        _log(logger, f"Creating ATAC memmap: {bin_path} (shape={shape}, dtype={dtype})")
        mm = np.memmap(bin_path, dtype=dtype, mode="w+", shape=shape)

        for start, end, p in chunks:
            arr = np.load(p, mmap_mode="r")
            expected_rows = end - start
            if arr.shape[0] != expected_rows:
                raise ValueError(
                    f"Chunk {p.name}: rows={arr.shape[0]} but expected {expected_rows} from filename."
                )
            mm[start:end] = arr

        mm.flush()
        del mm
        _log(logger, "ATAC memmap created.")
    else:
        _log(logger, f"Using existing ATAC memmap: {bin_path}")

    return np.memmap(bin_path, dtype=dtype, mode="r", shape=shape)


def maybe_load_label_metadata(args, dataset_dir: str | Path, logger=None):
    """
    Optional: keep behavior compatible with older scripts if these flags exist in your config.
    """
    dataset_dir = Path(dataset_dir)
    meta_path = dataset_dir / "id2type.pkl"

    if getattr(args, "cell_type_annotation", False) and meta_path.exists():
        with meta_path.open("rb") as f:
            id2type = pickle.load(f)
        args.cell_type_number = len(id2type)
        _log(logger, f"cell_type_number={args.cell_type_number}")

    if getattr(args, "batch_correction", False) and meta_path.exists():
        with meta_path.open("rb") as f:
            id2batch = pickle.load(f)
        args.batch_number = len(id2batch)
        _log(logger, f"batch_number={args.batch_number}")


def make_loader(
    dataset: Dataset,
    *,
    is_train: bool,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
        persistent_workers=(num_workers > 0),
    )


def build_rna_atac_loaders_single_split(args, logger=None):
    """
    Use args.rna_dataset_path and args.atac_dataset_path, then split into train/val.
    """
    rna_ds = HFDataset.load_from_disk(args.rna_dataset_path)

    atac_mm = build_or_load_atac_memmap(
        args.atac_dataset_path,
        context_length=int(args.context_length),
        peak_length=int(args.peak_length),
        logger=logger,
    )
    maybe_load_label_metadata(args, args.atac_dataset_path, logger=logger)
    gene_tokens = load_gene_tokens(args.atac_dataset_path)

    full_ds = CombinedRNATACDataset(rna_ds, atac_mm)

    train_ratio = float(getattr(args, "train_ratio", 0.95))
    seed = int(getattr(args, "split_seed", 42))

    train_size = int(len(full_ds) * train_ratio)
    val_size = len(full_ds) - train_size

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=gen)

    train_loader = make_loader(
        train_ds,
        is_train=True,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_mem),
    )
    val_loader = make_loader(
        val_ds,
        is_train=False,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_mem),
    )
    return train_loader, val_loader, gene_tokens


def _get_path(args, key: str):
    """把 Sacred/DotMap 配置里的路径字段规范化；没提供就返回 None。"""
    x = getattr(args, key, None)

    # Sacred/DotMap 常见：没配时访问属性得到 DotMap()
    if isinstance(x, DotMap):
        # 兼容你如果写成 {path: "..."} 的情况
        if "path" in x and isinstance(x.path, (str, os.PathLike)) and str(x.path).strip():
            return str(x.path)
        return None

    if isinstance(x, (str, os.PathLike)) and str(x).strip():
        return str(x)

    return None


def build_rna_atac_loaders_presplit_train_test(args, logger=None, include_test: bool = True):
    # ---- train ----
    rna_train_path  = _get_path(args, "rna_train_dataset_path")
    atac_train_path = _get_path(args, "atac_train_dataset_path")
    if rna_train_path is None or atac_train_path is None:
        raise ValueError(f"Missing train paths: rna_train={rna_train_path}, atac_train={atac_train_path}")

    rna_train = HFDataset.load_from_disk(rna_train_path)
    atac_train = build_or_load_atac_memmap(
        atac_train_path,
        context_length=int(args.context_length),
        peak_length=int(args.peak_length),
        logger=logger,
    )
    maybe_load_label_metadata(args, atac_train_path, logger=logger)
    gene_tokens = load_gene_tokens(atac_train_path)

    train_full = CombinedRNATACDataset(rna_train, atac_train)

    val_ratio = float(getattr(args, "val_ratio", 0.1))
    seed = int(getattr(args, "seed", 42))

    val_size = int(len(train_full) * val_ratio)
    train_size = len(train_full) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=gen)

    train_loader = make_loader(
        train_ds,
        is_train=True,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(getattr(args, "pin_mem", True)),
    )
    val_loader = make_loader(
        val_ds,
        is_train=False,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(getattr(args, "pin_mem", True)),
    )

    # ---- test (optional) ----
    test_loader = None
    if include_test:
        rna_test_path  = _get_path(args, "rna_test_dataset_path")
        atac_test_path = _get_path(args, "atac_test_dataset_path")

        if rna_test_path and atac_test_path:
            rna_test = HFDataset.load_from_disk(rna_test_path)
            atac_test = build_or_load_atac_memmap(
                atac_test_path,
                context_length=int(args.context_length),
                peak_length=int(args.peak_length),
                logger=logger,
            )
            test_ds = CombinedRNATACDataset(rna_test, atac_test)
            test_loader = make_loader(
                test_ds,
                is_train=False,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                pin_memory=bool(getattr(args, "pin_mem", True)),
            )
        else:
            if logger:
                logger.info("No test dataset paths provided; skipping test loader.")

    return train_loader, val_loader, test_loader, gene_tokens




def build_strategy(args):
    """Prefer explicit DDPStrategy instead of string aliases."""
    if getattr(args, "use_sharded_training", False):
        return "fsdp"

    timeout_s = int(36000)

    return DDPStrategy(
        find_unused_parameters=True,
        timeout=timedelta(seconds=timeout_s),
    )


def load_model_from_ckpt_or_init(ModelCls, args, logger=None):
    ckpt = getattr(args, "model_load_path", None)
    if ckpt and Path(ckpt).is_file():
        _log(logger, f"Loading checkpoint: {ckpt}")
        return ModelCls.load_from_checkpoint(
            ckpt,
            map_location="cpu",
            config=args,
            strict=False,
        )
    return ModelCls(args)


def set_trainable_params(model, task: str, report_path: str | Path):
    """
    Freeze parameters by task naming convention:
      - rnamlm: freeze '.atac'
      - atacmlm: freeze '.rna'
      - otherwise: train all
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w") as f:
        for name, p in model.named_parameters():
            if task == "rnamlm":
                p.requires_grad = ".atac" not in name
            elif task == "atacmlm":
                p.requires_grad = ".rna" not in name
            else:
                p.requires_grad = True
            f.write(f"{name}, {p.requires_grad}\n")