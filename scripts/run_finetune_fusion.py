from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import lightning as pl
from dotmap import DotMap
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

os.environ.setdefault("WANDB_MODE", "disabled")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from cellstory.logger import init_logger
from configs.config_finetune import ex
from models.beitv3_pl_value import BeitForPretrain as BeitForPretrain_Value
from datamodules.datasets_pt import create_dataset_by_split

from datamodules.utils_rna_atac import (
    ensure_dir,
    make_run_name,
    build_strategy,
    build_rna_atac_loaders_presplit_train_test,
    load_model_from_ckpt_or_init,
    set_trainable_params,
)


@ex.automain
def main(_config):
    args = DotMap(_config)
    pl.seed_everything(args.seed)

    ensure_dir(args.log_dir)
    ensure_dir(args.dirpath)

    logger = init_logger(args)

    run_name = make_run_name(args.exp_name)
    wandb_logger = WandbLogger(
        name=run_name,
        save_dir=args.log_dir,
        project=args.project_name,
        log_model=True,
    )

    callbacks = [LearningRateMonitor(logging_interval="step")]
    if args.model_task == "finetune":
        callbacks = [
            ModelCheckpoint(
                dirpath=args.dirpath,
                filename=f"{args.exp_name}_best",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=False,
                verbose=True,
            ),
            *callbacks,
        ]

    logger.info("Building dataloaders...")
    train_loader, val_loader, _, gene_tokens = build_rna_atac_loaders_presplit_train_test(
        args, logger=logger, include_test=False
    )

    args.rna_vocab_size = int(60668)
    args.atac_vocab_size = int(2002)
    args.gene_tokens = np.asarray(gene_tokens, dtype=np.int32)

    logger.info("Loading model...")
    model = load_model_from_ckpt_or_init(BeitForPretrain_Value, args, logger=logger)

    set_trainable_params(
        model,
        task=str(args.task),
        report_path=Path(args.dirpath) / "training_params_name_pair.txt",
    )

    strategy = build_strategy(args)

    max_steps = int(args.max_steps) if getattr(args, "max_steps", None) is not None else -1
    deterministic = bool(getattr(args, "deterministic", True))
    benchmark = bool(getattr(args, "benchmark", True)) and not deterministic

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        precision=args.precision,
        strategy=strategy,
        deterministic=deterministic,
        benchmark=benchmark,
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

    logger.info("Done.")