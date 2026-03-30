import os
import dotmap
from datetime import datetime
import datetime as dt
from lightning.pytorch.strategies import DDPStrategy,Strategy
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TORCH_DISTRIBUTED_TIMEOUT'] = '36000'
import lightning as pl
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from models.beitv3_pl_value import BeitForPretrain as BeitForPretrain_Value
from datamodules.datasets_pt import create_dataset_by_split
from cellstory.logger import init_logger
from configs.config_finetune import ex
# from configs.config_debug import ex
# os.environ['HTTP_PROXY'] = "http://10.233.17.241:3128"
# os.environ['HTTPS_PROXY'] = "http://10.233.17.241:3128"

@ex.automain
def main(_config):
    # 配置加载和设置
    args_ = dotmap.DotMap(_config)
    pl.seed_everything(args_.seed)  # 设置随机种子以确保实验可重复

    # 训练设置
    if not os.path.exists(args_.log_dir):
        os.makedirs(args_.log_dir, exist_ok=True)  # 创建日志目录
    if not os.path.exists(args_.dirpath):
        os.makedirs(args_.dirpath)  # 创建模型保存目录

    # init logger
    logger = init_logger(args_)

    # set run name for wandb
    now = datetime.now()
    time_str = (
        f"{now.year:04d}{now.month:02d}{now.day:02d}-{now.hour:02d}{now.minute:02d}"
    )
    run_name = f"{args_.exp_name}-{time_str}"
    # create wandb logger
    wandb_logger = pl.pytorch.loggers.WandbLogger(
        name=run_name,
        save_dir=args_.log_dir,
        project=args_.project_name,
        log_model=True,
    )

    # set callbacks
    # checkpoint callback, save model checkpoint
    checkpoint_callback_epoch = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath=args_.dirpath,  # 检查点保存路径
        filename=f"{args_.exp_name}_epoch{{epoch:02d}}-{{train_loss:.3f}}",  # 文件名包含epoch和验证损失
        every_n_epochs=1,  # 每个epoch保存一次
        save_top_k=-1,  # -1表示保存所有符合条件的检查点
        verbose=True,  # 打印保存检查点的信息
        monitor="train_loss",  # 监控的指标为验证损失
        mode="min",  # 选择验证损失最小的检查点
        save_last=True,  # 可选，是否单独保存最新的检查点
    )
    # learning rate callback
    lr_callback = pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback_epoch, lr_callback]

    grad_steps = args_.grad_steps  # 梯度累积步数
    max_steps = args_.max_steps if args_.max_steps is not None else None  # 最大训练步数

    # 定义数据集和模型
    logger.info("loading dataset")
    # TODO create dataloader without tokenization
    is_train = args_.model_task in ["for_finetune", "finetune"]
    mask_type = "value"  # name or value
    train_dataloader,_, rna_vocab_size, atac_vocab_size = create_dataset_by_split(
        args_, is_train=is_train
    )
    # set vocab_size for RNA & ATAC
    # let rna_vocab_size=rna_vocab_size
    args_.rna_vocab_size = rna_vocab_size
    # let atac_vocab_size=atac_vocab_size
    args_.atac_vocab_size = atac_vocab_size

    logger.info(
        f"vocab size: RNA: {args_.rna_vocab_size}, ATAC: {args_.atac_vocab_size}"
    )

    # 可能加载模型参数
    logger.info("loading model parameters")
    MODEL_CLS = BeitForPretrain if mask_type == "name" else BeitForPretrain_Value
    if args_.model_load_path and os.path.exists(args_.model_load_path):
        model = MODEL_CLS(args_)
        model = MODEL_CLS.load_from_checkpoint(
            args_.model_load_path, map_location="cpu", config=args_,strict=False
        )

    else:
        model = MODEL_CLS(args_)  # 初始化模型

    logger.info("begin to train")
    # 设置部分参数的requires_grad为False
    training_params_file = os.path.join(args_.dirpath, "training_params_name_pair.txt")
    prams_check_writer = open(training_params_file, "w")
    for name, param in model.named_parameters():
        if args_.task == "rnamlm":  # rna/B任务时，atac/A相关参数不更新
            if ".atac" in name:
                param.requires_grad = False
        if args_.task == "atacmlm":  # atac/A任务时，rna/B相关参数不更新
            if ".rna" in name:
                param.requires_grad = False
            # if "beit3" in name:
            #     param.requires_grad = False
        prams_check_writer.write("%s, %s\n" % (name, str(param.requires_grad)))
    prams_check_writer.close()

    # 训练设置
    if args_.use_sharded_training:  # 如果使用分片训练
        distributed_strategy = "fsdp"
    else:
        distributed_strategy = "ddp_find_unused_parameters_true"  # 使用DDP策略
    # init lightning trainer
    print(distributed_strategy)
  
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args_.num_gpus,
        num_nodes=args_.num_nodes,
        precision=args_.precision,
        strategy=distributed_strategy,
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
        val_check_interval=args_.val_check_interval,
    
    )

    # start training
    trainer.fit(model, train_dataloaders=train_dataloader,ckpt_path = args_.resume_from_checkpoint)
    logger.info(" finish training ")

