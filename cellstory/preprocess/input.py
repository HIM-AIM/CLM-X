import torch
from .preprocessor_tokenizer import preprocess_and_tokenize
from .gene_tokenizer import GeneVocab
from .dataset import (
    tokenized_dict_dataset_to_huggingface_dataset,
    save_huggingface_dataset,
    load_huggingface_dataset,
)
from cellstory.utils import get_obs
import logging

# get logger
logger = logging.getLogger(__name__)


def prepare_dataloader(args):
    """
    根据任务类型准备完整的数据集。
    """
    if args.model_task == "for_finetune":
        dataset, rna_vocab_size, atac_vocab_size = prepare_pretrain_dataset(args)
    elif args.model_task == "finetune":
        dataset, rna_vocab_size, atac_vocab_size = prepare_finetune_dataset(args)
    elif args.model_task == "inference":
        dataset, rna_vocab_size, atac_vocab_size = prepare_inference_dataset(args)
    else:
        raise ValueError(f"未知的 model_task: {args.model_task}")
    return dataset, rna_vocab_size, atac_vocab_size


def prepare_pretrain_dataset(args):
    """
    准备预训练数据集。
    """
    logger.info("Load for_finetune dataset")
    dataset, rna_vocab_size, atac_vocab_size = load_dataset(args)
    return dataset, rna_vocab_size, atac_vocab_size


def prepare_finetune_dataset(args):
    """
    准备微调数据集。
    """
    logger.info("Load finetune dataset")
    dataset, rna_vocab_size, atac_vocab_size = load_dataset(args)
    return dataset, rna_vocab_size, atac_vocab_size


def prepare_inference_dataset(args):
    """
    准备推理数据集。
    """
    logger.info("Load inference dataset")
    dataset, rna_vocab_size, atac_vocab_size = load_dataset(args)
    return dataset, rna_vocab_size, atac_vocab_size


def load_dataset(args):
    # load dataset with/o tokenization
    if args.input_mod == "RNA + ATAC":
        logger.info("Load RNA + ATAC dataset")
        dataset, rna_vocab_size, atac_vocab_size = load_multi_modal_dataset(args)
    elif args.input_mod == "RNA":
        logger.info("Load RNA dataset")
        dataset, rna_vocab_size, atac_vocab_size = load_rna_dataset(args)
    elif args.input_mod == "ATAC":
        logger.info("Load ATAC dataset")
        dataset, rna_vocab_size, atac_vocab_size = load_atac_dataset(args)
    # finally, check vocab_size of RNA & ATAC modal
    rna_vocab_size, atac_vocab_size = check_vocab_size(
        args, rna_vocab_size, atac_vocab_size
    )
    return dataset, rna_vocab_size, atac_vocab_size


def load_multi_modal_dataset(args):
    if args.tokenization:
        # load rna dataset
        rna_args = args.copy()
        rna_args.input_mod = "RNA"
        rna_args.input_h5ad = args.rna_h5ad
        rna_args.vocab_file = args.rna_vocab_file
        dataset_rna, rna_vocab = preprocess_and_tokenize(rna_args)
        rna_vocab_size = len(rna_vocab)
        # load atac dataset
        atac_args = args.copy()
        atac_args.input_mod = "ATAC"
        atac_args.input_h5ad = args.atac_h5ad
        atac_args.vocab_file = args.atac_vocab_file
        dataset_atac, atac_vocab = preprocess_and_tokenize(atac_args)
        atac_vocab_size = len(atac_vocab)

        # merge RNA & ATAC dataset
        dataset = {}
        for key in dataset_rna:
            data_rna = dataset_rna[key]
            data_atac = dataset_atac[key]

            new_data = {}

            for data_key in data_rna:
                new_key = f"{data_key}_rna"
                new_data[new_key] = data_rna[data_key]

            for data_key in data_atac:
                new_key = f"{data_key}_atac"
                new_data[new_key] = data_atac[data_key]

            dataset[key] = new_data
        if args.multi_modal_dataset_path is not None:
            dataset = tokenized_dict_dataset_to_huggingface_dataset(dataset)
            save_huggingface_dataset(dataset, args.multi_modal_dataset_path)
    elif args.multi_modal_dataset_path is not None:
        dataset = load_huggingface_dataset(args.multi_modal_dataset_path)
        rna_vocab_size = None
        atac_vocab_size = None
    else:
        raise ValueError("Please tokenize anndata or provide multi_modal_dataset_path")
    return dataset, rna_vocab_size, atac_vocab_size


def load_rna_dataset(args):
    if args.tokenization:
        dataset, rna_vocab = preprocess_and_tokenize(args)
        dataset = tokenized_dict_dataset_to_huggingface_dataset(dataset)
        rna_vocab_size = len(rna_vocab)
        if args.rna_dataset_path is not None:
            save_huggingface_dataset(dataset, args.rna_dataset_path)
    elif args.rna_dataset_path is not None:
        dataset = load_huggingface_dataset(args.rna_dataset_path)
        rna_vocab_size = None
    else:
        raise ValueError("Please tokenize anndata or provide rna_dataset_path")
    atac_vocab_size = None
    return dataset, rna_vocab_size, atac_vocab_size


def load_atac_dataset(args):
    if args.tokenization:
        dataset, atac_vocab = preprocess_and_tokenize(args)
        dataset = tokenized_dict_dataset_to_huggingface_dataset(dataset)
        atac_vocab_size = len(atac_vocab)
        if args.atac_dataset_path is not None:
            save_huggingface_dataset(dataset, args.atac_dataset_path)
    elif args.atac_dataset_path is not None:
        dataset = load_huggingface_dataset(args.atac_dataset_path)
        atac_vocab_size = None
    else:
        raise ValueError("Please tokenize anndata or provide atac_dataset_path")
    rna_vocab_size = None
    return dataset, rna_vocab_size, atac_vocab_size


def check_vocab_size(args, rna_vocab_size, atac_vocab_size):
    logger.info("Check vocabulary size of RNA & ATAC")
    # check vocab_size of RNA & ATAC, get it if needed
    rna_vocab_size = check_rna_vocab_size(args, rna_vocab_size)
    atac_vocab_size = check_atac_vocab_size(args, atac_vocab_size)
    return rna_vocab_size, atac_vocab_size


def check_rna_vocab_size(args, rna_vocab_size):
    if rna_vocab_size is None:
        if args.rna_vocab_size is not None:
            rna_vocab_size = args.rna_vocab_size
        elif args.rna_vocab_file is not None:
            rna_vocab = GeneVocab.from_file(args.rna_vocab_file)
            rna_vocab_size = len(rna_vocab)
        else:
            raise ValueError(
                "rna_vocab_size or rna_vocab_file must be specified in configs"
            )
    return rna_vocab_size


def check_atac_vocab_size(args, atac_vocab_size):
    if atac_vocab_size is None:
        if args.atac_vocab_size is not None:
            atac_vocab_size = args.atac_vocab_size
        elif args.atac_vocab_file is not None:
            atac_vocab = GeneVocab.from_file(args.atac_vocab_file)
            atac_vocab_size = len(atac_vocab)
        else:
            raise ValueError(
                "atac_vocab_size or atac_vocab_file must be specified in configs"
            )
    return atac_vocab_size

def prepare_rna_inference_data(args):
    """
    prepare dataloader & obs for inference adata
    """
    assert args.rna_h5ad is not None
    logger.info("Prepare dataloader")
    dataloader, rna_vocab_size, atac_vocab_size = prepare_dataloader(args)
    logger.info("Prepare obs")
    adata_obs = get_obs(args.rna_h5ad)

    return adata_obs, dataloader, rna_vocab_size, atac_vocab_size



