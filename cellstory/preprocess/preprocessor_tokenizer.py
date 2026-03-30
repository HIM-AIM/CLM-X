import gc
from copy import deepcopy
from pathlib import Path
from functools import partial
import math
import scanpy as sc
import numpy as np
from scipy.sparse import issparse

from .gene_tokenizer import tokenize_and_pad_batch, GeneVocab
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

from .preprocess import Preprocessor
import logging

# get logger
logger = logging.getLogger(__name__)


def preprocess_and_tokenize(args):
    """preprocess and tokenize RNA or ATAC data"""
    # TODO optimize arguments
    # TODO fix multi-modal vocab_size
    ################################################################################
    # parameters settings
    ################################################################################
    input_mod = args.input_mod
    tokenization_style =args.tokenization_style
    input_settings = dict(
        input_h5ad=args.input_h5ad,
        vocab_file=args.input_vocab,
        dirpath=args.dirpath,
        model_task=args.model_task,
        input_mod=args.input_mod,
    )
    pp_settings = dict(
        n_bins=args.n_bins,
        include_zero_gene=args.include_zero_gene,
        all_value_set_1 = args.all_value_set_1,
        filter_gene_by_counts=args.filter_gene_by_counts,
        filter_cell_by_counts=args.filter_cell_by_counts,
        subset_hvg=args.subset_hvg,
        normalize_total=args.normalize_total,
        log1p=args.log1p,
    )
    tk_settings = dict(
        # tokenization settings
        mask_token="<mask>",
        pad_token="<pad>",
        mask_ratio=args.mask_ratio,
        max_seq_len=args.context_length,
        mask_value=-1,
        pad_value=-2,
        mod_type=None,
        vocab_mod=None,
        data_key="X",
        pp_normed_key="X_normed",
        pp_log1p_key="X_log1p",
        pp_binned_key="X_binned",
        append_cls = args.append_cls,
        cell_type = args.cell_type
    )
    pp_tk_settings = dict()
    pp_tk_settings.update(input_settings)
    pp_tk_settings.update(pp_settings)
    pp_tk_settings.update(tk_settings)

    # input arguments

    if input_mod == "RNA":
        vocab_file = args.rna_vocab_file
        input_h5ad = args.input_h5ad
    elif input_mod == "ATAC":
        vocab_file = args.atac_vocab_file
        input_h5ad = args.atac_h5ad
    elif input_mod == "RNA + ATAC":
        vocab_file = args.vocab_file
        input_h5ad = args.input_h5ad
    save_dir = convert_to_path(args.dirpath)
    model_task = args.model_task
    input_mod = args.input_mod
    tokenize_batch_size = args.tokenize_batch_size


    if input_mod == "RNA":
        input_style = args.input_style
    elif input_mod == "ATAC":
        input_style = "normed_raw"

    # the values of this map coorespond to the keys in preprocessing
    input_layer_key_dict = {
        "normed_raw": "X_normed",
        # "normed_raw": "binary_data",
        "log1p": "X_normed",
        "binned": "X_binned",
    }
    input_layer_key = input_layer_key_dict[input_style]

    # preprocess settings
    data_key = pp_tk_settings["data_key"]
    pp_normed_key = pp_tk_settings["pp_normed_key"]
    pp_log1p_key = pp_tk_settings["pp_log1p_key"]
    pp_binned_key = pp_tk_settings["pp_binned_key"]
    normalize_total = pp_tk_settings["normalize_total"]
    log1p = pp_tk_settings["log1p"]
    all_value_set_1 = pp_tk_settings["all_value_set_1"]
    filter_gene_by_counts = pp_tk_settings["filter_gene_by_counts"]
    filter_cell_by_counts = pp_tk_settings["filter_cell_by_counts"]
    subset_hvg = pp_tk_settings["subset_hvg"]
    n_bins = pp_tk_settings["n_bins"]


    # tokenization settings
    mask_ratio = pp_tk_settings["mask_ratio"]
    max_len = pp_tk_settings["max_seq_len"]
    pad_token = pp_tk_settings["pad_token"]
    mask_token = pp_tk_settings["mask_token"]
    special_tokens = [pad_token, "<cls>", "<eoc>", mask_token]
    include_zero_gene = pp_tk_settings["include_zero_gene"]
    mod_type = pp_tk_settings["mod_type"]
    vocab_mod = pp_tk_settings["vocab_mod"]
    append_cls = pp_tk_settings["append_cls"]
    cell_type = pp_tk_settings["cell_type"]
    # value of special tokens
    mask_value = pp_tk_settings["mask_value"]
    pad_value = pp_tk_settings["pad_value"]
    # n_input_bins = n_bins

    ################################################################################
    # check & validate parameters settings
    ################################################################################
    assert input_mod in ["RNA", "ATAC", "RNA + ATAC"]
    assert input_style in ["normed_raw", "log1p", "binned"]
    assert model_task in ["for_finetune", "finetune", "inference"]

    ################################################################################
    # vocab
    ################################################################################
    vocab, vocab_special = create_vocab(
        input_h5ad,
        save_dir,
        input_mod,
        vocab_file,
        model_task,
        special_tokens,
        pad_token,
    )

    mask_token_id = vocab_special[mask_token]
    pad_token_id = vocab_special[pad_token]
    vocab_size = len(vocab_special)

    ################################################################################
    # set up the preprocessor and the tokenizer
    ################################################################################
    if input_mod == "RNA":
        preprocessor = Preprocessor(
            use_key=data_key,  # the key in adata.layers to use as raw data
            filter_gene_by_counts=filter_gene_by_counts,  # step 1
            filter_cell_by_counts=filter_cell_by_counts,  # step 2
            normalize_total=normalize_total,  # 3. whether to normalize the raw data and to what sum
            result_normed_key=pp_normed_key,  # the key in adata.layers to store the normalized data
            log1p=log1p,  # 4. whether to log1p the normalized data
            result_log1p_key=pp_log1p_key,
            subset_hvg=subset_hvg,  # 5. whether to subset the raw data to highly variable genes
            hvg_flavor="seurat_v3" if log1p else "cell_ranger",
            binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key=pp_binned_key,  # the key in adata.layers to store the binned data
        )
    elif input_mod == "ATAC":
        # TODO rewrite ATAC preprocess
        # for ATAC data, we currently keep it intact during preprocess

        preprocessor = Preprocessor(
            use_key=data_key,  # the key in adata.layers to use as raw data
            all_value_set_1=all_value_set_1,
            filter_gene_by_counts=False,  # step 1
            filter_cell_by_counts=False,  # step 2
            normalize_total=normalize_total,  # 3. whether to normalize the raw data and to what sum
            result_normed_key=pp_normed_key,  # the key in adata.layers to store the normalized data
            log1p=log1p,  # 4. whether to log1p the normalized data
            result_log1p_key=pp_log1p_key,
            subset_hvg=subset_hvg,  # 5. whether to subset the raw data to highly variable genes
            binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key=pp_binned_key,  # the key in adata.layers to store the binned data
        )

    # create tokenization partial function
    tokenize_func = partial(
        tokenize_and_pad_batch,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        vocab_size=vocab_size,
        tokenization_style=tokenization_style,
        mask_ratio=mask_ratio,
        max_len=max_len,
        vocab=vocab_special,
        mask_value=mask_value,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=append_cls,
        include_zero_gene=include_zero_gene,
        mod_type=mod_type,
        vocab_mod=vocab_mod,
    )

    ################################################################################
    # preprocess and tokenize batchwise
    ################################################################################
    dataset = preprocess_and_tokenize_batchwise(
        input_h5ad,
        preprocessor,
        tokenize_func,
        model_task,
        tokenization_style,
        input_layer_key,
        vocab,
        cell_type,
        tokenize_batch_size,
    )

    return dataset, vocab_special


def create_vocab(
    h5ad, save_dir, input_mod, vocab_file, task, special_tokens, pad_token
):
    """create vocab based on task"""
    adata = sc.read_h5ad(h5ad, backed="r")
    gene_names = adata.var_names.to_list()

    if vocab_file is None:
        vocab = GeneVocab(Vocab(VocabPybind(gene_names, None)))
    else:
        # load vocab file
        vocab = GeneVocab.from_file(vocab_file)
        # check inersection of adata var_names & vocab var_names
        # we suppose that var_names are the same with vocab gene_names
        adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata.var_names
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        # logger.info(
        #     f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        #     f"in vocabulary of size {len(vocab)}."
        # )
        # add genes first during for_finetune
        if task == "inference":
            # adata = adata[:, adata.var["id_in_vocab"] >= 0]
            pass
        else:
            # add genes to vocab
            for g in adata.var_names[gene_ids_in_vocab >= 0]:
                if g not in vocab:
                    vocab.append_token(g)
    # at last, add special tokens to vocab_special
    vocab_stoi = deepcopy(vocab.get_stoi())
    vocab_special = GeneVocab.from_dict(vocab_stoi)
    # add sepecial tokens
    for s in special_tokens:
        if s not in vocab_special:
            vocab_special.append_token(s)
    vocab_special.set_default_index(vocab_special[pad_token])

    # if task not in ["inference"]:
    #     save_dir = convert_to_path(save_dir)
    #     # if not inference, i.e. for_finetune or finetune, save vocab file
    #     vocab_special.save_json(save_dir / f"{input_mod}.vocab.json")
    adata.file.close()
    return vocab, vocab_special




def tokenize_adata(adata, tokenization_style, input_layer_key, cell_type, vocab, tokenize_func):
    """tokenize adata binned matrix % return dataset"""
    # extract information
    if input_layer_key in adata.layers:
        pp_counts = adata.layers[input_layer_key]
    else:
        logger.warning(
            f"input_layer_key {input_layer_key} not found in adata.layers, use X instead."
        )
        pp_counts = adata.X
    # check sparse format
    if issparse(pp_counts):
        pp_counts = pp_counts.toarray()
    # check genes and tokenized gene_ids
    if tokenization_style == "vector":
        total_features = adata.shape[1]
        num_patches = 5000
        average_patch_size = math.ceil(total_features / num_patches)

        def generate_patch_names_and_indices(names, num_patches):
            patch_names = []
            patch_indices = []
            total_features = len(names)
            patch_size = math.ceil(total_features / num_patches)

            for i in range(num_patches):
                start_index = i * patch_size
                end_index = min((i + 1) * patch_size, total_features)

                # Find the chromosome change within the patch
                current_chromosome = names[start_index].split(':')[0]
                for j in range(start_index, end_index):
                    if names[j].split(':')[0] != current_chromosome:
                        # Adjust end_index to the last position before chromosome change
                        end_index = j
                        break

                # Generate the patch name
                patch_name = f"{current_chromosome}:{names[start_index].split(':')[1].split('-')[0]}-{names[end_index - 1].split(':')[1].split('-')[1]}"
                patch_names.append(patch_name)
                patch_indices.append((start_index, end_index))

                # If we have reached the end of the features, stop
                if end_index == total_features:
                    break

            return patch_names, patch_indices

        new_feature_names, patch_indices = generate_patch_names_and_indices(adata.var_names.tolist(), num_patches)
        pp_genes = new_feature_names
        pp_gene_ids = np.array(vocab(pp_genes), dtype=int)
    else:
        pp_genes = adata.var_names.to_list()
        pp_gene_ids = np.array(vocab(pp_genes), dtype=int)
        patch_indices = None
        if cell_type:
            celltype = adata.obs["cell_type"].to_list()
    # tokenization
    dataset = tokenize_func(pp_counts, pp_gene_ids, patch_indices, celltype)
    return dataset


def preprocess_and_tokenize_batchwise(
    h5ad, preprocessor, tokenize_func, task, tokenization_style, input_layer_key, vocab, cell_type, step=100
):
    """read backed adata & preprocess & tokenize"""
    logger.info("Start batch preprocess and tokenization")
    # adata = sc.read_h5ad(h5ad, backed="r")
    adata = sc.read_h5ad(h5ad)
    # mask = [True] * adata.n_vars
    # if task in ["inference"]:
    #     mask = subset_adata_by_vocab(adata, vocab)
    # n_obs = adata.n_obs
    # dd_list = []
    # loop_n = 0
    # for start in range(0, n_obs, step):
    #     if loop_n % 50 == 0:
    #         logger.info(f"Finish {loop_n} batches: {loop_n * step} cells")
    #     end = start + step
    #     if end > n_obs:
    #         end = n_obs
    #     # extract to memory
    #     ad_mem = adata[start:end, mask].to_memory()
    #     # preprocess
    #     preprocessor(ad_mem)
    #     # tokenization
    #     dataset = tokenize_adata(ad_mem, tokenization_style, input_layer_key, vocab, tokenize_func)
    #     # clear memory
    #     del ad_mem
    #     gc.collect()
    #     dd_list.append(dataset)
    #     loop_n += 1
    # adata.file.close()
    # logger.info("Finish batch preprocess and tokenization")
    # logger.info("Start merging tokenized batches")
    # merged_dataset = merge_tokenized_batches(dd_list)
    # logger.info("Finish merging tokenized batches")
    # # clear memory
    # del dd_list
    # gc.collect()

    preprocessor(adata)
    # tokenization
    dataset = tokenize_adata(adata, tokenization_style, input_layer_key, cell_type, vocab, tokenize_func)
    # close backed adata

    return dataset


def merge_tokenized_batches(batch_list, start_n=0):
    """merge batch tokenized dataset"""
    n_seq_list = []
    tokenized_seq_list = []
    for batch in batch_list:
        n_seq_list.append(len(batch))
        tokenized_seq_list.extend(list(batch.values()))
    new_int_keys = range(start_n, sum(n_seq_list) + start_n)
    merged_dataset = dict(zip(new_int_keys, tokenized_seq_list))
    return merged_dataset


def convert_to_path(path):
    if isinstance(path, str):
        path = Path(path)
    return path
