import argparse
import sys

import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cellstory.preprocess.preprocessor_tokenizer import preprocess_and_tokenize
from cellstory.preprocess.dataset import (
    tokenized_dict_dataset_to_huggingface_dataset,
    save_huggingface_dataset
)
def _parse_args():
    p = argparse.ArgumentParser(description="RNA preprocess + tokenize ")

    # --- preprocess settings (keep your defaults) ---
    p.add_argument("--n_bins", type=int, default=51)
    p.add_argument("--include_zero_gene", action="store_true", default=False)
    p.add_argument("--filter_gene_by_counts", type=int, default=False)
    p.add_argument("--filter_cell_by_counts", type=int, default=False)
    p.add_argument("--subset_hvg", type=int, default=False)
    p.add_argument("--normalize_total", type=float, default=False)
    p.add_argument("--log1p", action="store_true", default=False)

    # --- tokenize settings ---
    p.add_argument("--append_cls", action="store_true", default=True)
    p.add_argument("--mask_ratio", type=float, default=0.0)
    p.add_argument("--context_length", type=int, default=2000)
    p.add_argument("--tokenize_batch_size", type=int, default=1000)
    p.add_argument("--cell_type", action="store_true", default=True)
    p.add_argument("--all_value_set_1", type=int, default=False)
    # --- keep compatibility with preprocess_and_tokenize(args) ---
    # input_mod / style / task
    p.add_argument("--input_mod", type=str, default="RNA")
    p.add_argument("--input_style", type=str, default="binned")          # normed_raw / log1p / binned
    p.add_argument("--tokenization_style", type=str, default="rna")      # rna
    p.add_argument("--model_task", type=str, default="for_finetune")
    p.add_argument("--dirpath", type=str, default="/t9k/mnt/scllm/finetune_datasets/benchmark/test3_dataset34-37+dataset40-42/dataset34-37/batch4/translation_and_cell_type/rna_pp/rna_train_norm1e4_log1p_hvg2000_test.dataset")

    # dummy args (preprocess_and_tokenize 会先读这些字段，再根据 input_mod 覆盖)
    p.add_argument("--input_h5ad", type=str, default="/t9k/mnt/scllm/finetune_datasets/benchmark/test3_dataset34-37+dataset40-42/dataset34-37/batch4/translation_and_cell_type/rna_pp/rna_train_norm1e4_log1p_hvg2000.h5ad")
    p.add_argument("--input_vocab", type=str, default="RNA")
    p.add_argument("--rna_vocab_file", type=str, default="/t9k/mnt/cellstory/data/RNA.vocab.json")

    return p.parse_args()


if __name__=="__main__":
    args = _parse_args()


    dataset, rna_vocab = preprocess_and_tokenize(args)
    dataset = tokenized_dict_dataset_to_huggingface_dataset(dataset)
    save_huggingface_dataset(dataset, args.dirpath)

  
    

             
   


    
    


    

  
  
 



