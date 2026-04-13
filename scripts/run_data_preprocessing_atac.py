import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cellstory.preprocess import pretrain_dataset, gene_tokenizer


def _parse_args():
    parser = argparse.ArgumentParser(
        description="The pre-training dataset is processed"
    )
    parser.add_argument(
        "--input_dataset",
        type=str,
        default="../data/finetune/batch_correction/dataset1-4/atac_data.h5ad,ATAC",
        help='Dataset input in the format "path,type". Multiple datasets can be separated by ";"',
    )
    parser.add_argument(
        "--cell_type_annotation",
        default=False
    )
    parser.add_argument(
        "--batch_label",
        default=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/finetune/batch_correction/dataset1-4/atac_data",
        help="Directory to save data",
    )
    parser.add_argument(
        "--ATAC_vocab_file",
        type=str,
        default="../vocab/ATAC.vocab.json",
        help="File containing the gene vocabulary.",
    )
    parser.add_argument(
        "--all_nonzero_value_set_1",
        type=int,
        default=True
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=2000
    )
    parser.add_argument(
        "--peak_length",
        type=int,
        default=600
    )
    parser.add_argument(
        "--context_select",
        default="random"
    )
    parser.add_argument(
        "--append_cls",
        default=True
    )
    parser.add_argument(
        "--preprocessing",
        default=True
    )
    parser.add_argument(
        "--tokenizer",
        default=True
    )
    parser.add_argument(
        "--all_peaks",
        default=True
    )
    return parser.parse_args()


def _parse_input_dataset(input_dataset: str):
    dataset_file_list = []
    data_types = []

    for item in input_dataset.split(";"):
        item = item.strip()
        if not item:
            continue

        parts = [x.strip() for x in item.split(",", 1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                '--input_dataset must be in the format "path,type". '
                'Example: "../data/finetune/batch_correction/dataset1-4/atac_data.h5ad,ATAC"'
            )

        dataset_file_list.append(parts[0])
        data_types.append(parts[1])

    if not dataset_file_list:
        raise ValueError("No valid dataset entries were found in --input_dataset.")

    return dataset_file_list, data_types


def natural_sort_key(item):
    if isinstance(item, (list, tuple)):
        item = item[0]

    if ":" in item and "-" in item:
        chromosome, positions = item.split(":")
        chr_num = pretrain_dataset.chr_to_num(chromosome)
        start, _ = positions.split("-")
        return int(chr_num), int(start)

    return float("inf"), float("inf")


if __name__ == "__main__":
    args = _parse_args()
    end_sum = []

    if args.preprocessing:
        dataset_file_list, data_types = _parse_input_dataset(args.input_dataset)

        os.makedirs(args.output_dir, exist_ok=True)

        for i in range(len(dataset_file_list)):
            adata = sc.read_h5ad(dataset_file_list[i])

            if args.cell_type_annotation:
                celltype_id_labels = adata.obs["cell_type"].astype("category").cat.codes.values
                id2type = dict(enumerate(adata.obs["cell_type"].astype("category").cat.categories))
                with open(f"{args.output_dir}/id2type.pkl", "wb") as f:
                    pickle.dump(id2type, f)

            if args.batch_label:
                batch_id_labels = adata.obs["batch_id"].astype("category").cat.codes.values
                id2type = dict(enumerate(adata.obs["batch_id"].astype("category").cat.categories))
                with open(f"{args.output_dir}/id2type.pkl", "wb") as f:
                    pickle.dump(id2type, f)

            vocab = gene_tokenizer.GeneVocab.from_file(args.ATAC_vocab_file)

            adata_var_index = adata.var.index
            sorted_adata_var_index = [
                idx for idx, _ in sorted(
                    enumerate(adata_var_index),
                    key=lambda x: natural_sort_key(x[1])
                )
            ]

            chr_start_end = [
                (
                    pretrain_dataset.chr_to_num(s.split(":")[0]),
                    int(s.split(":")[1].split("-")[0]),
                    int(s.split(":")[1].split("-")[1]),
                )
                for s in adata.var.index[sorted_adata_var_index]
            ]

            sorted_data = sorted(vocab.get_stoi().items(), key=natural_sort_key)

            indexes = []
            chroms = []
            starts = []
            ends = []

            for key, value in sorted_data:
                if ":" in key and "-" in key:
                    chromosome, positions = key.split(":")
                    chr_num = pretrain_dataset.chr_to_num(chromosome)
                    start, end = positions.split("-")
                    chroms.append(chr_num)
                    indexes.append(value)
                    starts.append(int(start))
                    ends.append(int(end))

            patch_indices, region_counts = pretrain_dataset.map_points_to_regions_and_get_indices(
                chr_start_end, chroms, starts, ends, indexes
            )

            n_obs = adata.n_obs
            step = 500

            for start in tqdm(range(0, n_obs, step)):
                end = min(start + step, n_obs)

                ad_mem = adata[start:end].to_memory()[:, sorted_adata_var_index]

                if not isinstance(ad_mem.X, csr_matrix):
                    ad_mem.X = csr_matrix(ad_mem.X)

                target_values, gene_tokens, vocab = pretrain_dataset.load_anndata(
                    ad_mem, data_types[i], args, patch_indices, vocab
                )

                if args.tokenizer:
                    patch_data, gene_ids = gene_tokenizer.tokenize_batch_edit(
                        data=target_values,
                        gene_ids=gene_tokens,
                        pad_token_id=vocab["<pad>"],
                        max_len=args.context_length,
                        target_length=args.peak_length,
                        pad_value=-2,
                        append_cls=args.append_cls,
                        all_nonzero_value_set_1=args.all_nonzero_value_set_1,
                        cls_id=vocab["<cls>"]
                    )

                    if args.cell_type_annotation:
                        cell_type_label = celltype_id_labels[start:end, np.newaxis, np.newaxis] * np.ones(
                            (1, 1, patch_data.shape[2])
                        )
                        patch_data = np.concatenate((patch_data, cell_type_label), axis=1)

                    if args.batch_label:
                        batch_label = batch_id_labels[start:end, np.newaxis, np.newaxis] * np.ones(
                            (1, 1, patch_data.shape[2])
                        )
                        patch_data = np.concatenate((patch_data, batch_label), axis=1)

                    np.save(f"{args.output_dir}/dataset_{i}_{start}_{end}.npy", patch_data)

                del ad_mem

            end_sum.append(end)
            gene_ids = np.array([vocab["<mask>"] if x is None else x for x in gene_ids])

            np.save(f"{args.output_dir}/gene_tokens.npy", gene_ids)
            vocab.save_json(f"{args.output_dir}/vocab_{data_types[i]}.json")

            if getattr(adata, "file", None) is not None:
                adata.file.close()

    npy_files = [str(f) for f in Path(args.output_dir).glob("dataset_*.npy")]

    def sort_by_index(file_path):
        match = re.search(r"dataset_(\d+)_(\d+)_(\d+)\.npy$", file_path)
        if match:
            data_num = int(match.group(1))
            start = int(match.group(2))
            end = int(match.group(3))
            return data_num, start, end
        return float("inf"), float("inf"), float("inf")

    npy_files = sorted(npy_files, key=sort_by_index)

    max_ends = {}
    for npy in npy_files:
        match = re.search(r"dataset_(\d+)_(\d+)_(\d+)\.npy$", npy)
        data_num = int(match.group(1))
        start = int(match.group(2))
        end = int(match.group(3))

        if data_num not in max_ends:
            max_ends[data_num] = end
        else:
            max_ends[data_num] = max(max_ends[data_num], end)

    total_sum = sum(max_ends.values())
    print("npy_files", npy_files)

    large_bin_path = f"{args.output_dir}/large_data.bin"
    if not os.path.exists(large_bin_path):
        with open(large_bin_path, "wb"):
            pass

    extra_channels = 0
    if args.cell_type_annotation:
        extra_channels += 1
    if args.batch_label:
        extra_channels += 1

    mm = np.memmap(
        large_bin_path,
        dtype=np.int8,
        mode="r+",
        shape=(total_sum, args.context_length + extra_channels, args.peak_length)
    )

    for index, filename in tqdm(enumerate(npy_files), total=len(npy_files)):
        match = re.search(r"dataset_(\d+)_(\d+)_(\d+)\.npy$", filename)
        data_num = int(match.group(1))
        start = int(match.group(2))
        end = int(match.group(3))

        array = np.load(filename, mmap_mode="r")
        mm[start + sum(end_sum[0:data_num]): end + sum(end_sum[0:data_num])] = array
        print("finish:", index)

    print(mm[0])
    print(mm.shape)
    mm.flush()
