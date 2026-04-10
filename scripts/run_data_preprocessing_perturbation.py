import json
import os
import argparse

import numpy as np
import scanpy as sc
from tqdm import tqdm
from datasets import Dataset, Features, Value, Sequence

from cellstory.preprocess import gene_tokenizer


def get_gene_ids(adata, vocab_path):
    tokens = adata.var["gene_name"].tolist()
    gene_name_to_idx = {gene: idx for idx, gene in enumerate(tokens)}
    vocab = gene_tokenizer.GeneVocab.from_file(vocab_path)

    adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in tokens]

    special_tokens = ["<pad>", "<cls>", "<eoc>", "<mask>"]
    for token in special_tokens:
        if token not in vocab:
            vocab.append_token(token)

    match_ratio = sum(token in vocab for token in tokens) / len(tokens)
    print(f"{match_ratio * 100:.2f}% of gene tokens are in the vocabulary.")

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in tokens],
        dtype=np.int32,
    )
    return gene_ids, gene_name_to_idx


def split_by_condition(adata, train_ratio=0.8, seed=42, ctrl_condition="ctrl"):
    all_conditions = sorted(adata.obs["condition"].unique().tolist())
    non_ctrl_conditions = [cond for cond in all_conditions if cond != ctrl_condition]

    if len(non_ctrl_conditions) < 2:
        raise ValueError(
            "At least two non-control conditions are required for a condition-level split."
        )

    rng = np.random.default_rng(seed)
    shuffled_conditions = non_ctrl_conditions.copy()
    rng.shuffle(shuffled_conditions)

    split_idx = int(len(shuffled_conditions) * train_ratio)
    split_idx = max(1, split_idx)
    split_idx = min(split_idx, len(shuffled_conditions) - 1)

    train_conditions = shuffled_conditions[:split_idx]
    test_conditions = shuffled_conditions[split_idx:]

    if ctrl_condition in all_conditions:
        train_conditions = [ctrl_condition] + train_conditions

    train_adata = adata[adata.obs["condition"].isin(train_conditions)].copy()
    test_adata = adata[adata.obs["condition"].isin(test_conditions)].copy()

    split_info = {
        "train": train_conditions,
        "test": test_conditions,
        "train_ratio": train_ratio,
        "seed": seed,
        "ctrl_condition": ctrl_condition,
    }
    return train_adata, test_adata, split_info


def to_dense_float32(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x).reshape(-1).astype(np.float32)


def get_ctrl_mean(adata, ctrl_condition="ctrl"):
    ctrl_x = adata[adata.obs["condition"] == ctrl_condition].X
    if ctrl_x.shape[0] == 0:
        raise ValueError(f"No cells found for control condition: {ctrl_condition}")

    ctrl_mean = ctrl_x.mean(axis=0)
    return np.asarray(ctrl_mean).reshape(-1).astype(np.float32)


def tokenizer_dataset(ctrl_mean, adata, gene_name_to_idx, gene_ids, ctrl_condition="ctrl"):
    all_data_dict = {}
    condition_list = adata.obs["condition"].tolist()

    for i in tqdm(range(adata.shape[0]), desc="Processing cells", total=adata.shape[0]):
        row_data = to_dense_float32(adata[i, :].X)
        condition = condition_list[i]

        if condition == ctrl_condition:
            pert_flag = np.full((len(row_data),), 2, dtype=np.int32)
            ctrl_value = row_data
        else:
            pert_flag = np.zeros(len(row_data), dtype=np.int32)
            pert = condition.split("+")[0]

            if pert not in gene_name_to_idx:
                print(f"Skipping cell {i}: perturbation {pert} not in gene_name_to_idx")
                continue

            pert_flag[gene_name_to_idx[pert]] = 1
            ctrl_value = ctrl_mean

        all_data_dict[i] = {
            "perturb_value": row_data,
            "gene_id": gene_ids,
            "ctrl_value": ctrl_value,
            "pert_flag": pert_flag,
        }

    features = Features({
        "index": Value("int32"),
        "perturb_value": Sequence(Value("float32")),
        "gene_id": Sequence(Value("int32")),
        "ctrl_value": Sequence(Value("float32")),
        "pert_flag": Sequence(Value("int32")),
    })

    dataset = Dataset.from_dict(
        {
            "index": np.array(list(all_data_dict.keys()), dtype=np.int32),
            "perturb_value": [v["perturb_value"] for v in all_data_dict.values()],
            "gene_id": [v["gene_id"] for v in all_data_dict.values()],
            "ctrl_value": [v["ctrl_value"] for v in all_data_dict.values()],
            "pert_flag": [v["pert_flag"] for v in all_data_dict.values()],
        },
        features=features,
    ).with_format("torch")

    return dataset


def _parse_args():
    parser = argparse.ArgumentParser(description="Process perturbation dataset")
    parser.add_argument(
        "--input_RNA_h5ad",
        type=str,
        default="../perturbation_data/replogle_k562_essential.h5ad",
        help="Input h5ad file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../perturbation_data/replogle_k562_essential",
        help="Output directory",
    )
    parser.add_argument(
        "--vocab_dir",
        type=str,
        default="../RNA.vocab.json",
        help="Vocabulary file",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Condition-level train split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting conditions",
    )
    parser.add_argument(
        "--ctrl_condition",
        type=str,
        default="ctrl",
        help="Name of the control condition",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    adata = sc.read_h5ad(args.input_RNA_h5ad)

    gene_ids, gene_name_to_idx = get_gene_ids(adata, args.vocab_dir)

    train_adata, test_adata, split_info = split_by_condition(
        adata,
        train_ratio=args.train_ratio,
        seed=args.seed,
        ctrl_condition=args.ctrl_condition,
    )

    with open(os.path.join(args.output_dir, "split_conditions.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    train_h5ad_path = os.path.join(args.output_dir, "train_data.h5ad")
    test_h5ad_path = os.path.join(args.output_dir, "test_data.h5ad")

    train_adata.write_h5ad(train_h5ad_path)
    print(f"Train AnnData saved to {train_h5ad_path}")

    test_adata.write_h5ad(test_h5ad_path)
    print(f"Test AnnData saved to {test_h5ad_path}")

    ctrl_mean = get_ctrl_mean(adata, ctrl_condition=args.ctrl_condition)

    print("Processing train dataset")
    train_dataset = tokenizer_dataset(
        ctrl_mean,
        train_adata,
        gene_name_to_idx,
        gene_ids,
        ctrl_condition=args.ctrl_condition,
    )

    print("Processing test dataset")
    test_dataset = tokenizer_dataset(
        ctrl_mean,
        test_adata,
        gene_name_to_idx,
        gene_ids,
        ctrl_condition=args.ctrl_condition,
    )

    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("Start saving Hugging Face datasets to disk")
    train_dataset.save_to_disk(train_dir, max_shard_size="20GB", num_proc=1)
    test_dataset.save_to_disk(test_dir, max_shard_size="20GB", num_proc=1)
    print("Finish saving Hugging Face datasets to disk")