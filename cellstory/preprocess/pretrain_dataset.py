import csv

import numpy as np
from anndata import AnnData

from tqdm import tqdm

import re


def map_points_to_regions_and_get_indices(total_points, chroms, starts, ends, indexes):
    patch_indices = [[None, None, None, None] for _ in range(len(starts))]

    region_counts = [0] * len(starts)

    for i in tqdm(range(len(total_points))):

        point_value = total_points[i][1]
        point_chrom = total_points[i][0]

        region_index = None

        for j in range(len(starts)):
            if starts[j] <= point_value <= ends[j] and point_chrom == chroms[j]:
                region_index = j
                break

        if region_index is not None:
            if patch_indices[region_index][0] is None:

                patch_indices[region_index] = [[chroms[region_index], starts[region_index], ends[region_index]],
                                               indexes[region_index], i, i]
            else:
                patch_indices[region_index][3] = i

            region_counts[region_index] += 1

    return patch_indices, region_counts


def chr_to_num(chr_str):
    if chr_str.startswith('chr'):
        chr_num = chr_str[3:]

        if chr_num == 'X':
            return 23
        elif chr_num == 'Y':
            return 24
        else:
            try:
                return int(chr_num)
            except:
                print(chr_num)
                print(chr_str)


    else:
        raise ValueError("Invalid chromosome format")


def sort_key(index):
    chr_part, pos_part = index.split(':')
    chr_num = chr_to_num(chr_part)
    start_pos = int(pos_part.split('-')[0])
    return (chr_num, start_pos)


def sort_by_index(file_path):
    match = re.search(r'dataset_0_(\d+)_(\d+)\.parquet$', file_path)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        return (start, end)
    else:
        return float('inf'), float('inf')


def read_csv_and_extract_columns(csv_file):
    paths = []
    types = []

    with open(csv_file, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)

        next(csv_reader)

        for row in csv_reader:
            paths.append(row[0])
            types.append(row[1])

    return paths, types


def _load_ATAC_all_peaks_anndata_layer_edit(
        adata,
        peak_length,
        data_key="X",
        patch_indices=None

):
    if not isinstance(adata, AnnData):
        raise ValueError("adata must be an AnnData object.")
    if data_key == "X":
        data = adata.X
    elif data_key in adata.layers:
        data = adata.layers[data_key]
    elif data_key in adata.obsm:
        data = adata.obsm[data_key]
    else:
        print(f"Data key {data_key} not found, skip loading.")
        return None
    n_rows, n_cols = data.shape

    tokenized_data = {"target_values": []}

    for i in range(n_rows):  # ~2s/100k cells
        row_data = np.squeeze(data[i].toarray())

        row_gene_tokens = []
        patch_values = []
        for gene_coordinates, peak_id, start, end in patch_indices:
            current_patch_values = [-2] * peak_length

            if len(row_data[start:end]) >= peak_length:
                current_patch_values[:peak_length] = row_data[start:end][:peak_length]

            else:
                current_patch_values[:len(row_data[start:end])] = row_data[start:end]

            patch_values.append(current_patch_values)
            row_gene_tokens.append(peak_id)

        tokenized_data["target_values"].append(patch_values)

    return tokenized_data["target_values"], np.array(row_gene_tokens)


def load_anndata(adata, data_type, args, patch_indices=None, vocab=None, data_key="X"):
    if not isinstance(adata, AnnData):
        raise ValueError("adata must be an AnnData object.")

    tokens = adata.var_names.tolist()

    chroms = []
    chromStarts = []
    chromEnds = []
    ATAC_name = adata.var.index.tolist()
    for ATAC_name in adata.var.index.tolist():
        chrom, chrom_position_range = ATAC_name.split(':')
        chromStart, chromEnd = chrom_position_range.split('-')
        chroms.append(chrom)
        chromStarts.append(chromStart)
        chromEnds.append(chromEnd)

    special_tokens = ["<pad>", "<cls>", "<eoc>", "<mask>"]
    # validate matching between tokens and vocab
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    target_values, gene_tokens = _load_ATAC_all_peaks_anndata_layer_edit(adata, args.peak_length, data_key,
                                                                         patch_indices)

    return target_values, gene_tokens, vocab






