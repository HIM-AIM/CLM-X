import json
import pickle
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple, Union
from typing_extensions import Self
import random
import numpy as np
import pandas as pd
import torch
import torchtext.vocab as torch_vocab
from torchtext.vocab import Vocab
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class GeneVocab(Vocab):


    def __init__(
            self,
            gene_list_or_vocab: Union[List[str], Vocab],
            specials: Optional[List[str]] = None,
            special_first: bool = True,
            default_token: Optional[str] = "<pad>",
    ) -> None:

        if isinstance(gene_list_or_vocab, Vocab):
            _vocab = gene_list_or_vocab
            if specials is not None:
                raise ValueError(
                    "receive non-empty specials when init from a Vocab object."
                )
        elif isinstance(gene_list_or_vocab, list):
            _vocab = self._build_vocab_from_iterator(
                gene_list_or_vocab,
                specials=specials,
                special_first=special_first,
            )
        else:
            raise ValueError(
                "gene_list_or_vocab must be a list of gene names or a Vocab object."
            )
        super().__init__(_vocab.vocab)
        if default_token is not None and default_token in self:
            self.set_default_token(default_token)

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> Self:

        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix == ".pkl":
            with file_path.open("rb") as f:
                vocab = pickle.load(f)
                return cls(vocab)
        elif file_path.suffix == ".json":
            with file_path.open("r") as f:
                token2idx = json.load(f)
                return cls.from_dict(token2idx)
        else:
            raise ValueError(
                f"{file_path} is not a valid file type. "
                "Only .pkl and .json are supported."
            )

    @classmethod
    def from_dict(
            cls,
            token2idx: Dict[str, int],
            default_token: Optional[str] = "<pad>",
    ) -> Self:

        _vocab = cls([])

        for t, i in sorted(token2idx.items(), key=lambda x: x[1]):
            _vocab.insert_token(t, i)

        if default_token is not None and default_token in _vocab:
            _vocab.set_default_token(default_token)

        return _vocab

    def _build_vocab_from_iterator(
            self,
            iterator: Iterable,
            min_freq: int = 1,
            specials: Optional[List[str]] = None,
            special_first: bool = True,
    ) -> Vocab:


        counter = Counter()
        counter.update(iterator)

        if specials is not None:
            for tok in specials:
                del counter[tok]

        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
        sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        if specials is not None:
            if special_first:
                specials = specials[::-1]
            for symbol in specials:
                ordered_dict.update({symbol: min_freq})
                ordered_dict.move_to_end(symbol, last=not special_first)

        word_vocab = torch_vocab.vocab(ordered_dict, min_freq=min_freq)
        return word_vocab

    @property
    def pad_token(self) -> Optional[str]:

        if getattr(self, "_pad_token", None) is None:
            self._pad_token = None
        return self._pad_token

    @pad_token.setter
    def pad_token(self, pad_token: str) -> None:

        if pad_token not in self:
            raise ValueError(f"{pad_token} is not in the vocabulary.")
        self._pad_token = pad_token

    def save_json(self, file_path: Union[Path, str]) -> None:

        if isinstance(file_path, str):
            file_path = Path(file_path)
        with file_path.open("w") as f:
            json.dump(self.get_stoi(), f, indent=2)

    def set_default_token(self, default_token: str) -> None:

        if default_token not in self:
            raise ValueError(
                f"{default_token} is not in the vocabulary."
            )
        self.set_default_index(self[default_token])





def tokenize_batch_edit(
        data,
        gene_ids,
        pad_token_id,
        max_len,
        target_length,
        pad_value: int = -2,
        append_cls: bool = False,
        all_nonzero_value_set_1: bool = True,
        cls_id: int = "<cls>",
) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:


    for i in tqdm(range(len(data))):

        values = np.array(data[i], dtype=np.int8)
        if all_nonzero_value_set_1:
            values[values > 0] = 1
        genes = np.array(gene_ids)

        num_tokens = len(genes)


        patched_values = values


        if append_cls:
            genes = np.insert(genes, 0, cls_id)
            cls_value = np.zeros(target_length)
            patched_values = np.vstack((cls_value[np.newaxis, :], patched_values))
            num_tokens += 1

        if num_tokens < max_len:
            pad_length = max_len - num_tokens
            padding = np.full((pad_length,), pad_token_id, dtype=genes.dtype)

            genes = np.concatenate((genes, padding))
            patched_values_padding = np.full(
                (pad_length, target_length), -2, dtype=patched_values.dtype
            )
            patched_values = np.concatenate((patched_values, patched_values_padding))

        data[i] = patched_values

    return np.array(data, dtype=np.int8), genes

def tokenize_batch(
        data: np.ndarray,
        gene_ids: np.ndarray,
        patch_indices: List[Tuple[int, int]],
        celltype: List[str],
        tokenization_style: str,
        mask_token_id: int,
        pad_token_id: int,
        vocab_size: int,
        mask_ratio: float,
        max_len: int,
        mask_value: int = -1,
        pad_value: int = -2,
        return_pt: bool = True,
        append_cls: bool = False,
        include_zero_gene: bool = True,
        cls_id: int = "<cls>",
        mod_type: np.ndarray = None,
        cls_id_mod_type: int = None,
) -> Dict[int, Dict[str, Union[torch.Tensor, np.ndarray, int]]]:


    if mod_type is not None and data.shape[1] != len(mod_type):
        raise ValueError(
            f" ({data.shape[1]}) ({len(mod_type)}) "
        )

    if len(celltype) != data.shape[0]:
        raise ValueError(
            f" ({len(celltype)})  ({data.shape[0]}) "
        )

    all_data_dict = {}

    if len(data.shape) == 1:
        data = data.reshape(1, -1)

    for i in range(len(data)):
        row = data[i]
        target_values_rna_full = row.copy()

        current_celltype_str = celltype[i]

        try:
            current_celltype = int(current_celltype_str)
        except Exception:
            current_celltype = current_celltype_str

        mod_types = None

        if tokenization_style == "rna":
            if include_zero_gene:
                values = row
                genes = gene_ids
                if mod_type is not None:
                    mod_types = mod_type
            else:
                idx = np.nonzero(row)[0].astype(int)
                values = row[idx]
                genes = np.array(gene_ids)[idx]
                if mod_type is not None:
                    mod_types = mod_type[idx]

            num_tokens = len(genes)

            if append_cls:
                genes = np.insert(genes, 0, cls_id)
                values = np.insert(values, 0, 0)
                if mod_type is not None:
                    mod_types = np.insert(mod_types, 0, cls_id_mod_type)
                num_tokens += 1

            if num_tokens > max_len:
                if append_cls:
                    idx = np.random.choice(num_tokens - 1, max_len - 1, replace=False)
                    idx = idx + 1
                    idx = np.insert(idx, 0, 0)
                else:
                    idx = np.random.choice(num_tokens, max_len, replace=False)
                genes = genes[idx]
                values = values[idx]
                if mod_types is not None:
                    mod_types = mod_types[idx]
                num_tokens = max_len
                padding_mask = [0] * max_len
            elif num_tokens <= max_len:
                pad_length = max_len - num_tokens
                padding = np.full((pad_length,), pad_token_id, dtype=genes.dtype)
                padding_mask = [0] * len(genes) + [1] * pad_length
                genes = np.concatenate((genes, padding))
                values_padding = np.full((pad_length,), pad_value, dtype=values.dtype)
                values = np.concatenate((values, values_padding))
                if mod_types is not None:
                    mod_types = np.concatenate(
                        (
                            mod_types,
                            np.full((pad_length,), pad_token_id, dtype=mod_types.dtype),
                        )
                    )

            if return_pt:
                genes = torch.from_numpy(genes).long()
                values = torch.from_numpy(values).float()
                target_values = values.clone()
                if mod_type is not None:
                    mod_types = torch.from_numpy(mod_types).long()
                target_values_rna_full = torch.from_numpy(target_values_rna_full).float()
            else:
                target_values = values.copy()

            masked_values, mask_pos = random_mask_value(
                values,
                mask_ratio=mask_ratio,
                mask_value=mask_value,
                pad_value=pad_value,
            )
        else:
            # TODO: 可扩展其他 tokenization_style 处理逻辑
            raise ValueError(f"未知的tokenization_style: {tokenization_style}")

        tokenize_data = {
            "gene_tokens": genes,
            "values": masked_values,
            "target_values": values,
            "target_values_rna_full": target_values_rna_full,
            "values_masked_pos": mask_pos,
            "padding_mask": padding_mask,
            "cell_type": current_celltype,
        }

        all_data_dict[i] = tokenize_data

    return all_data_dict


def _get_mask_token(mask_token_id, vocab_size, token):
    p = random.random()
    if p < 0.8:
        return mask_token_id
    elif p < 0.9:
        return token
    else:
        return random.randint(3, vocab_size - 1)



def tokenize_and_pad_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    patch_indices: list[tuple[int, int]],
    celltype,
    tokenization_style,
    mask_token_id,
    pad_token_id,
    vocab_size,
    mask_ratio,
    max_len: int,
    vocab: Vocab,
    pad_token: str,
    mask_value: int,
    pad_value: int,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_token: str = "<cls>",
    return_pt: bool = True,
    mod_type: np.ndarray = None,
    vocab_mod: Vocab = None,
) -> Dict[str, torch.Tensor]:
    """
    批量处理数据，对数据进行标记化和填充。返回带有基因标识符和计数的列表的元组。
    """

    cls_id = vocab[cls_token]  # 获取cls_token在词汇表中的ID
    if mod_type is not None:
        cls_id_mod_type = vocab_mod[cls_token]  # 获取修改类型的cls_token在词汇表中的ID

    # 对批量数据进行标记化
    tokenized_data = tokenize_batch(
        data,
        gene_ids,
        patch_indices,
        celltype,
        tokenization_style,
        mask_token_id,
        pad_token_id,
        vocab_size,
        mask_ratio,
        max_len,
        mask_value,
        pad_value,
        return_pt=return_pt,
        append_cls=append_cls,
        include_zero_gene=include_zero_gene,
        cls_id=cls_id,
        mod_type=mod_type,
        cls_id_mod_type=cls_id_mod_type if mod_type is not None else None,
    )

    return tokenized_data


def random_mask_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = 0,
) -> torch.Tensor:

    if isinstance(values, torch.Tensor):
        values = values.clone().detach().numpy()
    else:
        values = values.copy()

    row = values
    non_padding_idx = np.nonzero(row - pad_value)[0]
    n_mask = int(len(non_padding_idx) * mask_ratio)
    mask_idx = np.random.choice(
        non_padding_idx, n_mask, replace=False
    )
    row[mask_idx] = mask_value
    masked_pos = np.zeros_like(row)
    masked_pos[mask_idx] = 1
    row = torch.from_numpy(row).float()
    masked_pos = torch.from_numpy(masked_pos).float()
    return row, masked_pos

