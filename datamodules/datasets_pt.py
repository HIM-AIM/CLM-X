# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import os
import json
import random
import torch
from datasets import load_dataset
import torch.distributed as dist

from torchvision.datasets.folder import default_loader


from cellstory.preprocess.input import prepare_dataloader, prepare_dataloader
import logging

# get logger
logger = logging.getLogger(__name__)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        split,
        transform,
        tokenizer,
        max_text_len,
        task=None,
    ):
        index_files = self.get_index_files(split, task=task)
        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = max_text_len
        self.data_path = data_path
        items = []
        self.index_files = index_files

        offset = 0
        for _index_file in index_files:
            index_file = os.path.join(data_path, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                print(
                    "Load %d image-text pairs from %s. "
                    % (len(items) - offset, index_file)
                )
                offset = len(items)
        self.items = items
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.loader = default_loader
        self.transform = transform
        self.split = split

    @staticmethod
    def get_index_files(split):
        raise NotImplementedError()

    def _get_image(self, image_path: str):
        image_path = os.path.join(self.data_path, image_path)
        image = self.loader(image_path)
        return self.transform(image)

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.encode(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[: max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return (
            tokens + [self.pad_token_id] * (max_len - num_tokens),
            padding_mask,
            num_tokens,
        )

    def _get_image_text_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["text_segment"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = "{" + "\n  Number of items: %s," % self.__len__()
        body += "\n  data root = %s," % self.data_path
        body += "\n  split = %s," % self.split
        body += "\n  dataset index files = %s" % str(self.index_files)
        body += "\n  num max bpe tokens = %s" % self.num_max_bpe_tokens
        body += "\n  transforms = ["
        for t in self.transform.transforms:
            body += "\n    %s" % str(t)
        body += "\n  ]"
        body += "\n}"

        return head + body


class CaptioningDataset(BaseDataset):
    def __init__(
        self, data_path, split, transform, tokenizer, max_text_len, task, mask_prob
    ):
        super().__init__(
            data_path=data_path,
            split=split,
            transform=transform,
            tokenizer=tokenizer,
            max_text_len=max_text_len,
            task=task,
        )
        self.mask_token_id = tokenizer.mask_token_id
        self.language_vocab_size = tokenizer.vocab_size
        self.mask_prob = mask_prob

    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("coco_captioning.train.jsonl",)
        elif split == "val":
            return ("coco_captioning.val.jsonl",)
        elif split == "test":
            return (f"{task}.test.jsonl",)
        else:
            raise RuntimeError("split %s is not found!" % split)

    def _get_mask_token(self, token):
        p = random.random()
        if p < 0.8:
            return self.mask_token_id
        elif p < 0.9:
            return token
        else:
            return random.randint(3, self.language_vocab_size - 1)

    def _masking_on_text_tokens(self, tokens, num_tokens, mask_prob):
        bool_masked_pos = [0] * len(tokens)
        to_mask = min(int(num_tokens * mask_prob + 0.5), num_tokens - 1)
        to_mask = max(to_mask, 1)
        num_masked_tokens = 0
        while num_masked_tokens < to_mask:
            i = random.randint(1, num_tokens - 1)
            if bool_masked_pos[i] == 0:
                bool_masked_pos[i] = 1
                tokens[i] = self._get_mask_token(tokens[i])
                num_masked_tokens += 1

        return tokens, bool_masked_pos

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        data["image_id"] = item["image_id"]
        text_segment = item["text_segment"]
        if text_segment is not None:
            language_tokens, padding_mask, num_tokens = self._get_text_segment(
                text_segment
            )
            masked_tokens = language_tokens[:]
            masked_tokens, language_masked_pos = self._masking_on_text_tokens(
                masked_tokens, num_tokens, self.mask_prob
            )
            data["language_tokens"] = language_tokens
            data["masked_tokens"] = masked_tokens
            data["language_masked_pos"] = language_masked_pos
            data["padding_mask"] = padding_mask
        return data


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_text_len, mask_prob):
        # 初始化函数，定义类变量
        self.tokenizer = tokenizer  # 保存传入的分词器
        self.num_max_bpe_tokens = max_text_len  # 最大文本长度限制
        self.data_path = data_path  # 数据文件路径

        # 从tokenizer中获取特殊token的ID
        self.bos_token_id = tokenizer.bos_token_id  # 句子开始token
        self.eos_token_id = tokenizer.eos_token_id  # 句子结束token
        self.pad_token_id = tokenizer.pad_token_id  # 填充token

        self.mask_token_id = tokenizer.mask_token_id  # 掩码token
        self.language_vocab_size = tokenizer.vocab_size  # 词汇表大小
        self.mask_prob = mask_prob  # 掩码概率

        # 加载数据集并过滤掉空文本
        self.dataset = load_dataset("parquet", data_files=data_path)["train"]
        self.dataset = self.dataset.filter(lambda x: x["text"].strip() != "")

    def _get_mask_token(self, token):
        # 根据一定概率返回不同的mask token
        p = random.random()
        if p < 0.8:
            return self.mask_token_id  # 80%概率返回mask token
        elif p < 0.9:
            return token  # 10%概率返回原token
        else:
            return random.randint(
                3, self.language_vocab_size - 1
            )  # 10%概率返回随机token

    def _masking_on_text_tokens(self, tokens, num_tokens, mask_prob):
        # 对文本tokens进行掩码
        bool_masked_pos = [0] * len(tokens)  # 创建掩码位置标记列表
        to_mask = min(
            int(num_tokens * mask_prob + 0.5), num_tokens - 1
        )  # 计算需要掩码的token数
        to_mask = max(to_mask, 1)  # 确保至少有一个token被掩码
        num_masked_tokens = 0
        while num_masked_tokens < to_mask:
            i = random.randint(1, num_tokens - 1)  # 随机选择一个token进行掩码
            if bool_masked_pos[i] == 0:
                bool_masked_pos[i] = 1
                tokens[i] = self._get_mask_token(tokens[i])  # 应用掩码
                num_masked_tokens += 1

        return tokens, bool_masked_pos

    def _get_text_segment(self, text_segment, max_len=None):
        # 获取文本段的token表示
        if isinstance(text_segment, str):
            tokens = self.tokenizer.encode(text_segment)  # 编码文本
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError(
                "The text segment should contains at least one tokens!"
            )  # 确保tokens非空
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[: max_len - 2]  # 限制token的长度

        tokens = (
            [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        )  # 添加开始和结束token
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (
            max_len - num_tokens
        )  # 创建padding mask
        return (
            tokens + [self.pad_token_id] * (max_len - num_tokens),
            padding_mask,
            num_tokens,
        )

    def __getitem__(self, index: int):
        # 获取单个数据项
        data = dict()
        text_segment = self.dataset[index]["text"]
        language_tokens, padding_mask, num_tokens = self._get_text_segment(text_segment)
        masked_tokens = language_tokens[:]
        masked_tokens, language_masked_pos = self._masking_on_text_tokens(
            masked_tokens, num_tokens, self.mask_prob
        )
        data["language_tokens"] = language_tokens
        data["masked_tokens"] = masked_tokens
        data["language_masked_pos"] = language_masked_pos
        data["padding_mask"] = padding_mask

        return data

    def __len__(self) -> int:
        # 返回数据集的总长度
        return len(self.dataset)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def merge_batch_tensors_by_dict_key(batch):
    """
    batch collate
    """
    batch_tensors = {}
    for tensor_key in batch[0]:
        if isinstance(batch[0][tensor_key], torch.Tensor):
            batch_tensors[tensor_key] = torch.stack([d[tensor_key] for d in batch])
        else:
            batch_tensors[tensor_key] = torch.tensor(
                [d[tensor_key] for d in batch], dtype=torch.long
            )
    return batch_tensors


def create_dataloader(
    dataset, is_train, batch_size, num_workers, pin_mem, dist_eval=False
):
    sampler = None
    if is_train and dist_eval:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    elif is_train:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=is_train,
        collate_fn=merge_batch_tensors_by_dict_key,
    )


def create_dataset_by_split(args, is_train=True):


    logger.info("Prepare dataset")
    dataset, rna_vocab_size, atac_vocab_size = prepare_dataloader(args)

    total_size = len(dataset)
    val_size = int(total_size * args.val_ration)
    train_size = total_size - val_size

    logger.info(f"Splitting dataset: {train_size}训练样本, {val_size}验证样本")

    # 设置随机种子以确保分割的可重复性
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    logger.info("Convert train dataset to dataloader")
    train_dataloader = create_dataloader(
        train_dataset,
        is_train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_mem=args.pin_mem,
        dist_eval=args.dist_eval,
    )

    logger.info("Convert validation dataset to dataloader")
    val_dataloader = create_dataloader(
        val_dataset,
        is_train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_mem=args.pin_mem,
        dist_eval=args.dist_eval,
    )

    return train_dataloader, val_dataloader, rna_vocab_size, atac_vocab_size

def create_perturbation_dataset(args, is_train=True):

    logger.info("Prepare dataset")
    dataset, rna_vocab_size, atac_vocab_size = prepare_dataloader(args)

    total_size = len(dataset)


    logger.info(f"Splitting dataset: {total_size} test samples")


    logger.info("Convert test dataset to dataloader")
    test_dataloader = create_dataloader(
        dataset,
        is_train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_mem=args.pin_mem,
        dist_eval=args.dist_eval,
    )

    return test_dataloader, rna_vocab_size, atac_vocab_size