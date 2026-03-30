from datasets import Dataset, concatenate_datasets
from pathlib import Path
from typing import Union
import logging




# get logger
logger = logging.getLogger(__name__)


def tokenized_dict_dataset_to_huggingface_dataset(dict_dataset, batch_size=10000):
    logger.info("Start converting dict dataset to huggingface dataset")
    def data_generator(dict_dataset):
        for value in dict_dataset.values():
            yield value

    huggingface_datasets = []
    current_batch = []
    for idx, data in enumerate(data_generator(dict_dataset)):
        current_batch.append(data)
        if len(current_batch) >= batch_size:
            huggingface_batch = Dataset.from_list(current_batch)
            huggingface_batch.set_format(type="torch")
            huggingface_datasets.append(huggingface_batch)
            logger.info(f"Processed batch {len(huggingface_datasets)}")


            current_batch = []

    if current_batch:
        huggingface_batch = Dataset.from_list(current_batch)
        huggingface_batch.set_format(type="torch")
        huggingface_datasets.append(huggingface_batch)
        logger.info(f"Processed batch {len(huggingface_datasets)}")

    logger.info("Combining all batches into a single dataset")
    huggingface_ds = concatenate_datasets(huggingface_datasets)
    logger.info("Finish converting dict dataset to huggingface dataset")

    return huggingface_ds






def save_huggingface_dataset(dataset, dataset_path: str):
    logger.info("Start saving huggingface dataset to disk")
    dataset.save_to_disk(dataset_path)
    logger.info("Finish saving huggingface dataset to disk")


def load_huggingface_dataset(dataset_path: Union[str, Path]):
    logger.info("Start loading huggingface dataset from disk")
    return Dataset.load_from_disk(dataset_path)
    logger.info("Finish loading huggingface dataset from disk")













































