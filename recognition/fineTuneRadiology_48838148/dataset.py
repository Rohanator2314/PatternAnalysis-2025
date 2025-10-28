from typing import Dict, Any
from dataclasses import dataclass
from dotenv import dotenv_values

from huggingface_hub import login
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from transformers import AutoTokenizer

# Where to cache the dataset on disk after first download
DATASET_REPO_ID = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
DATASET_DISK_PATH = "data/BioLaySumm2025-LaymanRRG-opensource-track"
SPLITS = ["train", "validation", "test"]

ds = None


def download_dataset():
    config = dotenv_values(".env")
    if not "HF_TOKEN" in config or not config["HF_TOKEN"]:
        raise ValueError("HF_TOKEN not found in .env file")

    login(config["HF_TOKEN"])

    remote_ds = load_dataset(DATASET_REPO_ID)
    remote_ds.save_to_disk(DATASET_DISK_PATH)


def load_local_dataset() -> DatasetDict:
    """
    Load the locally cached dataset that was previously saved with save_to_disk.
    """
    global ds
    if ds is None:
        ds = load_from_disk(DATASET_DISK_PATH)
    return ds


@dataclass
class DataConfig:
    model_name: str
    text_col: str = "radiology_report"
    target_col: str = "layman_report"
    max_source_len: int = 512
    max_target_len: int = 128


def make_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def tokenise_function(cfg: DataConfig):
    tok = make_tokenizer(cfg.model_name)

    def _fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        # Tokenize the inputs and targets
        model_inputs = tok(
            batch[cfg.text_col],
            max_length=cfg.max_source_len,
            truncation=True,
        )
        labels = tok(
            text_target=batch[cfg.target_col],
            max_length=cfg.max_target_len,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return _fn


def get_tokenised_split(
    cfg: DataConfig, datasetDict: DatasetDict, split: str
) -> Dataset:
    dataset = datasetDict[split]
    # Sanity-check expected columns
    required_cols = {cfg.text_col, cfg.target_col}
    missing = required_cols.difference(set(dataset.column_names))
    if missing:
        raise KeyError(
            f"Missing expected columns: {missing}. Available: {dataset.column_names}. "
            f"Check that the dataset was loaded via load_from_disk and that split='{split}' exists."
        )

    columns_to_remove = []
    if "source" in dataset.column_names:
        columns_to_remove.append("source")
    if "images_path" in dataset.column_names:
        columns_to_remove.append("images_path")
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    tokenise_fn = tokenise_function(cfg)
    tokenised_datasets = dataset.map(
        tokenise_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    # tokenised_dataset = DatasetDict(tokenised_datasets)
    return tokenised_datasets


def get_tokenised_data(cfg: DataConfig, splits: list[str] = SPLITS) -> DatasetDict:
    datasetDict = load_local_dataset()

    return DatasetDict(
        {split: get_tokenised_split(cfg, datasetDict, split) for split in splits}
    )


if __name__ == "__main__":
    download_dataset()
    # ds = load_local_dataset()
    # for split in ds.keys():
    #     print(split, ds[split].column_names, len(ds[split]))
