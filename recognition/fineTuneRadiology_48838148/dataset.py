"""Dataset utilities for BioLaySumm radiology summarization: download, caching, tokenization, and configuration."""
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import dotenv_values

from huggingface_hub import login
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from transformers import AutoTokenizer

from utils import clean_and_overwrite_local_cache

# Where to cache the dataset on disk after first download
DATASET_REPO_ID = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
DATASET_DISK_PATH = "data/BioLaySumm2025-LaymanRRG-opensource-track"
SPLITS = ["train", "validation", "test"]

ds = None


def download_dataset():
    """Download the dataset from Hugging Face, cache it to disk, and clean the local cache.

    Expects an HF token in .env as HF_TOKEN; saves to DATASET_DISK_PATH and runs clean_and_overwrite_local_cache.
    """
    config = dotenv_values(".env")
    if not "HF_TOKEN" in config or not config["HF_TOKEN"]:
        raise ValueError("HF_TOKEN not found in .env file")

    login(config["HF_TOKEN"])

    remote_ds = load_dataset(DATASET_REPO_ID)
    remote_ds.save_to_disk(DATASET_DISK_PATH)

    clean_and_overwrite_local_cache()


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
    """Configuration for dataset columns, max sequence lengths, and the instruction prefix used for prompting."""
    model_name: str
    text_col: str = "radiology_report"
    target_col: str = "layman_report"
    max_source_len: int = 128 # ~p95 token length from README histogram (106)
    max_target_len: int = 128 # ~p95 token length from README histogram (127)
    # Instruction prefix to prepend to the source text for T5-style prompting
    instruction_prefix: str = "Summarize the following radiology report for a layperson:\n"


def make_tokenizer(model_name: str):
    """Create and return a fast Hugging Face tokenizer for the given model name."""
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def tokenise_function(cfg: DataConfig):
    """Build a dataset.map-compatible function that tokenizes inputs and targets according to DataConfig."""
    tok = make_tokenizer(cfg.model_name)

    def _fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize a batch of examples and attach label input_ids under 'labels'."""

        prefixed_inputs = [f"{cfg.instruction_prefix}{t}" for t in batch[cfg.text_col]]
        model_inputs = tok(
            prefixed_inputs,
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
    """Tokenize one split after validating columns and dropping extraneous ones."""
    dataset = datasetDict[split]

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

    return tokenised_datasets


def get_tokenised_data(cfg: DataConfig, splits: list[str] = SPLITS) -> DatasetDict:
    """Return a DatasetDict of tokenized splits using the provided configuration."""
    datasetDict = load_local_dataset()

    return DatasetDict(
        {split: get_tokenised_split(cfg, datasetDict, split) for split in splits}
    )


if __name__ == "__main__":
    download_dataset()
    ds = load_local_dataset()
    for split in ds.keys():
        print(split, ds[split].column_names, len(ds[split]))
