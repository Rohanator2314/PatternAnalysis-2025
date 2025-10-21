from typing import Dict, Any
from dataclasses import dataclass
from dotenv import dotenv_values

from huggingface_hub import login
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


ds = None


def download_dataset():
    config = dotenv_values(".env")
    login(config["HF_TOKEN"])

    ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")

    ds.save_to_disk("data/BioLaySumm2025-LaymanRRG-opensource-track")


def load_local_dataset():
    global ds
    if ds is None:
        ds = load_dataset("data/BioLaySumm2025-LaymanRRG-opensource-track")
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
        model_inputs = tok(
            batch[cfg.text_col],
            max_length=cfg.max_source_len,
            truncation=True,
        )
        with tok.as_target_tokenizer():
            labels = tok(
                batch[cfg.target_col],
                max_length=cfg.max_target_len,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return _fn


def get_tokenised_data(cfg: DataConfig, split: str) -> DatasetDict:
    dataset = load_local_dataset()[split]
    tokenise_fn = tokenise_function(cfg)
    # tokenised_datasets = {}
    # for split in dataset.keys():

    columns_to_remove = []
    if "source" in ds.column_names:
        columns_to_remove.append("source")
    if "images_path" in ds.column_names:
        columns_to_remove.append("images_path")
    if columns_to_remove:
        ds = ds.remove_columns(columns_to_remove)

    tokenised_datasets = dataset.map(
        tokenise_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    # tokenised_dataset = DatasetDict(tokenised_datasets)
    return tokenised_datasets


if __name__ == "__main__":
    download_dataset()
