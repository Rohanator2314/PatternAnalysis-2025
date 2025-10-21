from huggingface_hub import login
from dotenv import dotenv_values

from datasets import load_dataset

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


# Currently chatGPT functions
from typing import Dict, Any
from dataclasses import dataclass
from transformers import AutoTokenizer


@dataclass
class DataConfig:
    model_name: str
    text_col: str = "report"
    target_col: str = "summary"
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


if __name__ == "__main__":
    download_dataset()
