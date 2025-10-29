"""
Dataset cleaning utilities for BioLaySumm fine-tuning.

Features:
- Remove rows where len(layman_report) > RATIO * len(radiology_report) (plain text char lengths).
- Remove rows that are likely image-caption cases where radiology_report == 'interpretation' and the image_path is a file.
- Remove extraneous columns ('source', 'images_path') directly in the cached dataset.
- Overwrite the local cached dataset on disk after cleaning.

Usage (as a script):
    python -m a1.recognition.fineTuneRadiology_48838148.utils \
        --data-dir data/BioLaySumm2025-LaymanRRG-opensource-track \
        --length-ratio 50

You can also import and call:
    clean_and_overwrite_local_cache(data_dir=..., length_ratio=50.0)

Notes:
- This modifies the dataset stored at data_dir in place.
- Ensure you've downloaded the dataset (via dataset.download_dataset()) before running this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import os
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_from_disk, DatasetDict, Dataset



# Defaults mirror dataset.py
DEFAULT_DATA_DIR = "data/BioLaySumm2025-LaymanRRG-opensource-track"
DEFAULT_TEXT_COL = "radiology_report"
DEFAULT_TARGET_COL = "layman_report"
DEFAULT_DROP_COLS = ("source", "images_path")


@dataclass
class CleanConfig:
    data_dir: str = DEFAULT_DATA_DIR
    text_col: str = DEFAULT_TEXT_COL
    target_col: str = DEFAULT_TARGET_COL
    length_ratio: float = 50.0
    drop_columns: Sequence[str] = DEFAULT_DROP_COLS


def _normalize_str(x: Optional[str]) -> str:
    if isinstance(x, str):
        return x
    return ""


def _is_non_empty(s: Optional[str]) -> bool:
    return isinstance(s, str) and len(s.strip()) > 0


def plot_layman_length_histogram(
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    split: str = "train",
    target_col: str = DEFAULT_TARGET_COL,
    model_name: Optional[str] = None,
    bins: int = 50,
    show: bool = True,
    save_path: Optional[str] = None,
    add_special_tokens: bool = False,
    truncation: bool = False,
) -> dict:
    """
    Compute and plot histogram(s) of layman_report lengths for a given split.

    - Always computes character and word length histograms.
    - If `model_name` is provided, also computes token length histograms using the model's tokenizer.
    - Returns a dict with arrays and summary statistics for each length type.

    Args:
        data_dir: Path to dataset saved via `save_to_disk`.
        split: One of {"train", "validation", "test"} present in the dataset.
        target_col: Column name for layman text (defaults to 'layman_report').
        model_name: Optional Hugging Face model name to tokenize with (e.g., 'google/flan-t5-base').
        bins: Number of bins for hist plots.
        show: If True, shows the plot. If False, closes the figure (useful when only saving).
        save_path: If provided, saves the figure to this path.
        add_special_tokens: Whether to include special tokens when computing token lengths.
        truncation: Whether to allow tokenizer truncation during tokenization.

    Returns:
        dict with keys: char_lens, word_lens, token_lens (optional),
        and stats: char_stats, word_stats, token_stats (optional).
    """
    ds = load_from_disk(data_dir)
    if not isinstance(ds, DatasetDict):
        raise TypeError(f"Expected a DatasetDict at {data_dir}, got: {type(ds)}")
    if split not in ds:
        raise KeyError(f"Split '{split}' not found. Available: {list(ds.keys())}")

    d = ds[split]
    if target_col not in d.column_names:
        raise KeyError(
            f"Column '{target_col}' not found in split '{split}'. "
            f"Available: {d.column_names}"
        )

    texts = [t for t in d[target_col] if _is_non_empty(t)]
    char_lens = np.array([len(t) for t in texts], dtype=np.int32)
    word_lens = np.array([len(t.split()) for t in texts], dtype=np.int32)

    token_lens = None
    if model_name:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenized = tok(
            texts,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
        )
        token_lens = np.array([len(ids) for ids in tokenized["input_ids"]], dtype=np.int32)

    # Plot
    num_subplots = 3 if token_lens is not None else 2
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 4))

    if num_subplots == 2:
        ax_word, ax_char = axes
    else:
        ax_tok, ax_word, ax_char = axes

    if token_lens is not None:
        ax_tok.hist(token_lens, bins=bins, color="steelblue", edgecolor="white")
        ax_tok.set_title(f"Token lengths ({split})")
        ax_tok.set_xlabel("Tokens")
        ax_tok.set_ylabel("Count")

    ax_word.hist(word_lens, bins=bins, color="seagreen", edgecolor="white")
    ax_word.set_title(f"Word lengths ({split})")
    ax_word.set_xlabel("Words")
    ax_word.set_ylabel("Count")

    ax_char.hist(char_lens, bins=bins, color="indianred", edgecolor="white")
    ax_char.set_title(f"Character lengths ({split})")
    ax_char.set_xlabel("Characters")
    ax_char.set_ylabel("Count")

    fig.suptitle("Layman report length distributions")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    def _stats(arr: np.ndarray) -> dict:
        if arr.size == 0:
            return {"count": 0, "min": 0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}
        return {
            "count": int(arr.size),
            "min": int(arr.min()),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": int(arr.max()),
        }

    return {
        "char_lens": char_lens,
        "word_lens": word_lens,
        "token_lens": token_lens,
        "char_stats": _stats(char_lens),
        "word_stats": _stats(word_lens),
        "token_stats": _stats(token_lens) if token_lens is not None else None,
    }


def _clean_split(
    dset: Dataset,
    *,
    text_col: str,
    target_col: str,
    length_ratio: float,
    drop_columns: Sequence[str],
) -> Dataset:
    """
    Clean a single split:
    - Filter out rows where len(layman_report) > length_ratio * len(radiology_report).
      If radiology_report length == 0, drop if layman length > 0.
    - Filter out rows where radiology_report.strip().lower() == 'interpretation' and
      images_path is present and an image file.
    - Remove drop_columns if present.
    """
    # Sanity: keep only rows where both text and target exist and are strings
    def valid_strings_filter(batch):
        texts = batch[text_col]
        targets = batch[target_col]
        keep = []
        for t, u in zip(texts, targets):
            keep.append(_is_non_empty(t) and _is_non_empty(u))
        return keep

    dset = dset.filter(valid_strings_filter, batched=True)

    # Length-based filter
    def length_ratio_filter(batch):
        texts = batch[text_col]
        targets = batch[target_col]

        keep = []
        for t, u in zip(texts, targets):
            t_norm = _normalize_str(t)
            u_norm = _normalize_str(u)
            t_len = len(t_norm)
            u_len = len(u_norm)

            if t_len == 0:
                # If radiology report empty, drop if layman has any content
                keep.append(u_len == 0)
            else:
                keep.append(u_len <= length_ratio * t_len)
        return keep

    before_len = len(dset)
    dset = dset.filter(length_ratio_filter, batched=True)
    after_len_lr = len(dset)

    # Heuristic image-caption filter on 'interpretation' rows
    # Drop if radiology_report == 'interpretation' and (images_path is non-empty or source looks image-based)
    cols = set(dset.column_names)
    has_images_path = "images_path" in cols

    def interpretation_filter(batch):
        texts = batch[text_col]
        targets = batch[target_col]
        images = batch["images_path"] if has_images_path else [None] * len(texts)

        keep = []
        for t, ip in zip(texts, images):
            t_norm = _normalize_str(t).strip().lower()
            ip_norm = _normalize_str(ip).strip()

            # Does the image path end with .jpg or .png?
            has_image_path = bool(ip_norm) and ip_norm.endswith(('.jpg', '.png', '.jpeg', '.gif'))

            # Drop if this looks like an image-caption example with trivial radiology text
            drop = (t_norm == "interpretation") and (has_image_path)
            keep.append(not drop)
        return keep

    before_len2 = len(dset)
    dset = dset.filter(interpretation_filter, batched=True)
    after_len_ic = len(dset)

    # Remove extraneous columns
    cols_to_drop = [c for c in drop_columns if c in dset.column_names]
    if cols_to_drop:
        dset = dset.remove_columns(cols_to_drop)

    # Basic logging
    removed_len_ratio = before_len - after_len_lr
    removed_interpret = before_len2 - after_len_ic
    print(
        f"Cleaned split: start={before_len}, "
        f"after_len_ratio={after_len_lr} (-{removed_len_ratio}), "
        f"after_interpretation={after_len_ic} (-{removed_interpret}), "
        f"final={len(dset)}, dropped_cols={cols_to_drop}"
    )

    return dset


def clean_dataset(
    ds: DatasetDict,
    *,
    text_col: str = DEFAULT_TEXT_COL,
    target_col: str = DEFAULT_TARGET_COL,
    length_ratio: float = 50.0,
    drop_columns: Sequence[str] = DEFAULT_DROP_COLS,
) -> DatasetDict:
    """
    Apply cleaning to all splits in the DatasetDict. Non-existent splits are ignored.
    """
    cleaned = {}
    for split in ["train", "validation", "test"]:
        if split in ds:
            print(f"Processing split: {split}")
            cleaned[split] = _clean_split(
                ds[split],
                text_col=text_col,
                target_col=target_col,
                length_ratio=length_ratio,
                drop_columns=drop_columns,
            )
    return DatasetDict(cleaned)


def clean_and_overwrite_local_cache(
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    text_col: str = DEFAULT_TEXT_COL,
    target_col: str = DEFAULT_TARGET_COL,
    length_ratio: float = 50.0,
    drop_columns: Sequence[str] = DEFAULT_DROP_COLS,
) -> None:
    """
    Load the locally cached dataset from data_dir, clean it, and overwrite on disk.
    """
    print(f"Loading dataset from: {data_dir}")
    ds = load_from_disk(data_dir)
    assert isinstance(ds, DatasetDict), "Expected a DatasetDict at the provided data_dir."

    cleaned = clean_dataset(
        ds,
        text_col=text_col,
        target_col=target_col,
        length_ratio=length_ratio,
        drop_columns=drop_columns,
    )
    print(f"Saving cleaned dataset back to: {data_dir}")
    input("Press Enter to continue...")
    parent_dir = os.path.dirname(os.path.abspath(data_dir))
    tmp_dir = tempfile.mkdtemp(prefix="cleaned-", dir=parent_dir)
    backup_dir = data_dir + ".bak"

    try:
        # Save cleaned dataset to a temporary directory first
        cleaned.save_to_disk(tmp_dir)

        # Remove any previous backup
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)

        # Move the original dataset out of the way
        if os.path.exists(data_dir):
            os.replace(data_dir, backup_dir)

        # Atomically move the temp directory into place
        os.replace(tmp_dir, data_dir)

        # Cleanup backup
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        print("Done.")
    except Exception as e:
        # Attempt to restore from backup on failure
        if os.path.exists(backup_dir):
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir, ignore_errors=True)
            os.replace(backup_dir, data_dir)
        # Ensure temp dir is removed if still present
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        # If tmp_dir still exists (e.g., on failure), remove it
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean cached BioLaySumm dataset in place.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Path to the dataset saved with save_to_disk (default: %(default)s)",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default=DEFAULT_TEXT_COL,
        help="Column name for source text (radiology report).",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=DEFAULT_TARGET_COL,
        help="Column name for target text (layman report).",
    )
    parser.add_argument(
        "--length-ratio",
        type=float,
        default=50.0,
        help="Drop rows where len(target) > ratio * len(text) (plain text char lengths).",
    )
    parser.add_argument(
        "--drop-columns",
        type=str,
        nargs="*",
        default=list(DEFAULT_DROP_COLS),
        help="Columns to remove from the dataset (if present).",
    )

    args = parser.parse_args()

    clean_and_overwrite_local_cache(
        data_dir=args.data_dir,
        text_col=args.text_col,
        target_col=args.target_col,
        length_ratio=args.length_ratio,
        drop_columns=args.drop_columns,
    )

    # Uncomment to plot layman length histogram
    # stats = plot_layman_length_histogram(
    #     data_dir="data/BioLaySumm2025-LaymanRRG-opensource-track",
    #     split="train",
    #     target_col=DEFAULT_TARGET_COL,
    #     model_name="google/flan-t5-base",  # optional
    #     bins=60,
    #     show=True,
    #     save_path=None,
    #     add_special_tokens=False,
    #     truncation=False,
    # )
    # print(stats["token_stats"], stats["word_stats"], stats["char_stats"])
