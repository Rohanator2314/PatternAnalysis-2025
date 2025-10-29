# Generating Layperson Summaries of Expert Radiology Reports through Fine-tuning an existing LLM

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
    - [Requirements](#requirements)
    - [UV](#uv)
3. [Usage](#usage)
4. [File Structure](#file-structure)
5. [Dataset](#dataset)
6. [Data Augmentation](#data-augmentation)
7. [Model Choice](#model-choice)
8. [Fine Tuning](#fine-tuning)
9. [LoRA](#lora)
10. [Training hyperparameters results etc](#training-hyperparameters-results-etc)
11. [Conclusion](#conclusion)
12. [References](#references)

## Overview

**Problem:** Translate radiology reports into layperson-friendly summaries.

**Model:** `person/model` (encoder–decoder). Train with Hugging Face [`Trainer`](https://huggingface.co/docs/transformers/en/main_classes/trainer) and evaluate using [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge).

This project fine tunes an opensource encoder-decoder LLM with radiology report inputs and lay summary outputs from the [BioLaySumm2025](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track) dataset.

## Installation

### Requirements

- Python 3.8+
- [UV](https://docs.astral.sh/uv/) package manager (recommended) or pip
- Hugging Face account with user access token

### Setup

1. Navigate to the project directory:

   ```bash
   cd recognition/fineTuneRadiology_48838148
   ```

2. Install dependencies using one of the following methods:

   **Using UV (recommended):**

   ```bash
   uv sync
   ```

   **Using pip (works on more devices):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

3. Create a `.env` file from the example and populate your Hugging Face user access token:

   ```
   HF_TOKEN=your_huggingface_token_here
   ```

This token is required to download the dataset programmatically by running the `dataset.py` script directly. Alternatively, you can download the dataset directly from the [Hugging Face website](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track).

## Usage

## File Structure

## Dataset

The [dataset](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track) provided on Hugging face contains 4 colums:

| Column Name       | Data type | Example Data              |
| ----------------- | --------- | ------------------------- |
| source            | string    | `PadChest`                |
| images_path       | string    | `21684..._02-012-057.png` |
| radiology_report  | string    | `Within normal limits.`   |
| layman_report     | string    | `Everything looks normal.`|

The overall dataset is already split into a training split of training data, validating data and testing data. This is from a total of $`\approx 171K`$ rows:

$` 150,454\ (Train) + 10,000\ (Validate) + 10,537\ (Test) = 170,991 `$

## Data Augmentation

For the purposes of this fine tuning, only the `radiology_report` and `layman_report` features were required.

```
Loading dataset from: data/BioLaySumm2025-LaymanRRG-opensource-track
Processing split: train
Filter: 100%|███████████████████████████████████████████████| 150454/150454 [00:00<00:00, 527177.85 examples/s]
Filter: 100%|███████████████████████████████████████████████| 150454/150454 [00:00<00:00, 153764.31 examples/s]
Filter: 100%|███████████████████████████████████████████████| 150374/150374 [00:01<00:00, 125842.54 examples/s]
Cleaned split: start=150454, after_len_ratio=150374 (-80), after_interpretation=150048 (-326), final=150048, dropped_cols=['source', 'images_path']
Processing split: validation
Filter: 100%|█████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 443996.74 examples/s]
Filter: 100%|█████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 152369.99 examples/s]
Filter: 100%|███████████████████████████████████████████████████| 9996/9996 [00:00<00:00, 126034.89 examples/s]
Cleaned split: start=10000, after_len_ratio=9996 (-4), after_interpretation=9983 (-13), final=9983, dropped_cols=['source', 'images_path']
Processing split: test
Filter: 100%|█████████████████████████████████████████████████| 10537/10537 [00:00<00:00, 590176.69 examples/s]
Cleaned split: start=0, after_len_ratio=0 (-0), after_interpretation=0 (-0), final=0, dropped_cols=['source', 'images_path']
Saving cleaned dataset back to: data/BioLaySumm2025-LaymanRRG-opensource-track
Press Enter to continue...
```

**Histogram Findings -- Layman Report**:

| Metric | Count | Min | P50 | P75 | P90 | P95 | P99 | Max |
|--------|------:|----:|----:|----:|----:|----:|----:|----:|
| Tokens | 150048 | 2 | 40.0 | 67.0 | 99.0 | 127.0 | 290.0 | 1175 |
| Words  | 150048 | 1 | 30.0 | 50.0 | 73.0 | 94.0  | 208.0 | 817  |
| Chars  | 150048 | 7 | 167.0| 280.0| 408.0| 522.0 | 1163.5299999999988 | 4709 |

![Layman report length distribution](assets/layman_len_report.png)

**Histogram Findings -- Radiology Report**:

| Metric | Count | Min | P50 | P75 | P90 | P95 | P99 | Max |
|--------|------:|----:|----:|----:|----:|----:|----:|----:|
| Tokens | 150048 | 1 | 30.0 | 53.0 | 81.0 | 106.0 | 284.0 | 1725 |
| Words  | 150048 | 1 | 16.0 | 30.0 | 46.0 | 61.0  | 154.0 | 964  |
| Chars  | 150048 | 5 | 119.0| 211.0| 322.0| 419.0 | 1078.0 | 6706 |

![Radiology report length distribution](assets/radiology_len_report.png)
## Model Choice

## Fine Tuning

## LoRA

## Training hyperparameters results etc

result first run:
```
[56268/56268 3:03:30, Epoch 3/3]
```

| Epoch | Training Loss | Validation Loss | Rouge1  | Rouge2  | RougeL  |
|------:|---------------:|----------------:|--------:|--------:|--------:|
| 1     | 0.593300       | 0.485298        | 0.551873| 0.426262| 0.522039|
| 2     | 0.462300       | 0.441893        | 0.559679| 0.438793| 0.530375|
| 3     | 0.417800       | 0.424618        | 0.561618| 0.442797| 0.533015|

```
There were missing keys in the checkpoint model loaded: ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'].
TrainOutput(global_step=56268, training_loss=0.5316248244294194, metrics={'train_runtime': 11012.296, 'train_samples_per_second': 40.876, 'train_steps_per_second': 5.11, 'total_flos': 7.291082302193664e+16, 'train_loss': 0.5316248244294194, 'epoch': 3.0})
```

## Conclusion

## References
