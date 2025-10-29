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

**Problem:**\
After getting imaging done, patients often want to know their results immediately and directly, rather than going to a GP or specialist who will brief them on their radiology report. Patients have access to their radiology reports, but often need help understanding them. In addition, tools like chatGPT are not all together trust-worthy enough to translate radiology reports into layperson-friendly summaries.\
This project aims to address these issues by fine-tuning an open-source encoder-decoder language model with radiology report inputs and lay summary outputs from the [BioLaySumm2025](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track) dataset -- increasing its accuracy and reliability.

**Finetuning Model:**\
`google/flan-t5-base` (encoder–decoder) was chosen to finetune -- this choice is discussed in detail in the [Model Choice](#model-choice) section. The model is trained with Hugging Face [`Trainer`](https://huggingface.co/docs/transformers/en/main_classes/trainer) and evaluated using [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge).

## Installation

### Requirements

- Python 3.13+
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

   ```env
   HF_TOKEN=your_huggingface_token_here
   ```

This token is required to download the dataset programmatically by running the `dataset.py` script directly. Alternatively, you can download the dataset directly from the [Hugging Face website](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track) into the `data` directory.

## Usage

### Training

1. Decide on your desired training parameters, such as the number of epochs, batch size, and learning rate.
2. Run the training script with the desired parameters:

   ```bash
   python train.py --epochs 10 --batch_size 32 --learning_rate 0.001 --lora
   ```

These are the accepted arguments for the `train.py` script:

| Argument      | Type    | Default - A100 intended |Description                                          |
|---------------|---------|-------------------------|-----------------------------------------------------|
| --model_name  | string  | "google/flan-t5-base"   | Name or path of the pretrained model to fine-tune   |
| --out_dir     | string  | "./outputs"             | Directory to save checkpoints and outputs           |
| --batch_size  | int     | 16                      | Training batch size per device                      |
| --epochs      | int     | 3                       | Number of training epochs                           |
| --lr          | float   | 6e-5                    | Initial learning rate                               |
| --grad_accum  | int     | 3                       | Gradient accumulation steps to simulate larger batch size |
| --fp16        | bool    | False                   | Use 16-bit floating point mixed precision (flag)    |
| --bf16        | bool    | False                   | Use bfloat16 mixed precision (flag)                 |
| --lora        | bool    | True                    | Enable LoRA (Low-Rank Adaptation) parameter-efficient tuning (flag) |

### Predictions
Evaluate the model with your own input:

   ```bash
   python predict.py --model_dir model.pth --input_text "Within normal limits." --is_lora
   ```

## File Structure

- `assets/` — Images and plots used in the README and analysis.
  - `layman_len_report.png` — Histogram of layman summary lengths.
  - `radiology_len_report.png` — Histogram of radiology report lengths.
  - `data-card.png`, `example_stuff.txt` — Misc assets.
- `data/` — Local cache for the BioLaySumm dataset saved by `dataset.py` at `data/BioLaySumm2025-LaymanRRG-opensource-track` (created after first download).
- `outputs/` — Training artifacts and checkpoints saved by `train.py` (e.g., `best_model` or `best_model_lora`).
- `train.py` — Fine-tunes `google/flan-t5-base` with Hugging Face Trainer (LoRA optional).
- `predict.py` — Loads a trained checkpoint and generates lay summaries (`--model_dir`, `--input_text`, `--is_lora`).
- `dataset.py` — Handles dataset download, local caching, and tokenization utilities.
- `utils.py` — Dataset cleaning and histogram/analysis helpers.
- `modules.py` — Thin model wrapper (`SummarizationModel`) and generation config.
- `pyproject.toml`, `uv.lock` — Project and dependency lock files for UV.
- `requirements.txt` — Dependencies for pip-based installation.
- `.python-version` — Python tool version pin.
- `.gitignore` — Git ignore rules.

## Dataset

The [dataset](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track) provided on Hugging face contains 4 columns:

| Column Name       | Data type | Example Data              | Description             |
| ----------------- | --------- | ------------------------- | ------------------------|
| source            | string    | `PadChest`                | The source of the data  |
| images_path       | string    | `21684..._02-012-057.png` | Associated image path   |
| radiology_report  | string    | `Within normal limits.`   | Expert report           |
| layman_report     | string    | `Everything looks normal.`| Layman report           |

The overall dataset is already split into a training split of training data, validating data and testing data. This is from a total of $`\approx 171K`$ rows:

$` 150,454\ (Train) + 10,000\ (Validate) + 10,537\ (Test) = 170,991 `$

## Data Augmentation

For the purposes of this fine tuning, only the `radiology_report` and `layman_report` features were required (the `source` and `images_path` columns are dropped during cleaning).
Therefore the dataset was filtered to only include these two features.

> The `source` column contains the extra danger of data leakage. It is not used in the fine tuning process.

In addition, because some rows rely on images as context, i.e about ~423 entries contain trivial radiology text such as "interpretation". These are filtered out during cleaning by the helpers in `utils.py` (see `_clean_split` and `clean_and_overwrite_local_cache`) before tokenization.

Finally, to optimize the training, the length of the layman report and radiology reports also need to be considered. In `utils.py` the `plot_layman_length_histogram` function was created to visualize the distribution of layman report lengths and output the percentile lengths. As can be seen from the table below, the length of the 95th percentile layman report in tokens is significantly lower than the max tokens. As such the 95th percentile length is used as the max length for the layman report, and similarly the 95th percentile radiology report length is used as the max length for the radiology report.

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

Based off this, the model truncates input tokens to a maximum of 128.

## Model Choice

- Model: `google/flan-t5-base`, an encoder–decoder Transformer instruction-tuned on a wide range of tasks, importantly including summarization. The base model offers a strong quality/compute trade-off and is small enough for consumer GPUs while remaining expressive for domain adaptation. FLAN-T5 is particularly effective for tasks like **radiology-to-layperson summarization** because it was trained on instruction-following data, making it highly capable in handling **contextual instructions**
- Why base: faster training, lower memory footprint, and reduced overfitting risk compared to larger variants, while still achieving solid summarization performance. Larger checkpoints can be swapped in later if compute allows.

## Fine Tuning

We fine-tune the `google/flan-t5-base` encoder-decoder model with a **cross-entropy loss** to predict layperson summaries from radiology reports. The encoder processes the radiology text, while the decoder generates the corresponding summary.
To steer the model towards generating **layperson-friendly summaries**, we prepend the text with "Summarize the following radiology report for a layperson:".

- **Training Setup**: The model is trained with Hugging Face's **Seq2SeqTrainer** using **ROUGE** as the evaluation metric.
- Loss function: Cross-entropy loss
- Rouge metrics: ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used to evaluate the quality of text summarization. It measures the overlap between the generated summary and the reference summary, considering different n-gram sizes (unigrams, bigrams, trigrams, etc.). ROUGE-N measures n-gram overlap, ROUGE-L measures longest common subsequence overlap, and ROUGE-S measures skip-bigram overlap. ROUGE-LSum measures skip-trigram overlap.
- HF Seq2SeqTrainer: The model is trained using Hugging Face's Seq2SeqTrainer, which provides a convenient interface for training sequence-to-sequence models. It includes features like automatic hyperparameter tuning, distributed training, and easy integration with popular libraries like PyTorch and TensorFlow. It also supports various training strategies, such as teacher forcing, which we use to fine-tune the encoder–decoder model.

We fine-tune the encoder–decoder model with teacher forcing to maximize the conditional likelihood of the lay summary given the radiology report.

- Architecture: T5-style Transformer. The encoder reads the input radiology report; the decoder generates the lay summary. Decoder cross-attends to encoder states to condition generation on the source text.
- Objective: Cross-entropy over decoder tokens (label padding set to -100 so padding is ignored in the loss).
- Data preprocessing:
  - Instruction prefix: Each input is prepended with
    "Summarize the following radiology report for a layperson:"
    This matches `DataConfig.instruction_prefix` in `dataset.py` to steer the model explicitly toward layperson summarization.
  - Sequence lengths: `max_source_len=128`, `max_target_len=128`, chosen based on the dataset length histograms (roughly around the 95th percentile).
- Tokenization: Uses the pretrained SentencePiece tokenizer that ships with the checkpoint.
- Training setup: Hugging Face `Seq2SeqTrainer` with `predict_with_generate=True` and ROUGE metrics (`rouge1`, `rouge2`, `rougeL`, `rougeLsum`). The best checkpoint is selected by `rougeLsum`.

### Optimizations

- Parameter-efficient fine-tuning (LoRA) to reduce memory usage and speed up training without updating the full model.
- Mixed precision and math optimizations:
  - `--fp16` or `--bf16` flags to enable mixed precision; TF32 enabled on Ampere+ for faster matmul where supported.
- Gradient accumulation to reach an effective larger batch size when constrained by GPU memory.
- Generation during evaluation uses `num_beams=4` and `generation_max_length=128` for consistent ROUGE scoring.

## LoRA

When `--lora` is enabled (default), LoRA adapters are applied to attention modules with:
- `r=16`, `lora_alpha=32`, `lora_dropout=0.1`, `task_type=SEQ_2_SEQ_LM`.

Only adapter parameters are updated; base weights remain frozen. Checkpoints are saved under `outputs/best_model_lora`. For inference with adapters, pass `--is_lora` to `predict.py`. Without LoRA, full fine-tuning saves to `outputs/best_model`.

## Full parameter training
For the first run of training, the model was trained without optimizations such as LoRA, gradient accumulation and input text prepending. The model was also evaluated on ROUGE-L rather than ROUGE-Lsum. The training results are shown below:

| Metric                                | Value                                |
|--------------------------------------:|-------------------------------------:|
| ROUGE-1 (final epoch)                 | 0.6778344478 (67.78%)                |
| ROUGE-2 (final epoch)                 | 0.4689090514 (46.89%)                |
| ROUGE-L (final epoch)                 | 0.6188416958 (61.88%)                |
| Total training time                   | 8041.1694 seconds (≈ 2:14:01.17)     |
| Total epochs                          | 3.0                                  |

### Sample predictions:

---

#### **CASE 1**
**SOURCE**:\
Right parahilar infiltrate and atelectasis. Increased retrocardiac density related to atelectasis and consolidation associated with right pleural effusion. Clinical data is important for correct radiological assessment.

**SUMMARY**:\
There is an area of lung inflammation and partially collapsed lung on the right side near the bronchus. There is also an increased density behind the heart, which could be due to the collapsed lung and lung tissue thickening, along with fluid buildup around the lung on the right side. It is important to have clinical data to accurately assess the radiological findings.

**SUMMARY -- FROM DATA**:\
There is a cloudiness near the right lung's airways and a part of the lung has collapsed. The area behind the heart is denser, which could be due to the collapsed lung and a possible lung infection along with fluid around the right lung. It's important to consider the patient's medical history for a proper understanding of the x-ray.

---

#### **CASE 2**
**SOURCE**:\
Calcified granuloma in the right lung vertex.

**SUMMARY**:\
There is a calcified granuloma, which is a type of hardened lump, located at the top of the right lung.

**SUMMARY -- FROM DATA**:\
There is a calcified granuloma located at the top of the right lung.

---

### Analysis
It can be seen that the model correctly translates the radiological findings while also following the same sentence structure as the actual layman report. This shows that the fine tuning has indeed had an effect, although more epochs would be needed to achieve better results and more similar language (resulting also in higher ROUGE-L scores).

results second run with lora and the "summarize: " added before input: TO COME
| Metric                                | Value                                |
|--------------------------------------:|-------------------------------------:|
| ROUGE-1 (final epoch)                 | 0.6778344478 (67.78%)                |
| ROUGE-2 (final epoch)                 | 0.4689090514 (46.89%)                |
| ROUGE-L (final epoch)                 | 0.6188416958 (61.88%)                |
| ROUGE-Lsum (final epoch)              | 0.6188210466 (61.88%)                |
| Total training time                   | 8041.1694 seconds (≈ 2:14:01.17)     |
| Total epochs                          | 3.0                                  |

### Sample predictions:

---

#### **CASE 1**
**SOURCE**:\
Right parahilar infiltrate and atelectasis. Increased retrocardiac density related to atelectasis and consolidation associated with right pleural effusion. Clinical data is important for correct radiological assessment.

**BASE MODEL (with prompt)**:\
Clinical data are important for correct radiological assessment of right parahilar infiltrate and atelectasis.

**SUMMARY**:\
The right side of the diaphragm, which is the muscle that separates the chest from the abdomen, is infected and collapsed. There is increased density behind the heart, which is related to collapsed lung and solid areas around the right lung, which is associated with fluid buildup around the lungs. Clinical data is important for a correct radiological assessment.

**SUMMARY -- FROM DATA**:\
There is a cloudiness near the right lung's airways and a part of the lung has collapsed. The area behind the heart is denser, which could be due to the collapsed lung and a possible lung infection along with fluid around the right lung. It's important to consider the patient's medical history for a proper understanding of the x-ray.

---

#### **CASE 2**
**SOURCE**:\
Calcified granuloma in the right lung vertex.

**BASE MODEL (with prompt)**:\
A calcified granuloma in the right lung vertex.

**SUMMARY**:\
A calcified granuloma is present in the right lung area.

**SUMMARY -- FROM DATA**:\
There is a calcified granuloma located at the top of the right lung.

---
Prompt: `Summarize this radiology report:\n`

### Analysis
It can be seen that the model correctly translates the radiological findings while also following the same sentence structure as the actual layman report. This shows that the fine tuning has indeed had an effect, although more epochs would be needed to achieve better results and more similar language (resulting also in higher ROUGE-L scores).

## Optimized Training
In order to optimize the training, the following steps were taken:
1. Use of LoRA -- Fine-tuning with LoRA (Low-Rank Adaptation) allows for efficient training of large models by adapting only a small number of parameters, which allow for more efficient training -- more training can be done in the same time without any significant loss in performance.
2. More suitable hyperparameters -- Adjusted the learning rate down, batch size up, gradient accumulation steps up. This takes more memory and computational resources, but is well within the limits of the A100 GPU used especially with LoRA.
3. Data augmentation -- Inputs were prepended with "Summarize this radiology report: ", this helps the model understand the context better and generate more accurate summaries, and is especially useful on `flan-t5` as the model is already fine-tuned for summarization.
4. Evaluates the best model off of ROUGE-Lsum rather than ROUGE-L (Not major, but a small change)

## Conclusion

## References
