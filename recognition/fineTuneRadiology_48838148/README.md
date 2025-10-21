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

uv pip install -r .\recognition\fineTuneRadiology_48838148\requirements.txt

cmake

huggingface account required to download dataset programatically

### Requirements

### UV

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

## Model Choice

## Fine Tuning

## LoRA

## Training hyperparameters results etc

## Conclusion

## References