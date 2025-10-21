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

### UV

## Usage

## File Structure

## Dataset

The dataset provided on Hugging face contains 4 colums:

| Column Name:  | `source`  | `images_path` | `radiology_report`        | `layman_report`           |
| Data type:    | `string`  | `string`      | `string`                  | `string`                  |
| Example Data: | `PadChest`| `21....png`   | `Within normal limits.`   | `Everything looks normal.`|

## Data Augmentation

## Model Choice

## Fine Tuning

## LoRA

## Training hyperparameters results etc

## Conclusion

## References