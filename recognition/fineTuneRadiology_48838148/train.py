# train.py
# Fine-tune FLAN-T5/T5 on radiology→lay summaries with Hugging Face Trainer.
import argparse
from pathlib import Path

import numpy as np
from evaluate import load
from datasets import DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer as Trainer,
    Seq2SeqTrainingArguments as TrainingArguments,
)

from dataset import get_tokenised_data, DataConfig

# WARN: Enable TF32 only on supported GPUs
import torch
torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.backends.cuda.matmul.fp32_precision = 'ieee'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def build_datasets(cfg: DataConfig):
    ds = get_tokenised_data(cfg, splits=["train", "validation"])

    return ds


def compute_metrics_builder(tokenizer):
    # ROUGE for summaries
    rouge = load("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # decode
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # replace -100 in labels as well
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        results = rouge.compute(
            predictions=[p.strip() for p in decoded_preds],
            references=[l.strip() for l in decoded_labels],
            use_stemmer=True,
        )
        # return a few standard aggregates
        return {
            # TODO: no such thing as mid feature?
            "rouge1": results["rouge1"].mid.fmeasure,
            "rouge2": results["rouge2"].mid.fmeasure,
            "rougeL": results["rougeL"].mid.fmeasure,
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/flan-t5-base")
    parser.add_argument("--out_dir", default="./outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    if args.bf16:
        print("Using bfloat16 training")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    cfg = DataConfig(model_name=args.model_name)
    ds = build_datasets(cfg)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    compute_metrics = compute_metrics_builder(tokenizer)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.bf16,
        report_to=["none"],  # or "wandb"/"tensorboard" if desired
        predict_with_generate=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(out_dir / "best_model"))
    tokenizer.save_pretrained(str(out_dir / "best_model"))


if __name__ == "__main__":
    main()
