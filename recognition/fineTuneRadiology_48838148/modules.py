from dataclasses import dataclass
from typing import Optional, List

import importlib
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from pathlib import Path
import math
from contextlib import nullcontext
# PEFT is lazily imported when LoRA is enabled
from dataset import get_tokenised_data, DataConfig
from evaluate import load
from utils import plot_loss_curve, plot_lr_curve
import numpy as np

MODEL = "google/flan-t5-base"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    num_beams: int = 4
    length_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    no_repeat_ngram_size: int = 3
    max_input_length: int = 512


class SummarizationModel(torch.nn.Module):
    """
    Thin wrapper around an encoder–decoder LLM for conditional generation.
    """

    def __init__(self, model_name: str = MODEL, tokenizer_name_or_path: Optional[str] = None):
        super().__init__()
        tok_source = tokenizer_name_or_path or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def forward(self, **batch):
        return self.model(**batch)

    @torch.no_grad()
    def generate(
        self,
        inputs: List[str],
        gen_cfg: GenerationConfig = GenerationConfig(),
        device: Optional[torch.device] = None,
    ) -> List[str]:
        self.model.eval()

        device = device or next(self.model.parameters()).device
        enc = self.tokenizer(
            inputs,
            truncation=True,
            padding=True,
            max_length=gen_cfg.max_input_length,
            return_tensors="pt",
        ).to(device)

        outputs = self.model.generate(
            **enc,
            max_new_tokens=gen_cfg.max_new_tokens,
            num_beams=gen_cfg.num_beams,
            length_penalty=gen_cfg.length_penalty,
            no_repeat_ngram_size=gen_cfg.no_repeat_ngram_size,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def to_device(self, device: str):
        self.model.to(device)
        return self

    # --- ROUGE utilities ---

    def compute_rouge_from_ids(self, preds, labels):
        """
        Compute ROUGE given predicted and label token ids (supports -100 masking).
        Accepts numpy arrays or torch tensors (batch, seq_len).
        """
        rouge = load("rouge")
        tok = self.tokenizer

        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()

        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)

        decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)

        results = rouge.compute(
            predictions=[p.strip() for p in decoded_preds],
            references=[l.strip() for l in decoded_labels],
            use_stemmer=True,
        )
        return {
            "rouge1": results["rouge1"],
            "rouge2": results["rouge2"],
            "rougeL": results["rougeL"],
            "rougeLsum": results["rougeLsum"],
        }

    def compute_rouge_from_text(self, predictions: List[str], references: List[str]):
        """
        Compute ROUGE given lists of prediction and reference strings.
        """
        rouge = load("rouge")
        results = rouge.compute(
            predictions=[p.strip() for p in predictions],
            references=[r.strip() for r in references],
            use_stemmer=True,
        )
        return {
            "rouge1": results["rouge1"],
            "rouge2": results["rouge2"],
            "rougeL": results["rougeL"],
            "rougeLsum": results["rougeLsum"],
        }

    @torch.no_grad()
    def generate_and_rouge(
        self,
        inputs: List[str],
        references: List[str],
        gen_cfg: GenerationConfig = GenerationConfig(),
        device: Optional[torch.device] = None,
    ):
        """
        Generate summaries and compute ROUGE vs. references.
        Returns (metrics_dict, decoded_predictions).
        """
        preds = self.generate(inputs, gen_cfg=gen_cfg, device=device)
        metrics = self.compute_rouge_from_text(preds, references)
        return metrics, preds

    def compute_metrics_builder(self):
        """
        Return a Trainer-compatible compute_metrics callable that consumes (preds, labels) ids.
        """
        tok = self.tokenizer
        rouge = load("rouge")

        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            preds = np.where(preds != -100, preds, tok.pad_token_id)
            decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tok.pad_token_id)
            decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)
            results = rouge.compute(
                predictions=[p.strip() for p in decoded_preds],
                references=[l.strip() for l in decoded_labels],
                use_stemmer=True,
            )
            return {
                "rouge1": results["rouge1"],
                "rouge2": results["rouge2"],
                "rougeL": results["rougeL"],
                "rougeLsum": results["rougeLsum"],
            }

        return compute_metrics

    @classmethod
    def from_finetuned(cls, model_dir: str, device: Optional[str] = None):
        """
        Load a fully fine-tuned model (and tokenizer) from a directory.
        """
        sm = cls(model_name=model_dir, tokenizer_name_or_path=model_dir)
        if device is not None:
            sm.to_device(device)
        return sm

    @classmethod
    def from_lora(
        cls,
        adapter_dir: str,
        base_model_name: str = MODEL,
        tokenizer_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Load a base model and apply a LoRA adapter from adapter_dir.
        tokenizer_name_or_path can be set to adapter_dir to mimic loading tokenizer from the adapter.
        """
        # Lazy import via importlib to avoid hard dependency unless needed
        peft = importlib.import_module("peft")
        PeftModel = getattr(peft, "PeftModel")

        sm = cls(model_name=base_model_name, tokenizer_name_or_path=tokenizer_name_or_path or adapter_dir)
        sm.model = PeftModel.from_pretrained(sm.model, adapter_dir)
        if device is not None:
            sm.to_device(device)
        return sm


class Train:
    """
    Fine-tuning trainer for Seq2Seq models (supports full and LoRA).

    Usage:
        t = Train(model_name="google/flan-t5-base", out_dir="./outputs", batch_size=16, epochs=3, lr=6e-5, grad_accum=3, fp16=False, bf16=False, lora=True)
        t.train()
        t.save(kind="last")  # or kind="best"
    """

    def __init__(
        self,
        model_name: str = MODEL,
        out_dir: str = "./outputs",
        batch_size: int = 16,
        epochs: int = 3,
        lr: float = 6e-5,
        grad_accum: int = 3,
        fp16: bool = False,
        bf16: bool = False,
        lora: bool = True,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.03,
        log_every: int = 50,
    ):
        if grad_accum < 1:
            raise ValueError(f"Invalid grad_accum={grad_accum}. It must be >= 1.")
        self.lora = lora

        self.model_name = model_name
        self.out_dir = Path(out_dir)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.grad_accum = grad_accum
        self.fp16 = fp16
        self.bf16 = bf16
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.log_every = log_every
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # Optionally apply LoRA (lazy import)
        if self.lora:
            peft = importlib.import_module("peft")
            LoraConfig = getattr(peft, "LoraConfig")
            TaskType = getattr(peft, "TaskType")
            get_peft_model = getattr(peft, "get_peft_model")
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            self.model = get_peft_model(base_model, lora_cfg)
        else:
            self.model = base_model
        # Disable cache during training for memory/perf stability
        if hasattr(base_model, "config"):
            base_model.config.use_cache = False
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        self.model.to(self.device)

        # Data
        cfg = DataConfig(model_name=self.model_name)
        ds = get_tokenised_data(cfg, splits=["train", "validation"])
        collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        pin_memory = self.device.type == "cuda"

        self.train_loader = DataLoader(
            ds["train"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collator,
            pin_memory=pin_memory,
        )
        self.val_loader = DataLoader(
            ds["validation"],
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
            pin_memory=pin_memory,
        )
        if len(self.train_loader) == 0:
            raise ValueError("Empty training dataloader: no batches to train on.")

        # Optimizer & scheduler (cosine with warmup)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)

        steps_per_epoch = math.ceil(len(self.train_loader) / max(1, self.grad_accum))
        self.total_steps = steps_per_epoch * self.epochs
        self.warmup_steps = max(0, int(self.warmup_ratio * self.total_steps))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )

        # AMP settings
        if self.device.type == "cuda":
            if self.bf16 and torch.cuda.is_bf16_supported():
                self.autocast_dtype = torch.bfloat16
            elif self.fp16:
                self.autocast_dtype = torch.float16
            else:
                self.autocast_dtype = None
        else:
            self.autocast_dtype = None
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.device.type == "cuda" and self.autocast_dtype == torch.float16)
        )

        # Logging/metrics
        self.loss_history: list[float] = []
        self.lr_history: list[float] = []
        self.best_val_loss: float = float("inf")
        # Per-epoch statistics
        self.epoch_times_sec: list[float] = []
        self.epoch_gpu_peak_mem_bytes: list[int] = []

        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _train_one_epoch(self):
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        device = self.device
        scaler = self.scaler
        grad_accum_steps = self.grad_accum
        log_every = self.log_every
        use_autocast = (device.type == "cuda") and (self.autocast_dtype is not None)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        opt_step_count = 0

        for step, batch in enumerate(self.train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            # Skip fully-masked label batches
            labels = batch.get("labels", None)
            if labels is not None and torch.all(labels == -100):
                continue

            cm = torch.autocast(device_type="cuda", dtype=self.autocast_dtype) if use_autocast else nullcontext()
            with cm:
                outputs = model(**batch)
                loss = outputs.loss

            # Guard non-finite loss
            if not torch.isfinite(loss):
                print("[warn] Non-finite loss encountered; skipping step.")
                optimizer.zero_grad(set_to_none=True)
                continue

            # Gradient accumulation
            loss = loss / grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item()

            if step % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Guard against NaN/Inf gradients
                bad_grad = False
                for p in model.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        bad_grad = True
                        break
                if bad_grad:
                    print("[warn] Detected NaN/Inf in gradients; skipping optimizer/scheduler step.")
                    optimizer.zero_grad(set_to_none=True)
                    running_loss = 0.0
                    continue

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Logging
                opt_step_count += 1
                current_loss = running_loss
                running_loss = 0.0
                current_lr = optimizer.param_groups[0]["lr"]
                self.loss_history.append(current_loss)
                self.lr_history.append(current_lr)

                if opt_step_count % log_every == 0:
                    print(f"[train] opt_step={opt_step_count} loss={current_loss:.4f} lr={current_lr:.6g}")

    @torch.no_grad()
    def _evaluate(self) -> float:
        model = self.model
        device = self.device
        use_autocast = (device.type == "cuda") and (self.autocast_dtype is not None)

        model.eval()
        total_loss = 0.0
        count = 0

        for batch in self.val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.get("labels", None)
            if labels is not None and torch.all(labels == -100):
                continue

            cm = torch.autocast(device_type="cuda", dtype=self.autocast_dtype) if use_autocast else nullcontext()
            with cm:
                outputs = model(**batch)
                loss = outputs.loss

            if not torch.isfinite(loss):
                print("[warn] Non-finite val loss encountered; skipping batch.")
                continue

            total_loss += loss.item()
            count += 1

        return total_loss / max(1, count)

    def train(self):
        import time
        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")
            # Reset CUDA peak memory stats for this epoch
            if self.device.type == "cuda":
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
            t0 = time.time()
            self._train_one_epoch()
            val_loss = self._evaluate()
            dt = time.time() - t0
            # Gather peak GPU memory usage for this epoch
            peak_bytes = 0
            if self.device.type == "cuda":
                try:
                    peak_bytes = torch.cuda.max_memory_allocated()
                except Exception:
                    peak_bytes = 0
            self.epoch_times_sec.append(dt)
            self.epoch_gpu_peak_mem_bytes.append(int(peak_bytes))
            peak_gb = peak_bytes / (1024 ** 3)
            print(f"[val] epoch={epoch} val_loss={val_loss:.4f} (epoch time {dt/60:.1f} min)")
            print(f"[stats] epoch={epoch} time={dt:.2f}s peak_gpu_mem={peak_gb:.2f} GB")

            # Save best and last
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_dir = self.out_dir / ("best_model_lora" if self.lora else "best_model")
                print(f"New best val_loss={self.best_val_loss:.4f}. Saving to {best_dir} ...")
                self._save_model(best_dir)

            self._save_model(self.out_dir / ("last_model_lora" if self.lora else "last_model"))

        # Plots
        plot_loss_curve(self.loss_history, str(self.out_dir / "loss_curve.png"))
        plot_lr_curve(self.lr_history, str(self.out_dir / "lr_curve.png"))

        # Save training statistics
        import json
        stats = {
            "epoch_times_sec": self.epoch_times_sec,
            "epoch_gpu_peak_mem_bytes": self.epoch_gpu_peak_mem_bytes,
            "epoch_gpu_peak_mem_gb": [round(b / (1024 ** 3), 6) for b in self.epoch_gpu_peak_mem_bytes],
            "best_val_loss": self.best_val_loss,
            "epochs": self.epochs,
        }
        with open(self.out_dir / "training_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    def _save_model(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out_dir))
        self.tokenizer.save_pretrained(str(out_dir))

    def save(self, kind: str = "last"):
        """
        Save the current model and tokenizer.

        Args:
            kind: "last" (default) or "best"
        """
        if kind == "best":
            out_dir = self.out_dir / "best_model_lora"
        else:
            out_dir = self.out_dir / "last_model_lora"
        print(f"Saving model to {out_dir} ...")
        self._save_model(out_dir)
