from dataclasses import dataclass
from typing import Optional, List

import importlib
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
