from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = "google/flan-t5-base"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    num_beams: int = 4
    leancy_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    no_repeat_ngram_size: int = 3


class SummarizationModel(torch.nn.Module):
    """
    Thin wrapper around an encoder–decoder LLM for conditional generation.
    """

    def __init__(self, model_name: str = MODEL):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
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
            max_length=512,
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
