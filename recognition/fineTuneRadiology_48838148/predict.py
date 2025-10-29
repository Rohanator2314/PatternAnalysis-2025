# predict.py
# Load the trained checkpoint and run inference on raw text inputs.
import argparse
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


def generate_summaries(model_dir: str, inputs: List[str], is_lora: bool = False, max_new_tokens: int = 128, num_beams: int = 4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if is_lora:
        base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        model = PeftModel.from_pretrained(base, model_dir).to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

    enc = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(device)
    output_ids = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned model (e.g., outputs/best_model)")
    parser.add_argument("--input_text", nargs="*", default=[
    "FINDINGS: There is a left lower lobe consolidation consistent with pneumonia. No pleural effusion.\nIMPRESSION: Left lower lobe pneumonia."
    ])
    parser.add_argument("--is_lora", action="store_true", help="Whether to use LoRA")
    args = parser.parse_args()


    preds = generate_summaries(args.model_dir, args.input_text, args.is_lora)
    for i, (src, pred) in enumerate(zip(args.input_text, preds), start=1):
        print(f"Case {i}\nSOURCE:\n{src}\n---\nSUMMARY:\n{pred}\n")


if __name__ == "__main__":
    main()
