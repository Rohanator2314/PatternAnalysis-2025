# predict.py
# Load the trained checkpoint and run inference on raw text inputs.
import argparse
from typing import List

import torch

from modules import SummarizationModel, GenerationConfig


def generate_summaries(model_dir: str, inputs: List[str], is_lora: bool = False, max_new_tokens: int = 128, num_beams: int = 4):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_lora:
        sm = SummarizationModel.from_lora(
            adapter_dir=model_dir,
            base_model_name="google/flan-t5-base",
            tokenizer_name_or_path=model_dir,
            device=device,
        )
    else:
        sm = SummarizationModel.from_finetuned(model_dir=model_dir, device=device)

    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, num_beams=num_beams)
    return sm.generate(inputs=inputs, gen_cfg=gen_cfg, device=torch.device(device))


def generate_summaries_base(inputs: List[str], max_new_tokens: int = 128, num_beams: int = 4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sm = SummarizationModel.from_finetuned(model_dir="google/flan-t5-base", device=device)

    # Add 'Summarize this radiology report:\n' to the input text
    inputs = ["Summarize this radiology report:\n" + text for text in inputs]

    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, num_beams=num_beams)
    return sm.generate(inputs=inputs, gen_cfg=gen_cfg, device=torch.device(device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned model (e.g., outputs/best_model)")
    parser.add_argument("--input_text", nargs="*", default=[
    "FINDINGS: There is a left lower lobe consolidation consistent with pneumonia. No pleural effusion.\nIMPRESSION: Left lower lobe pneumonia."
    ])
    parser.add_argument("--is_lora", action="store_true", help="Whether to use LoRA")
    args = parser.parse_args()


    preds = generate_summaries(args.model_dir, args.input_text, args.is_lora)
    bases = generate_summaries_base(args.input_text)
    for i, (src, pred, base) in enumerate(zip(args.input_text, preds, bases), start=1):
        print(f"Case {i}\nSOURCE:\n{src}\n---\nSUMMARY:\n{pred}\n---\nBASE:\n{base}\n")


if __name__ == "__main__":
    main()
