# predict.py
# Load the trained checkpoint and run inference on raw text inputs.
import argparse
import random
from typing import List

import torch

from modules import SummarizationModel, GenerationConfig
from dataset import load_local_dataset, DataConfig


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
    inputs = ["Summarize this radiology report into layman terms:\n" + text for text in inputs]

    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, num_beams=num_beams)
    return sm.generate(inputs=inputs, gen_cfg=gen_cfg, device=torch.device(device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned model (e.g., outputs/best_model)")
    parser.add_argument("--is_lora", action="store_true", help="Whether to use LoRA")
    mex = parser.add_mutually_exclusive_group()
    mex.add_argument("--validate", action="store_true", help="Evaluate ROUGE on the validation split instead of manual inputs")
    mex.add_argument("--random_samples", type=int, help="Run quick comparison on N randomly sampled validation texts")
    mex.add_argument("--input_text", nargs="*", default=[
    "FINDINGS: There is a left lower lobe consolidation consistent with pneumonia. No pleural effusion.\nIMPRESSION: Left lower lobe pneumonia."
    ])
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for validation generation")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=4)
    args = parser.parse_args()

    if args.validate:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load model
        if args.is_lora:
            sm = SummarizationModel.from_lora(
                adapter_dir=args.model_dir,
                base_model_name="google/flan-t5-base",
                tokenizer_name_or_path=args.model_dir,
                device=device,
            )
        else:
            sm = SummarizationModel.from_finetuned(model_dir=args.model_dir, device=device)

        # Load raw validation split
        raw_ds = load_local_dataset()
        cfg = DataConfig(model_name="google/flan-t5-base")
        val = raw_ds["validation"]
        texts = val[cfg.text_col]
        refs = val[cfg.target_col]
        # Build prompted inputs like training
        prefixed_inputs = [f"{cfg.instruction_prefix}{t}" for t in texts]

        # Batched generation
        preds: List[str] = []
        bs = max(1, int(args.batch_size))
        gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
        for i in range(0, len(prefixed_inputs), bs):
            batch_inputs = prefixed_inputs[i:i+bs]
            batch_preds = sm.generate(inputs=batch_inputs, gen_cfg=gen_cfg, device=torch.device(device))
            preds.extend(batch_preds)

        # Compute ROUGE
        metrics = sm.compute_rouge_from_text(preds, refs)
        print("Validation ROUGE:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return

    if args.random_samples is not None:
        # Sample N random texts from validation set and compare summaries
        raw_ds = load_local_dataset()
        cfg = DataConfig(model_name="google/flan-t5-base")
        val = raw_ds["validation"]
        texts = val[cfg.text_col]
        refs = val[cfg.target_col]
        n = max(1, int(args.random_samples))
        n = min(n, len(texts))
        idxs = random.sample(range(len(texts)), n)
        sampled_inputs = [texts[i] for i in idxs]
        sampled_refs = [refs[i] for i in idxs]

        preds = generate_summaries(args.model_dir, sampled_inputs, args.is_lora, args.max_new_tokens, args.num_beams)
        bases = generate_summaries_base(sampled_inputs, args.max_new_tokens, args.num_beams)
        for i, (src, ref, pred, base) in enumerate(zip(sampled_inputs, sampled_refs, preds, bases), start=1):
            print(f"Case {i}\nSOURCE:\n{src}\n---\nSUMMARY -- FROM DATA:\n{ref}\n---\nSUMMARY:\n{pred}\n---\nBASE WITH PROMPT:\n{base}\n")
        return

    # Default: quick manual inputs compare to base
    preds = generate_summaries(args.model_dir, args.input_text, args.is_lora, args.max_new_tokens, args.num_beams)
    bases = generate_summaries_base(args.input_text, args.max_new_tokens, args.num_beams)
    for i, (src, pred, base) in enumerate(zip(args.input_text, preds, bases), start=1):
        print(f"Case {i}\nSOURCE:\n{src}\n---\nSUMMARY:\n{pred}\n---\nBASE WITH PROMPT:\n{base}\n")


if __name__ == "__main__":
    main()
