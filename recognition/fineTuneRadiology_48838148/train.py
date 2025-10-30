"""CLI entry point for fine-tuning Seq2Seq models (FLAN-T5) on the BioLaySumm dataset.
Parses command-line arguments, configures optional precision hints, and invokes
training using the Train helper (full fine-tuning or LoRA via --lora).
"""

import argparse
import torch

from modules import Train


def main():
    """Parse CLI arguments and run training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/flan-t5-base")
    parser.add_argument("--out_dir", default="./outputs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--grad_accum", type=int, default=3)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--lora", action="store_true")
    args = parser.parse_args()

    if args.bf16:
        print("Using bfloat16 training")
    elif args.fp16:
        print("Using float16 training")

    # Best-effort TF32 enabling on supported GPUs
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore[attr-defined]
        torch.backends.cuda.matmul.fp32_precision = "ieee"  # type: ignore[attr-defined]
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    trainer = Train(
        model_name=args.model_name,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        grad_accum=args.grad_accum,
        fp16=args.fp16,
        bf16=args.bf16,
        lora=args.lora,
    )

    trainer.train()


if __name__ == "__main__":
    main()
