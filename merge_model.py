"""
Merge LoRA adapters into the base model and save as a standalone model.

The merged model can be loaded directly without peft, making it easier
to deploy or share.

Usage:
    python merge_model.py --checkpoint_dir ./output/qwen3-4b-mmlu-lora/final_checkpoint
    python merge_model.py --checkpoint_dir ./output/qwen3-4b-mmlu-lora/final_checkpoint --output_dir ./output/qwen3-4b-mmlu-merged
"""

import argparse
import logging
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to the LoRA checkpoint directory",
    )
    parser.add_argument(
        "--base_model", type=str, default=None,
        help="Base model name or path (defaults to config)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/qwen3-4b-mmlu-merged",
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true",
        help="Push merged model to Hugging Face Hub",
    )
    parser.add_argument(
        "--hub_model_id", type=str, default=None,
        help="Model ID for HuggingFace Hub (e.g., 'username/model-name')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_cfg = ModelConfig()
    base_model_name = args.base_model or model_cfg.model_name_or_path

    logger.info("=" * 60)
    logger.info("Merging LoRA Adapter into Base Model")
    logger.info("=" * 60)
    logger.info(f"Base model:  {base_model_name}")
    logger.info(f"LoRA checkpoint: {args.checkpoint_dir}")
    logger.info(f"Output dir:  {args.output_dir}")

    logger.info("\nStep 1/4: Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    logger.info("Step 2/4: Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.checkpoint_dir)

    logger.info("Step 3/4: Merging weights...")
    model = model.merge_and_unload()

    logger.info("Step 4/4: Saving merged model...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True,
    )
    tokenizer.save_pretrained(args.output_dir)

    logger.info("=" * 60)
    logger.info(f"Merged model saved to: {args.output_dir}")
    logger.info("=" * 60)

    if args.push_to_hub and args.hub_model_id:
        logger.info(f"Pushing to Hub: {args.hub_model_id}")
        model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        logger.info("Push complete!")

    logger.info("\nTo evaluate the merged model:")
    logger.info(f"  python evaluate.py --model_dir {args.output_dir}")


if __name__ == "__main__":
    main()
