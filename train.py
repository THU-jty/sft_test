"""
Qwen3-4B-Instruct-2507 MMLU LoRA Fine-Tuning Script

Usage:
    python train.py
    python train.py --output_dir ./output/custom_run --num_train_epochs 5
"""

import argparse
import logging
import os
import sys

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

from config import (
    DataConfig,
    LoraConfig as LoraConfigDC,
    ModelConfig,
    TrainingConfig,
)
from data_utils import load_and_prepare_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-4B on MMLU with LoRA")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--no_flash_attn", action="store_true",
        help="Disable flash attention (use if GPU does not support it)",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_cfg: ModelConfig):
    """Load Qwen3-4B model in bf16 and its tokenizer."""
    logger.info(f"Loading model: {model_cfg.model_name_or_path}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(model_cfg.torch_dtype, torch.bfloat16)

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": model_cfg.device_map,
        "trust_remote_code": True,
    }
    if model_cfg.attn_implementation:
        model_kwargs["attn_implementation"] = model_cfg.attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_name_or_path, **model_kwargs,
        )
    except Exception:
        logger.warning(
            "Flash Attention 2 not available, falling back to default attention."
        )
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_name_or_path, **model_kwargs,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name_or_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.use_cache = False
    return model, tokenizer


def setup_lora(model, lora_cfg: LoraConfigDC):
    """Inject LoRA adapters into the model."""
    logger.info(
        f"Setting up LoRA: r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}, "
        f"targets={lora_cfg.target_modules}"
    )
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)

    trainable_params, total_params = model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    return model, peft_config


def build_training_args(train_cfg: TrainingConfig) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from our config."""
    return TrainingArguments(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        bf16=train_cfg.bf16,
        logging_steps=train_cfg.logging_steps,
        eval_strategy=train_cfg.eval_strategy,
        eval_steps=train_cfg.eval_steps,
        save_strategy=train_cfg.save_strategy,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        max_grad_norm=train_cfg.max_grad_norm,
        dataloader_num_workers=train_cfg.dataloader_num_workers,
        report_to=train_cfg.report_to,
        seed=train_cfg.seed,
        load_best_model_at_end=train_cfg.load_best_model_at_end,
        metric_for_best_model=train_cfg.metric_for_best_model,
        greater_is_better=train_cfg.greater_is_better,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        logging_first_step=True,
    )


def main():
    args = parse_args()

    model_cfg = ModelConfig()
    lora_cfg = LoraConfigDC()
    data_cfg = DataConfig()
    train_cfg = TrainingConfig()

    if args.model_name_or_path:
        model_cfg.model_name_or_path = args.model_name_or_path
    if args.output_dir:
        train_cfg.output_dir = args.output_dir
    if args.num_train_epochs:
        train_cfg.num_train_epochs = args.num_train_epochs
    if args.learning_rate:
        train_cfg.learning_rate = args.learning_rate
    if args.per_device_train_batch_size:
        train_cfg.per_device_train_batch_size = args.per_device_train_batch_size
    if args.gradient_accumulation_steps:
        train_cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.lora_r:
        lora_cfg.r = args.lora_r
    if args.lora_alpha:
        lora_cfg.lora_alpha = args.lora_alpha
    if args.max_seq_length:
        data_cfg.max_seq_length = args.max_seq_length
    if args.no_flash_attn:
        model_cfg.attn_implementation = None

    os.makedirs(train_cfg.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Qwen3-4B MMLU LoRA Fine-Tuning")
    logger.info("=" * 60)
    logger.info(f"Model: {model_cfg.model_name_or_path}")
    logger.info(f"Output: {train_cfg.output_dir}")
    logger.info(f"LoRA r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}")
    logger.info(f"Epochs: {train_cfg.num_train_epochs}, LR: {train_cfg.learning_rate}")
    logger.info(f"Batch size: {train_cfg.per_device_train_batch_size} x {train_cfg.gradient_accumulation_steps} (grad accum)")
    logger.info("=" * 60)

    model, tokenizer = load_model_and_tokenizer(model_cfg)
    model, peft_config = setup_lora(model, lora_cfg)

    dataset = load_and_prepare_data(data_cfg, tokenizer=None)
    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Val samples: {len(dataset['validation'])}")

    training_args = build_training_args(train_cfg)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        max_seq_length=data_cfg.max_seq_length,
    )

    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger.info("Saving final model...")
    trainer.save_model(os.path.join(train_cfg.output_dir, "final_checkpoint"))
    tokenizer.save_pretrained(os.path.join(train_cfg.output_dir, "final_checkpoint"))

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {train_cfg.output_dir}/final_checkpoint")
    logger.info(f"Train loss: {metrics.get('train_loss', 'N/A')}")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("  1. Evaluate: python evaluate.py --checkpoint_dir ./output/qwen3-4b-mmlu-lora/final_checkpoint")
    logger.info("  2. Merge:    python merge_model.py --checkpoint_dir ./output/qwen3-4b-mmlu-lora/final_checkpoint")


if __name__ == "__main__":
    main()
