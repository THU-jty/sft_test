"""
MMLU Evaluation Script for fine-tuned Qwen3-4B model.

Evaluates on the MMLU test split (14,042 questions across 57 subjects),
reports per-subject accuracy, per-category accuracy, and overall accuracy.

Usage:
    # Evaluate LoRA checkpoint (before merging)
    python evaluate.py --checkpoint_dir ./output/qwen3-4b-mmlu-lora/final_checkpoint

    # Evaluate merged full model
    python evaluate.py --model_dir ./output/qwen3-4b-mmlu-merged

    # Evaluate base model (for comparison)
    python evaluate.py --model_dir Qwen/Qwen3-4B-Instruct-2507

    # Compare base vs fine-tuned
    python evaluate.py --checkpoint_dir ./output/qwen3-4b-mmlu-lora/final_checkpoint --compare_base
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    DataConfig,
    EvalConfig,
    MMLU_SUBJECT_CATEGORIES,
    ModelConfig,
    SUBJECT_TO_CATEGORY,
)
from data_utils import CHOICE_LABELS, load_test_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model on MMLU test set")
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument(
        "--model_dir", type=str, default=None,
        help="Path to full model (merged or base model)",
    )
    parser.add_argument(
        "--base_model", type=str, default=None,
        help="Base model name (used when loading LoRA checkpoint)",
    )
    parser.add_argument(
        "--compare_base", action="store_true",
        help="Also evaluate base model for comparison",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument(
        "--no_flash_attn", action="store_true",
        help="Disable flash attention",
    )
    parser.add_argument(
        "--max_subjects", type=int, default=None,
        help="Limit number of subjects for quick testing",
    )
    return parser.parse_args()


def load_model_for_eval(
    model_dir: str = None,
    checkpoint_dir: str = None,
    base_model: str = None,
    use_flash_attn: bool = True,
):
    """Load model for evaluation: either a full model or base + LoRA."""
    model_cfg = ModelConfig()
    if base_model:
        model_cfg.model_name_or_path = base_model

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if model_dir:
        logger.info(f"Loading full model from: {model_dir}")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
        except Exception:
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    elif checkpoint_dir:
        base_name = base_model or model_cfg.model_name_or_path
        logger.info(f"Loading base model: {base_name}")
        try:
            model = AutoModelForCausalLM.from_pretrained(base_name, **model_kwargs)
        except Exception:
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(base_name, **model_kwargs)
        logger.info(f"Loading LoRA adapter from: {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, checkpoint_dir)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    else:
        raise ValueError("Either --model_dir or --checkpoint_dir must be provided")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def extract_answer(text: str) -> str:
    """Extract the answer letter (A/B/C/D) from model output."""
    text = text.strip()

    pattern = r"(?:The answer is|the answer is|Answer:?|answer:?)\s*([A-D])"
    match = re.search(pattern, text)
    if match:
        return match.group(1)

    for char in text:
        if char in "ABCD":
            return char

    return ""


def evaluate_single_example(
    model,
    tokenizer,
    question: str,
    choices: list,
    system_prompt: str,
) -> str:
    """Generate model answer for a single MMLU question."""
    choices_text = "\n".join(
        f"{CHOICE_LABELS[i]}. {choices[i]}" for i in range(len(choices))
    )
    user_content = f"{question}\n\n{choices_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return extract_answer(response)


def run_evaluation(model, tokenizer, test_ds, data_cfg: DataConfig, max_subjects=None):
    """Run full MMLU evaluation and return results."""
    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)
    overall_correct = 0
    overall_total = 0

    subjects_in_data = set()
    if "subject" in test_ds.column_names:
        subjects_in_data = set(test_ds["subject"])
        if max_subjects:
            subjects_in_data = set(sorted(subjects_in_data)[:max_subjects])

    logger.info(f"Evaluating on {len(test_ds)} examples...")

    for example in tqdm(test_ds, desc="Evaluating"):
        subject = example.get("subject", "unknown")

        if max_subjects and subject not in subjects_in_data:
            continue

        answer_idx = example["answer"]
        if isinstance(answer_idx, str):
            correct_answer = answer_idx
        else:
            correct_answer = CHOICE_LABELS[answer_idx]

        predicted = evaluate_single_example(
            model, tokenizer,
            example["question"], example["choices"],
            data_cfg.system_prompt,
        )

        is_correct = predicted == correct_answer
        subject_correct[subject] += int(is_correct)
        subject_total[subject] += 1
        overall_correct += int(is_correct)
        overall_total += 1

    subject_accuracy = {}
    for subject in sorted(subject_total.keys()):
        acc = subject_correct[subject] / subject_total[subject] if subject_total[subject] > 0 else 0.0
        subject_accuracy[subject] = {
            "correct": subject_correct[subject],
            "total": subject_total[subject],
            "accuracy": round(acc * 100, 2),
        }

    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    for subject, stats in subject_accuracy.items():
        category = SUBJECT_TO_CATEGORY.get(subject, "Other")
        category_correct[category] += stats["correct"]
        category_total[category] += stats["total"]

    category_accuracy = {}
    for category in sorted(category_total.keys()):
        acc = category_correct[category] / category_total[category] if category_total[category] > 0 else 0.0
        category_accuracy[category] = {
            "correct": category_correct[category],
            "total": category_total[category],
            "accuracy": round(acc * 100, 2),
        }

    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0

    return {
        "overall": {
            "correct": overall_correct,
            "total": overall_total,
            "accuracy": round(overall_acc * 100, 2),
        },
        "by_category": category_accuracy,
        "by_subject": subject_accuracy,
    }


def print_results(results: dict, label: str = ""):
    """Pretty-print evaluation results."""
    header = f"MMLU Evaluation Results"
    if label:
        header += f" ({label})"
    logger.info("=" * 70)
    logger.info(header)
    logger.info("=" * 70)

    logger.info(
        f"\nOverall Accuracy: {results['overall']['accuracy']:.2f}% "
        f"({results['overall']['correct']}/{results['overall']['total']})"
    )

    logger.info("\n--- By Category ---")
    for category, stats in sorted(results["by_category"].items()):
        logger.info(
            f"  {category:20s}: {stats['accuracy']:6.2f}% "
            f"({stats['correct']:4d}/{stats['total']:4d})"
        )

    logger.info("\n--- By Subject ---")
    for subject, stats in sorted(results["by_subject"].items()):
        logger.info(
            f"  {subject:45s}: {stats['accuracy']:6.2f}% "
            f"({stats['correct']:3d}/{stats['total']:3d})"
        )
    logger.info("=" * 70)


def print_comparison(base_results: dict, ft_results: dict):
    """Print side-by-side comparison of base vs fine-tuned."""
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON: Base Model vs Fine-Tuned Model")
    logger.info("=" * 70)

    base_acc = base_results["overall"]["accuracy"]
    ft_acc = ft_results["overall"]["accuracy"]
    diff = ft_acc - base_acc
    logger.info(f"\nOverall: {base_acc:.2f}% -> {ft_acc:.2f}% ({"+" if diff >= 0 else ""}{diff:.2f}%)")

    logger.info("\n--- By Category ---")
    all_categories = set(
        list(base_results["by_category"].keys()) +
        list(ft_results["by_category"].keys())
    )
    for cat in sorted(all_categories):
        b = base_results["by_category"].get(cat, {}).get("accuracy", 0)
        f = ft_results["by_category"].get(cat, {}).get("accuracy", 0)
        d = f - b
        logger.info(f"  {cat:20s}: {b:6.2f}% -> {f:6.2f}% ({"+" if d >= 0 else ""}{d:.2f}%)")

    logger.info("=" * 70)


def main():
    args = parse_args()

    data_cfg = DataConfig()
    eval_cfg = EvalConfig()

    if args.batch_size:
        eval_cfg.per_device_batch_size = args.batch_size
    if args.output_file:
        eval_cfg.output_file = args.output_file

    use_flash = not args.no_flash_attn

    test_ds = load_test_data(data_cfg)

    model, tokenizer = load_model_for_eval(
        model_dir=args.model_dir,
        checkpoint_dir=args.checkpoint_dir,
        base_model=args.base_model,
        use_flash_attn=use_flash,
    )

    ft_results = run_evaluation(model, tokenizer, test_ds, data_cfg, args.max_subjects)
    print_results(ft_results, label="Fine-Tuned" if args.checkpoint_dir else "Model")

    del model
    torch.cuda.empty_cache()

    base_results = None
    if args.compare_base:
        base_model_name = args.base_model or ModelConfig().model_name_or_path
        logger.info(f"\nNow evaluating base model for comparison: {base_model_name}")
        base_model, base_tokenizer = load_model_for_eval(
            model_dir=base_model_name, use_flash_attn=use_flash,
        )
        base_results = run_evaluation(
            base_model, base_tokenizer, test_ds, data_cfg, args.max_subjects,
        )
        print_results(base_results, label="Base Model")
        print_comparison(base_results, ft_results)

        del base_model
        torch.cuda.empty_cache()

    output = {
        "fine_tuned": ft_results,
    }
    if base_results:
        output["base_model"] = base_results

    output_file = eval_cfg.output_file
    import os
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
