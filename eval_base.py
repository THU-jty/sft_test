"""
快速评估 Qwen3-4B-Instruct-2507 基座模型在 MMLU 上的表现。

Usage:
    # 完整评估（14,042 题，耗时较长）
    python eval_base.py

    # 快速测试（只评估前 5 个 subject）
    python eval_base.py --max_subjects 5

    # 指定模型路径（如本地已下载）
    python eval_base.py --model_name_or_path /path/to/local/model
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

CHOICE_LABELS = ["A", "B", "C", "D"]

SUBJECT_TO_CATEGORY = {
    "abstract_algebra": "STEM", "anatomy": "STEM", "astronomy": "STEM",
    "college_biology": "STEM", "college_chemistry": "STEM",
    "college_computer_science": "STEM", "college_mathematics": "STEM",
    "college_physics": "STEM", "computer_security": "STEM",
    "conceptual_physics": "STEM", "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM", "high_school_biology": "STEM",
    "high_school_chemistry": "STEM", "high_school_computer_science": "STEM",
    "high_school_mathematics": "STEM", "high_school_physics": "STEM",
    "high_school_statistics": "STEM", "machine_learning": "STEM",
    "formal_logic": "Humanities", "high_school_european_history": "Humanities",
    "high_school_us_history": "Humanities", "high_school_world_history": "Humanities",
    "international_law": "Humanities", "jurisprudence": "Humanities",
    "logical_fallacies": "Humanities", "moral_disputes": "Humanities",
    "moral_scenarios": "Humanities", "philosophy": "Humanities",
    "prehistory": "Humanities", "professional_law": "Humanities",
    "world_religions": "Humanities",
    "econometrics": "Social Sciences", "high_school_geography": "Social Sciences",
    "high_school_government_and_politics": "Social Sciences",
    "high_school_macroeconomics": "Social Sciences",
    "high_school_microeconomics": "Social Sciences",
    "high_school_psychology": "Social Sciences",
    "human_sexuality": "Social Sciences", "professional_psychology": "Social Sciences",
    "public_relations": "Social Sciences", "security_studies": "Social Sciences",
    "sociology": "Social Sciences", "us_foreign_policy": "Social Sciences",
    "business_ethics": "Other", "clinical_knowledge": "Other",
    "college_medicine": "Other", "global_facts": "Other",
    "human_aging": "Other", "management": "Other", "marketing": "Other",
    "medical_genetics": "Other", "miscellaneous": "Other",
    "nutrition": "Other", "professional_accounting": "Other",
    "professional_medicine": "Other", "virology": "Other",
}

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. "
    "Answer the following multiple-choice question by selecting the correct option."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
    )
    parser.add_argument("--max_subjects", type=int, default=None,
                        help="Only evaluate first N subjects (for quick test)")
    parser.add_argument("--output_file", type=str, default="./output/base_eval_results.json")
    parser.add_argument("--no_flash_attn", action="store_true")
    return parser.parse_args()


def extract_answer(text: str) -> str:
    text = text.strip()
    pattern = r"(?:The answer is|the answer is|Answer:?|answer:?)\s*([A-D])"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    for char in text:
        if char in "ABCD":
            return char
    return ""


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Qwen3-4B-Instruct-2507 Base Model MMLU Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name_or_path}")

    # ---- Load model ----
    logger.info("Loading model...")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if not args.no_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, **model_kwargs,
        )
    except Exception:
        logger.warning("Flash Attention 2 not available, falling back to default.")
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, **model_kwargs,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    logger.info("Model loaded.")

    # ---- Load MMLU test set ----
    logger.info("Loading MMLU test set...")
    test_ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    logger.info(f"Total test examples: {len(test_ds)}")

    subjects = sorted(set(test_ds["subject"])) if "subject" in test_ds.column_names else ["all"]
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]
        logger.info(f"Limiting to {len(subjects)} subjects: {subjects}")

    # ---- Evaluate ----
    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)
    total_correct = 0
    total_count = 0

    for example in tqdm(test_ds, desc="Evaluating"):
        subject = example.get("subject", "all")
        if args.max_subjects and subject not in subjects:
            continue

        question = example["question"].strip()
        choices = example["choices"]
        answer_idx = example["answer"]
        correct = CHOICE_LABELS[answer_idx] if isinstance(answer_idx, int) else answer_idx

        choices_text = "\n".join(
            f"{CHOICE_LABELS[i]}. {choices[i]}" for i in range(len(choices))
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{question}\n\n{choices_text}"},
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
        predicted = extract_answer(response)

        is_correct = predicted == correct
        subject_correct[subject] += int(is_correct)
        subject_total[subject] += 1
        total_correct += int(is_correct)
        total_count += 1

    # ---- Print results ----
    overall_acc = total_correct / total_count * 100 if total_count else 0

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS: Qwen3-4B-Instruct-2507 (Base, No Fine-Tuning)")
    logger.info("=" * 70)
    logger.info(f"\nOverall Accuracy: {overall_acc:.2f}% ({total_correct}/{total_count})")

    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    subject_results = {}

    for subj in sorted(subject_total.keys()):
        acc = subject_correct[subj] / subject_total[subj] * 100
        subject_results[subj] = {
            "correct": subject_correct[subj],
            "total": subject_total[subj],
            "accuracy": round(acc, 2),
        }
        cat = SUBJECT_TO_CATEGORY.get(subj, "Other")
        category_correct[cat] += subject_correct[subj]
        category_total[cat] += subject_total[subj]

    category_results = {}
    logger.info("\n--- By Category ---")
    for cat in sorted(category_total.keys()):
        acc = category_correct[cat] / category_total[cat] * 100
        category_results[cat] = {
            "correct": category_correct[cat],
            "total": category_total[cat],
            "accuracy": round(acc, 2),
        }
        logger.info(f"  {cat:20s}: {acc:6.2f}% ({category_correct[cat]:4d}/{category_total[cat]:4d})")

    logger.info("\n--- By Subject ---")
    for subj in sorted(subject_results.keys()):
        s = subject_results[subj]
        logger.info(f"  {subj:45s}: {s['accuracy']:6.2f}% ({s['correct']:3d}/{s['total']:3d})")

    logger.info("=" * 70)

    # ---- Save results ----
    results = {
        "model": args.model_name_or_path,
        "overall": {
            "correct": total_correct,
            "total": total_count,
            "accuracy": round(overall_acc, 2),
        },
        "by_category": category_results,
        "by_subject": subject_results,
    }

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
