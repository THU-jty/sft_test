import logging
from typing import Dict, List, Optional

from datasets import DatasetDict, concatenate_datasets, load_dataset

from config import DataConfig

logger = logging.getLogger(__name__)

CHOICE_LABELS = ["A", "B", "C", "D"]


def format_mmlu_example(example: Dict, system_prompt: str) -> Dict:
    """Convert a single MMLU example to Qwen3 Instruct chat format."""
    question = example["question"].strip()
    choices = example["choices"]
    answer_idx = example["answer"]
    if isinstance(answer_idx, str):
        answer_idx = CHOICE_LABELS.index(answer_idx)

    choices_text = "\n".join(
        f"{CHOICE_LABELS[i]}. {choices[i]}" for i in range(len(choices))
    )
    user_content = f"{question}\n\n{choices_text}"
    assistant_content = f"The answer is {CHOICE_LABELS[answer_idx]}."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    return {"messages": messages}


def load_mmlu_subjects() -> List[str]:
    """Load all available MMLU subject names."""
    builder = load_dataset("cais/mmlu", "all", trust_remote_code=True)
    if hasattr(builder, "column_names") and "subject" in builder.column_names.get("test", []):
        subjects = list(set(builder["test"]["subject"]))
        subjects.sort()
        return subjects
    return []


def load_and_prepare_data(
    data_config: DataConfig,
    tokenizer=None,
) -> DatasetDict:
    """Load MMLU dataset and convert to chat format.

    Loads auxiliary_train + dev for training, val for validation.
    If tokenizer is provided, applies chat template and tokenization.
    """
    logger.info("Loading MMLU dataset from HuggingFace...")

    train_datasets = []
    for split in data_config.train_splits:
        ds_split = _resolve_split_name(split)
        logger.info(f"Loading split: {split} (resolved to: {ds_split})")
        ds = load_dataset(
            data_config.dataset_name, "all",
            split=ds_split, trust_remote_code=True,
        )
        train_datasets.append(ds)

    train_ds = concatenate_datasets(train_datasets)
    logger.info(f"Training set size: {len(train_ds)}")

    val_split = _resolve_split_name(data_config.val_split)
    val_ds = load_dataset(
        data_config.dataset_name, "all",
        split=val_split, trust_remote_code=True,
    )
    logger.info(f"Validation set size: {len(val_ds)}")

    train_ds = train_ds.shuffle(seed=42)

    logger.info("Formatting examples to chat format...")
    train_ds = train_ds.map(
        lambda ex: format_mmlu_example(ex, data_config.system_prompt),
        num_proc=data_config.num_proc,
        desc="Formatting train",
    )
    val_ds = val_ds.map(
        lambda ex: format_mmlu_example(ex, data_config.system_prompt),
        num_proc=data_config.num_proc,
        desc="Formatting val",
    )

    if tokenizer is not None:
        train_ds = train_ds.map(
            lambda ex: _apply_chat_template(ex, tokenizer, data_config.max_seq_length),
            num_proc=data_config.num_proc,
            desc="Tokenizing train",
        )
        val_ds = val_ds.map(
            lambda ex: _apply_chat_template(ex, tokenizer, data_config.max_seq_length),
            num_proc=data_config.num_proc,
            desc="Tokenizing val",
        )

    return DatasetDict({"train": train_ds, "validation": val_ds})


def load_test_data(data_config: DataConfig) -> "Dataset":
    """Load the MMLU test split for evaluation."""
    test_split = _resolve_split_name(data_config.test_split)
    test_ds = load_dataset(
        data_config.dataset_name, "all",
        split=test_split, trust_remote_code=True,
    )
    logger.info(f"Test set size: {len(test_ds)}")
    return test_ds


def _resolve_split_name(split: str) -> str:
    """Resolve user-friendly split names to actual HuggingFace split names."""
    mapping = {
        "auxiliary_train": "auxiliary_train",
        "dev": "dev",
        "validation": "validation",
        "val": "validation",
        "test": "test",
    }
    return mapping.get(split, split)


def _apply_chat_template(
    example: Dict,
    tokenizer,
    max_seq_length: int,
) -> Dict:
    """Apply Qwen3 chat template and tokenize."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenized = tokenizer(
        text,
        max_length=max_seq_length,
        truncation=True,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized
