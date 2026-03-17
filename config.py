from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: Optional[str] = "flash_attention_2"


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DataConfig:
    dataset_name: str = "cais/mmlu"
    max_seq_length: int = 512
    train_splits: List[str] = field(default_factory=lambda: [
        "auxiliary_train", "dev",
    ])
    val_split: str = "validation"
    test_split: str = "test"
    num_proc: int = 8
    system_prompt: str = (
        "You are a knowledgeable assistant. "
        "Answer the following multiple-choice question by selecting the correct option."
    )


@dataclass
class TrainingConfig:
    output_dir: str = "./output/qwen3-4b-mmlu-lora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    bf16: bool = True
    logging_steps: int = 50
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 4
    report_to: str = "tensorboard"
    seed: int = 42
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    gradient_checkpointing: bool = True


@dataclass
class EvalConfig:
    per_device_batch_size: int = 16
    output_file: str = "./output/eval_results.json"


MMLU_SUBJECT_CATEGORIES = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics",
        "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history",
        "high_school_us_history", "high_school_world_history",
        "international_law", "jurisprudence", "logical_fallacies",
        "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy",
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management",
        "marketing", "medical_genetics", "miscellaneous",
        "nutrition", "professional_accounting", "professional_medicine",
        "virology",
    ],
}

SUBJECT_TO_CATEGORY = {}
for category, subjects in MMLU_SUBJECT_CATEGORIES.items():
    for subject in subjects:
        SUBJECT_TO_CATEGORY[subject] = category
