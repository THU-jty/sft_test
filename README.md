# Qwen3-4B-Instruct-2507 MMLU LoRA 微调

使用 LoRA 对 Qwen3-4B-Instruct-2507 在 MMLU 数据集上进行监督微调 (SFT)，提升模型在多领域知识问答上的表现。

## 环境要求

- Python 3.10+
- CUDA 12.1+
- GPU 显存 >= 16GB（推荐 24GB+）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练

```bash
python train.py
```

常用参数覆盖：

```bash
# 自定义训练参数
python train.py \
    --output_dir ./output/my_run \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --lora_r 32 \
    --lora_alpha 64

# 从断点恢复训练
python train.py --resume_from_checkpoint ./output/qwen3-4b-mmlu-lora/checkpoint-5000

# 不使用 Flash Attention（GPU 不支持时）
python train.py --no_flash_attn
```

### 3. 评估

```bash
# 评估微调后的模型
python evaluate.py --checkpoint_dir ./output/qwen3-4b-mmlu-lora/final_checkpoint

# 与基座模型对比
python evaluate.py \
    --checkpoint_dir ./output/qwen3-4b-mmlu-lora/final_checkpoint \
    --compare_base

# 快速测试（只评估前 5 个 subject）
python evaluate.py \
    --checkpoint_dir ./output/qwen3-4b-mmlu-lora/final_checkpoint \
    --max_subjects 5
```

### 4. 合并模型

将 LoRA 适配器合并到基座模型，生成可独立部署的完整模型：

```bash
python merge_model.py \
    --checkpoint_dir ./output/qwen3-4b-mmlu-lora/final_checkpoint \
    --output_dir ./output/qwen3-4b-mmlu-merged
```

## 项目结构

```
.
├── config.py          # 超参数配置（模型、LoRA、训练、评估）
├── data_utils.py      # MMLU 数据加载与格式化
├── train.py           # 主训练脚本
├── evaluate.py        # MMLU 评估脚本
├── merge_model.py     # LoRA 合并脚本
├── requirements.txt   # Python 依赖
└── README.md          # 本文件
```

## 技术细节

### 微调方法

- **LoRA** (Low-Rank Adaptation): 只训练约 0.5% 的参数，显存占用低，防止灾难性遗忘
- 目标模块: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- rank=16, alpha=32

### 数据格式

MMLU 多选题被转换为 Qwen3 Instruct 对话格式：

```
System: You are a knowledgeable assistant...
User:   What is the embryological origin of the hyoid bone?
        A. The first pharyngeal arch
        B. The first and second pharyngeal arches
        C. The second pharyngeal arch
        D. The second and third pharyngeal arches
Assistant: The answer is D.
```

### 训练数据分配

| 分割 | 样本数 | 用途 |
|------|--------|------|
| auxiliary_train + dev | ~100,127 | 训练 |
| validation | 1,531 | 验证 |
| test | 14,042 | 最终评估 |

### 默认超参数

| 参数 | 值 |
|------|-----|
| Learning Rate | 2e-4 |
| Epochs | 3 |
| Batch Size (effective) | 32 (8 × 4) |
| Max Sequence Length | 512 |
| LR Scheduler | Cosine |
| Warmup | 5% |
| Precision | bf16 |

## 监控训练

训练过程中会将日志写入 TensorBoard：

```bash
tensorboard --logdir ./output/qwen3-4b-mmlu-lora
```

## 常见问题

**Q: 显存不足怎么办？**
- 减小 `--per_device_train_batch_size`（如 4 或 2）
- 增大 `--gradient_accumulation_steps` 以保持有效批大小
- 减小 `--max_seq_length`（如 256）
- 减小 `--lora_r`（如 8）

**Q: 如何使用不同的基座模型？**
```bash
python train.py --model_name_or_path Qwen/Qwen3-8B-Instruct-2507
```

**Q: 训练中断了怎么办？**
```bash
python train.py --resume_from_checkpoint ./output/qwen3-4b-mmlu-lora/checkpoint-XXXX
```
