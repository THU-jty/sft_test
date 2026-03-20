"""使用本地 GPU 上的 Qwen3 模型，从文件名列表中提取 10 个最相关的主题。"""

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SUPPORTED_MODELS = {
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
}

SYSTEM_PROMPT = """你是一个文件分类专家。你的任务是根据给定的文件名列表，分析这些文件可能涉及的内容领域，
然后提取出最相关的 10 个主题分类。

要求：
1. 主题应该具有区分度，不要太宽泛也不要太具体
2. 主题应该能覆盖大部分文件
3. 每个主题用简短的中文词语表示（2-6个字）
4. 严格输出 JSON 格式，格式为：{"topics": ["主题1", "主题2", ...]}
5. 只输出 JSON，不要输出任何其他内容"""


def build_user_prompt(filenames: list[str]) -> str:
    file_list = "\n".join(f"- {name}" for name in filenames)
    return f"以下是一组文件名，请分析并提取 10 个最相关的主题分类：\n\n{file_list}"


def load_model(model_key: str) -> tuple:
    """加载指定的 Qwen3 模型和 tokenizer。"""
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(
            f"不支持的模型: {model_key}，可选: {list(SUPPORTED_MODELS.keys())}"
        )

    model_name = SUPPORTED_MODELS[model_key]
    print(f"[加载模型] {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[模型就绪] {model_name}")
    return model, tokenizer


def extract_topics(
    filenames: list[str],
    model_key: str = "qwen3-4b",
    model=None,
    tokenizer=None,
    max_new_tokens: int = 512,
) -> list[str]:
    """从文件名列表中提取 10 个主题。

    可以传入已加载的 model/tokenizer 以避免重复加载。
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(filenames)},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"[模型原始输出]\n{response}\n")

    return parse_topics(response)


def parse_topics(response: str) -> list[str]:
    """从模型输出中解析 JSON 格式的主题列表。"""
    json_match = re.search(r'\{.*"topics"\s*:\s*\[.*?\].*?\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get("topics", [])
        except json.JSONDecodeError:
            pass

    # fallback: 尝试从列表格式中提取
    items = re.findall(r'["\u201c]([^"\u201d]+)["\u201d]', response)
    if items:
        return items[:10]

    raise ValueError(f"无法从模型输出中解析主题列表:\n{response}")
