"""使用本地 GPU 上的 Qwen3 模型，从文件名列表中提取 10 个最相关的主题。"""

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_manager import ALL_MODELS, is_model_downloaded, get_model_local_path

DEFAULT_SYSTEM_PROMPT = """你是一个文件分类专家。你的任务是根据给定的文件名列表，分析这些文件可能涉及的内容领域，
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
    """从本地缓存加载指定的 Qwen3 模型和 tokenizer（不会触发下载）。"""
    if model_key not in ALL_MODELS or ALL_MODELS[model_key]["type"] != "llm":
        llm_keys = [k for k, v in ALL_MODELS.items() if v["type"] == "llm"]
        raise ValueError(f"不支持的 LLM 模型: {model_key}，可选: {llm_keys}")

    if not is_model_downloaded(model_key):
        raise RuntimeError(
            f"模型 {model_key} 尚未下载，请先运行: python model_manager.py download {model_key}"
        )

    local_path = str(get_model_local_path(model_key))
    print(f"[加载模型] {model_key} (从 {local_path}) ...")

    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[模型就绪] {model_key}")
    return model, tokenizer


def extract_topics(
    filenames: list[str],
    model_key: str = "qwen3-4b",
    model=None,
    tokenizer=None,
    max_new_tokens: int = 512,
    system_prompt: str | None = None,
) -> tuple[list[str], str]:
    """从文件名列表中提取 10 个主题。

    Args:
        system_prompt: 自定义 System Prompt，为 None 时使用内置默认值

    Returns:
        (topics, raw_response): 解析后的主题列表 和 模型原始输出文本
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_key)

    prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": prompt},
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

    return parse_topics(response), response


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
