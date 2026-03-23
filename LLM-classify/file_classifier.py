"""使用 Embedding 模型计算文件名与主题的相似度，完成分类。

支持的 Embedding 模型：
- Qwen3-0.6B: 使用 last_hidden_state 的 mean pooling 作为 embedding
- BGE-base-zh-v1.5: 专用的中文 embedding 模型
"""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

from model_manager import ALL_MODELS, is_model_downloaded, get_model_local_path


def load_embedding_model(model_key: str) -> tuple:
    """从本地缓存加载 embedding 模型和 tokenizer（不会触发下载）。"""
    if model_key not in ALL_MODELS or ALL_MODELS[model_key]["type"] != "embedding":
        emb_keys = [k for k, v in ALL_MODELS.items() if v["type"] == "embedding"]
        raise ValueError(f"不支持的 Embedding 模型: {model_key}，可选: {emb_keys}")

    if not is_model_downloaded(model_key):
        raise RuntimeError(
            f"模型 {model_key} 尚未下载，请先运行: python model_manager.py download {model_key}"
        )

    local_path = str(get_model_local_path(model_key))
    print(f"[加载 Embedding 模型] {model_key} (从 {local_path}) ...")

    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        local_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[Embedding 模型就绪] {model_key}")
    return model, tokenizer


def get_embeddings(
    texts: list[str],
    model,
    tokenizer,
    model_key: str,
    batch_size: int = 32,
) -> np.ndarray:
    """获取一组文本的 embedding 向量。"""
    # BGE 模型推荐在查询前加 "为这个句子生成表示以用于检索中文相关句子：" 前缀
    # 但这里是对称任务（文件名 vs 主题），不需要加前缀
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        if model_key == "bge-base-zh":
            emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        else:
            # Qwen3-0.6B: mean pooling
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            emb = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)

        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)


def classify_files(
    filenames: list[str],
    topics: list[str],
    embedding_model_key: str = "bge-base-zh",
    model=None,
    tokenizer=None,
) -> dict[str, dict]:
    """将每个文件名分配到最相似的主题下。

    Returns:
        dict: {
            "classification": {主题: [文件名列表]},
            "details": [{filename, topic, score}, ...]
        }
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_embedding_model(embedding_model_key)

    print(f"[计算 Embedding] 文件数: {len(filenames)}, 主题数: {len(topics)}")

    file_embeddings = get_embeddings(filenames, model, tokenizer, embedding_model_key)
    topic_embeddings = get_embeddings(topics, model, tokenizer, embedding_model_key)

    # 余弦相似度矩阵: (n_files, n_topics)
    similarity_matrix = file_embeddings @ topic_embeddings.T
    best_topic_indices = np.argmax(similarity_matrix, axis=1)

    classification = {topic: [] for topic in topics}
    details = []

    for i, filename in enumerate(filenames):
        best_idx = best_topic_indices[i]
        best_topic = topics[best_idx]
        best_score = float(similarity_matrix[i, best_idx])

        classification[best_topic].append(filename)
        details.append({
            "filename": filename,
            "topic": best_topic,
            "score": round(best_score, 4),
        })

    # 按主题排序 details
    details.sort(key=lambda x: (x["topic"], -x["score"]))

    return {"classification": classification, "details": details}
