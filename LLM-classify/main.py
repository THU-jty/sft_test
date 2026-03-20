"""文件智能分类主流程。

工作流：
1. 扫描指定目录，收集所有文件名
2. 使用 Qwen3-4B / Qwen3-8B 从文件名中提取 10 个主题
3. 使用 Embedding 模型 (Qwen3-0.6B / BGE-base-zh-v1.5) 将每个文件分类到对应主题
4. 输出分类结果
"""

import argparse
import json
import sys
from pathlib import Path

from file_scanner import scan_directory
from topic_extractor import extract_topics, load_model, SUPPORTED_MODELS
from file_classifier import classify_files, load_embedding_model, SUPPORTED_EMBEDDING_MODELS


def print_results(results: dict, output_file: str | None = None):
    """格式化输出分类结果。"""
    classification = results["classification"]
    details = results["details"]
    sorted_topics = sorted(classification.items(), key=lambda x: len(x[1]), reverse=True)

    lines = []
    lines.append("=" * 60)
    lines.append("分类概览")
    lines.append("=" * 60)

    for topic, files in sorted_topics:
        count = len(files)
        bar = "█" * count
        lines.append(f"  【{topic}】{count} 个文件  {bar}")

    lines.append(f"\n  总计: {len(details)} 个文件，{len(classification)} 个主题")

    lines.append("\n" + "=" * 60)
    lines.append("分类详情")
    lines.append("=" * 60)

    for topic, files in sorted_topics:
        lines.append(f"\n📂 【{topic}】({len(files)} 个文件)")
        lines.append("-" * 40)
        if files:
            for f in files:
                score = next(
                    (d["score"] for d in details if d["filename"] == f), 0
                )
                lines.append(f"  {f}  (相似度: {score:.4f})")
        else:
            lines.append("  （无文件归入此分类）")

    lines.append("\n" + "=" * 60)

    output = "\n".join(lines)
    print(output)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[结果已保存到 {output_file}]")


def main():
    parser = argparse.ArgumentParser(
        description="基于 LLM 的文件智能分类工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认模型 (Qwen3-4B + BGE-base-zh-v1.5)
  python main.py /path/to/your/files

  # 指定推理模型为 Qwen3-8B
  python main.py /path/to/your/files --llm qwen3-8b

  # 指定 Embedding 模型为 Qwen3-0.6B
  python main.py /path/to/your/files --embedding qwen3-0.6b

  # 不递归扫描子目录，结果保存到文件
  python main.py /path/to/your/files --no-recursive --output result.json

  # 手动指定主题（跳过 LLM 推理步骤）
  python main.py /path/to/your/files --topics "编程,设计,文档,音乐,视频,图片,数据,配置,日志,测试"
        """,
    )
    parser.add_argument("directory", help="要扫描的文件目录路径")
    parser.add_argument(
        "--llm",
        choices=list(SUPPORTED_MODELS.keys()),
        default="qwen3-4b",
        help="用于提取主题的 LLM 模型 (默认: qwen3-4b)",
    )
    parser.add_argument(
        "--embedding",
        choices=list(SUPPORTED_EMBEDDING_MODELS.keys()),
        default="bge-base-zh",
        help="用于文件分类的 Embedding 模型 (默认: bge-base-zh)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="不递归扫描子目录",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="将结果保存为 JSON 文件",
    )
    parser.add_argument(
        "--topics",
        type=str,
        default=None,
        help="手动指定主题（逗号分隔），跳过 LLM 推理步骤",
    )

    args = parser.parse_args()

    # Step 1: 扫描目录
    print(f"\n[Step 1] 扫描目录: {args.directory}")
    filenames = scan_directory(args.directory, recursive=not args.no_recursive)
    print(f"  找到 {len(filenames)} 个文件")

    if not filenames:
        print("[错误] 没有找到任何文件，退出。")
        sys.exit(1)

    # Step 2: 提取主题
    if args.topics:
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]
        print(f"\n[Step 2] 使用手动指定的主题: {topics}")
    else:
        print(f"\n[Step 2] 使用 {args.llm} 提取主题 ...")
        llm_model, llm_tokenizer = load_model(args.llm)
        topics = extract_topics(
            filenames,
            model_key=args.llm,
            model=llm_model,
            tokenizer=llm_tokenizer,
        )
        # 释放 LLM 显存
        del llm_model, llm_tokenizer
        import torch
        torch.cuda.empty_cache()

    print(f"  提取到的主题: {topics}")

    if not topics:
        print("[错误] 未能提取到任何主题，退出。")
        sys.exit(1)

    # Step 3: 文件分类
    print(f"\n[Step 3] 使用 {args.embedding} 进行文件分类 ...")
    emb_model, emb_tokenizer = load_embedding_model(args.embedding)
    results = classify_files(
        filenames,
        topics,
        embedding_model_key=args.embedding,
        model=emb_model,
        tokenizer=emb_tokenizer,
    )

    # Step 4: 输出结果
    print(f"\n[Step 4] 输出分类结果\n")
    print_results(results, output_file=args.output)


if __name__ == "__main__":
    main()
