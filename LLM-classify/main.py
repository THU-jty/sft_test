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
from topic_extractor import extract_topics, load_model
from file_classifier import classify_files, load_embedding_model
from model_manager import ALL_MODELS, get_downloaded_models

# ============================================================
# 在这里修改 System Prompt，控制 LLM 如何提取分类主题
# ============================================================
SYSTEM_PROMPT = """你是一个文件分类专家。你的任务是根据给定的文件名列表，分析这些文件可能涉及的内容领域，
然后提取出最相关的 10 个主题分类。

要求：
1. 主题应该具有区分度，不要太宽泛也不要太具体
2. 主题应该能覆盖大部分文件
3. 每个主题用简短的中文词语表示（2-10个字）
4. 严格输出 JSON 格式，格式为：{"topics": ["主题1", "主题2", ...]}
5. 只输出 JSON，不要输出任何其他内容
6. 提取的内容中,可以包含一些专有领域的关键词信息，但不要违背之前的要求"""
# ============================================================


def print_results(results: dict, output_file: str | None = None):
    """格式化输出分类结果（只显示分类名称和文件数量）。"""
    classification = results["classification"]
    details = results["details"]
    sorted_topics = sorted(classification.items(), key=lambda x: len(x[1]), reverse=True)
    total = len(details)

    lines = []
    lines.append("=" * 60)
    lines.append("分类结果")
    lines.append("=" * 60)

    for topic, files in sorted_topics:
        count = len(files)
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        lines.append(f"  【{topic}】{count} 个文件 ({pct:.1f}%)  {bar}")

    lines.append("-" * 60)
    lines.append(f"  总计: {total} 个文件，{len(classification)} 个主题")
    lines.append("=" * 60)

    output = "\n".join(lines)
    print(output)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        summary = {
            topic: {"count": len(files), "percentage": f"{len(files) / total * 100:.1f}%"}
            for topic, files in sorted_topics
        }
        ordered = {
            "raw_response": results.get("raw_response"),
            "topics": results.get("topics", []),
            "total_files": total,
            "total_topics": len(classification),
            "summary": summary,
            "classification": results["classification"],
            "details": results["details"],
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(ordered, f, ensure_ascii=False, indent=2)
        print(f"\n[结果已保存到 {output_file}]")


def check_available_models():
    """检查已下载的模型，如果没有可用模型则给出提示。"""
    llm_models = get_downloaded_models("llm")
    emb_models = get_downloaded_models("embedding")

    if not llm_models and not emb_models:
        print("[错误] 没有找到任何已下载的模型。")
        print("  请先运行以下命令下载模型:")
        print("  python model_manager.py download qwen3-4b")
        print("  python model_manager.py download bge-base-zh")
        print("\n  查看所有可用模型: python model_manager.py list")
        sys.exit(1)

    return llm_models, emb_models


def main():
    available_llm, available_emb = check_available_models()
    llm_keys = list(available_llm.keys()) if available_llm else []
    emb_keys = list(available_emb.keys()) if available_emb else []

    parser = argparse.ArgumentParser(
        description="基于 LLM 的文件智能分类工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
已下载的模型:
  LLM:       {', '.join(llm_keys) if llm_keys else '(无，需先下载)'}
  Embedding: {', '.join(emb_keys) if emb_keys else '(无，需先下载)'}

使用示例:
  python main.py /path/to/your/files
  python main.py /path/to/your/files --llm qwen3-8b --embedding qwen3-0.6b
  python main.py /path/to/your/files --topics "编程,文档,图片" --output result.json

模型管理:
  python model_manager.py list              # 查看所有模型
  python model_manager.py download qwen3-4b # 下载指定模型
        """,
    )
    parser.add_argument("directory", help="要扫描的文件目录路径")

    if llm_keys:
        parser.add_argument(
            "--llm",
            choices=llm_keys,
            default=llm_keys[0],
            help=f"用于提取主题的 LLM 模型 (默认: {llm_keys[0]})",
        )
    parser.add_argument(
        "--embedding",
        choices=emb_keys if emb_keys else None,
        default=emb_keys[0] if emb_keys else None,
        help=f"用于文件分类的 Embedding 模型 (默认: {emb_keys[0] if emb_keys else 'N/A'})",
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
    raw_response = None
    if args.topics:
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]
        print(f"\n[Step 2] 使用手动指定的主题: {topics}")
    else:
        if not llm_keys:
            print("[错误] 没有已下载的 LLM 模型，无法自动提取主题。")
            print("  请用 --topics 手动指定主题，或先下载 LLM 模型:")
            print("  python model_manager.py download qwen3-4b")
            sys.exit(1)

        print(f"\n[Step 2] 使用 {args.llm} 提取主题 ...")
        llm_model, llm_tokenizer = load_model(args.llm)
        topics, raw_response = extract_topics(
            filenames,
            model_key=args.llm,
            model=llm_model,
            tokenizer=llm_tokenizer,
            system_prompt=SYSTEM_PROMPT,
        )
        del llm_model, llm_tokenizer
        import torch
        torch.cuda.empty_cache()

    print(f"  提取到的主题: {topics}")

    if not topics:
        print("[错误] 未能提取到任何主题，退出。")
        sys.exit(1)

    # Step 3: 文件分类
    if not emb_keys:
        print("[错误] 没有已下载的 Embedding 模型，无法进行分类。")
        print("  请先下载 Embedding 模型:")
        print("  python model_manager.py download bge-base-zh")
        sys.exit(1)

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
    results["topics"] = topics
    results["raw_response"] = raw_response
    print(f"\n[Step 4] 输出分类结果\n")
    print_results(results, output_file=args.output)


if __name__ == "__main__":
    main()
