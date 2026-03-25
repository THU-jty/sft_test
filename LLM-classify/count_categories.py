import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="统计每个分类的文档数目")
    parser.add_argument("--input", "-i", default="data/category_articles_with_titles_v12_wikipages.json",
                        help="输入 JSON 文件路径")
    parser.add_argument("--output", "-o", default="data/category_doc_counts.txt",
                        help="输出结果文件路径")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = [(cat, len(docs)) for cat, docs in data.items()]
    stats.sort(key=lambda x: (x[1] != 0, x[1]))

    total_cats = len(stats)
    total_docs = sum(c for _, c in stats)
    empty_cats = sum(1 for _, c in stats if c == 0)

    lines = []
    lines.append(f"分类总数: {total_cats}")
    lines.append(f"文档总数: {total_docs}")
    lines.append(f"空分类数: {empty_cats}")
    lines.append("=" * 80)
    lines.append(f"{'分类':<60} {'文档数':>8}")
    lines.append("-" * 80)
    for cat, count in stats:
        lines.append(f"{cat:<60} {count:>8}")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"已保存到 {args.output}（共 {total_cats} 个分类，{total_docs} 篇文档）")

if __name__ == "__main__":
    main()
