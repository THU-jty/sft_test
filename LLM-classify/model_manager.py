"""模型下载与管理。

独立运行此脚本来下载/查看模型，main.py 只从本地已下载的模型中加载。

用法:
  python model_manager.py list                    # 列出所有支持的模型及下载状态
  python model_manager.py download qwen3-4b       # 下载指定模型
  python model_manager.py download --all          # 下载所有模型
  python model_manager.py download qwen3-4b --mirror  # 使用国内镜像下载
"""

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download, scan_cache_dir

MIRROR_ENDPOINTS = [
    "https://hf-mirror.com",
    "https://huggingface.sukaka.top",
]
DEFAULT_ENDPOINT = "https://huggingface.co"

MAX_RETRIES = 3
RETRY_DELAY_SEC = 5

ALL_MODELS = {
    "qwen3-4b": {"repo": "Qwen/Qwen3-4B", "type": "llm"},
    "qwen3-8b": {"repo": "Qwen/Qwen3-8B", "type": "llm"},
    "qwen3-0.6b": {"repo": "Qwen/Qwen3-0.6B", "type": "embedding"},
    "bge-base-zh": {"repo": "BAAI/bge-base-zh-v1.5", "type": "embedding"},
}


def get_cached_repos() -> set[str]:
    """扫描 HuggingFace 缓存，返回已下载的 repo_id 集合。"""
    try:
        cache_info = scan_cache_dir()
        return {repo.repo_id for repo in cache_info.repos}
    except Exception:
        return set()


def is_model_downloaded(model_key: str) -> bool:
    """检查指定模型是否已下载到本地。"""
    if model_key not in ALL_MODELS:
        return False
    repo_id = ALL_MODELS[model_key]["repo"]
    return repo_id in get_cached_repos()


def get_downloaded_models(model_type: str | None = None) -> dict[str, dict]:
    """获取已下载的模型列表。

    Args:
        model_type: 过滤类型，"llm" 或 "embedding"，None 表示全部
    """
    cached = get_cached_repos()
    result = {}
    for key, info in ALL_MODELS.items():
        if model_type and info["type"] != model_type:
            continue
        if info["repo"] in cached:
            result[key] = info
    return result


def list_models():
    """列出所有支持的模型及其下载状态。"""
    cached = get_cached_repos()

    print("=" * 60)
    print("模型列表")
    print("=" * 60)

    print("\n推理模型 (LLM):")
    print("-" * 40)
    for key, info in ALL_MODELS.items():
        if info["type"] != "llm":
            continue
        status = "✅ 已下载" if info["repo"] in cached else "❌ 未下载"
        print(f"  {key:<15} {info['repo']:<30} {status}")

    print("\nEmbedding 模型:")
    print("-" * 40)
    for key, info in ALL_MODELS.items():
        if info["type"] != "embedding":
            continue
        status = "✅ 已下载" if info["repo"] in cached else "❌ 未下载"
        print(f"  {key:<15} {info['repo']:<30} {status}")

    print()


def _set_mirror(endpoint: str):
    """设置 HuggingFace 镜像端点（通过环境变量）。"""
    os.environ["HF_ENDPOINT"] = endpoint


def _download_once(repo_id: str) -> bool:
    """尝试下载一次，成功返回 True，失败返回 False 并打印错误。"""
    try:
        snapshot_download(repo_id, resume_download=True)
        return True
    except Exception as e:
        print(f"  下载出错: {e}")
        return False


def download_model(model_key: str, use_mirror: bool = False):
    """下载指定模型到本地 HuggingFace 缓存，支持重试和镜像源。"""
    if model_key not in ALL_MODELS:
        print(f"[错误] 未知模型: {model_key}")
        print(f"  可选: {', '.join(ALL_MODELS.keys())}")
        sys.exit(1)

    info = ALL_MODELS[model_key]
    repo_id = info["repo"]

    if is_model_downloaded(model_key):
        print(f"[跳过] {model_key} ({repo_id}) 已存在于本地缓存")
        return

    endpoints = MIRROR_ENDPOINTS + [DEFAULT_ENDPOINT] if use_mirror else [DEFAULT_ENDPOINT]

    for endpoint in endpoints:
        _set_mirror(endpoint)
        print(f"[下载] {model_key} ({repo_id}) | 源: {endpoint}")

        for attempt in range(1, MAX_RETRIES + 1):
            print(f"  第 {attempt}/{MAX_RETRIES} 次尝试 ...")
            if _download_once(repo_id):
                print(f"[完成] {model_key} 下载成功")
                return

            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY_SEC * attempt
                print(f"  等待 {wait}s 后重试 ...")
                time.sleep(wait)

        print(f"  源 {endpoint} 全部重试失败，尝试下一个源 ...\n")

    print(f"[错误] {model_key} 所有下载源均失败。")
    print("  建议:")
    print("  1. 检查网络连接")
    print("  2. 使用 --mirror 参数尝试国内镜像: python model_manager.py download qwen3-4b --mirror")
    print("  3. 手动设置代理: set HTTPS_PROXY=http://127.0.0.1:7890")
    sys.exit(1)


def download_all(use_mirror: bool = False):
    """下载所有模型。"""
    for key in ALL_MODELS:
        download_model(key, use_mirror=use_mirror)


def main():
    parser = argparse.ArgumentParser(description="模型下载与管理工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    subparsers.add_parser("list", help="列出所有模型及下载状态")

    dl_parser = subparsers.add_parser("download", help="下载模型")
    dl_parser.add_argument(
        "model",
        nargs="?",
        choices=list(ALL_MODELS.keys()),
        help="要下载的模型名称",
    )
    dl_parser.add_argument("--all", action="store_true", help="下载所有模型")
    dl_parser.add_argument(
        "--mirror",
        action="store_true",
        help="使用国内镜像源下载 (hf-mirror.com 等)",
    )

    args = parser.parse_args()

    if args.command == "list":
        list_models()
    elif args.command == "download":
        if args.all:
            download_all(use_mirror=args.mirror)
        elif args.model:
            download_model(args.model, use_mirror=args.mirror)
        else:
            parser.error("请指定模型名称或使用 --all 下载全部")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
