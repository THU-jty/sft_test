"""模型下载与管理。

支持两种下载源：
1. modelscope（魔搭社区，国内首选，速度快）
2. huggingface（需科学上网或镜像）

用法:
  python model_manager.py list                          # 列出所有模型及下载状态
  python model_manager.py download qwen3-4b             # 从 modelscope 下载（默认）
  python model_manager.py download --all                # 下载所有模型
  python model_manager.py download qwen3-4b --source hf # 从 huggingface 下载
"""

import argparse
import os
import sys
import time
from pathlib import Path

MAX_RETRIES = 3
RETRY_DELAY_SEC = 5

# 统一的本地存储目录，所有模型都下载到这里
DEFAULT_MODEL_DIR = Path.home() / "autodl-tmp" / "models"

ALL_MODELS = {
    "qwen3-4b": {
        "hf_repo": "Qwen/Qwen3-4B",
        "ms_repo": "Qwen/Qwen3-4B",
        "type": "llm",
    },
    "qwen3-8b": {
        "hf_repo": "Qwen/Qwen3-8B",
        "ms_repo": "Qwen/Qwen3-8B",
        "type": "llm",
    },
    "qwen3.5-4b": {
        "hf_repo": "Qwen/Qwen3.5-4B",
        "ms_repo": "Qwen/Qwen3.5-4B",
        "type": "llm",
    },
    "qwen3.5-9b": {
        "hf_repo": "Qwen/Qwen3.5-9B",
        "ms_repo": "Qwen/Qwen3.5-9B",
        "type": "llm",
    },
    "qwen3-0.6b-llm": {
        "hf_repo": "Qwen/Qwen3-0.6B",
        "ms_repo": "Qwen/Qwen3-0.6B",
        "type": "llm",
    },
    "qwen3-0.6b": {
        "hf_repo": "Qwen/Qwen3-0.6B",
        "ms_repo": "Qwen/Qwen3-0.6B",
        "type": "embedding",
    },
    "bge-base-zh": {
        "hf_repo": "BAAI/bge-base-zh-v1.5",
        "ms_repo": "BAAI/bge-base-zh-v1.5",
        "type": "embedding",
    },
}


def get_model_dir() -> Path:
    """获取模型存储根目录，优先使用环境变量 LLM_CLASSIFY_MODEL_DIR。"""
    custom = os.environ.get("LLM_CLASSIFY_MODEL_DIR")
    if custom:
        return Path(custom)
    return DEFAULT_MODEL_DIR


def get_model_local_path(model_key: str) -> Path:
    """获取某个模型的本地存储路径。"""
    return get_model_dir() / model_key


def is_model_downloaded(model_key: str) -> bool:
    """检查模型是否已下载到本地（通过检测目录下是否有模型文件）。"""
    if model_key not in ALL_MODELS:
        return False
    local_path = get_model_local_path(model_key)
    if not local_path.exists():
        return False
    model_files = list(local_path.glob("*.safetensors")) + list(local_path.glob("*.bin"))
    config_exists = (local_path / "config.json").exists()
    return config_exists and len(model_files) > 0


def get_downloaded_models(model_type: str | None = None) -> dict[str, dict]:
    """获取已下载的模型列表。

    Args:
        model_type: 过滤类型，"llm" 或 "embedding"，None 表示全部
    """
    result = {}
    for key, info in ALL_MODELS.items():
        if model_type and info["type"] != model_type:
            continue
        if is_model_downloaded(key):
            result[key] = info
    return result


def list_models():
    """列出所有支持的模型及其下载状态。"""
    model_dir = get_model_dir()
    print("=" * 65)
    print(f"模型列表  (存储目录: {model_dir})")
    print("=" * 65)

    print("\n推理模型 (LLM):")
    print("-" * 50)
    for key, info in ALL_MODELS.items():
        if info["type"] != "llm":
            continue
        status = "✅ 已下载" if is_model_downloaded(key) else "❌ 未下载"
        print(f"  {key:<15} {info['ms_repo']:<30} {status}")

    print("\nEmbedding 模型:")
    print("-" * 50)
    for key, info in ALL_MODELS.items():
        if info["type"] != "embedding":
            continue
        status = "✅ 已下载" if is_model_downloaded(key) else "❌ 未下载"
        print(f"  {key:<15} {info['ms_repo']:<30} {status}")

    print()


def _download_from_modelscope(repo_id: str, local_dir: Path) -> bool:
    """从 modelscope 下载模型。"""
    try:
        from modelscope import snapshot_download as ms_download
        ms_download(repo_id, local_dir=str(local_dir))
        return True
    except ImportError:
        print("  [提示] modelscope 未安装，正在自动安装 ...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope", "-q"])
        from modelscope import snapshot_download as ms_download
        ms_download(repo_id, local_dir=str(local_dir))
        return True
    except Exception as e:
        print(f"  modelscope 下载出错: {e}")
        return False


def _download_from_huggingface(repo_id: str, local_dir: Path, use_mirror: bool = True) -> bool:
    """从 huggingface 下载模型（支持镜像）。"""
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id, local_dir=str(local_dir))
        return True
    except Exception as e:
        print(f"  huggingface 下载出错: {e}")
        return False


def download_model(model_key: str, source: str = "modelscope"):
    """下载指定模型到本地目录，带重试机制。"""
    if model_key not in ALL_MODELS:
        print(f"[错误] 未知模型: {model_key}")
        print(f"  可选: {', '.join(ALL_MODELS.keys())}")
        sys.exit(1)

    if is_model_downloaded(model_key):
        local_path = get_model_local_path(model_key)
        print(f"[跳过] {model_key} 已存在于 {local_path}")
        return

    info = ALL_MODELS[model_key]
    local_dir = get_model_local_path(model_key)
    local_dir.mkdir(parents=True, exist_ok=True)

    if source == "modelscope":
        repo_id = info["ms_repo"]
        download_fn = lambda: _download_from_modelscope(repo_id, local_dir)
    else:
        repo_id = info["hf_repo"]
        download_fn = lambda: _download_from_huggingface(repo_id, local_dir)

    print(f"[下载] {model_key} ({repo_id}) | 源: {source}")
    print(f"  目标目录: {local_dir}")

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  第 {attempt}/{MAX_RETRIES} 次尝试 ...")
        if download_fn():
            print(f"[完成] {model_key} 下载成功 -> {local_dir}")
            return

        if attempt < MAX_RETRIES:
            wait = RETRY_DELAY_SEC * attempt
            print(f"  等待 {wait}s 后重试 ...")
            time.sleep(wait)

    # 主源全部失败，如果用的是 hf 就没有 fallback 了
    if source == "modelscope":
        print(f"\n  modelscope 下载失败，自动切换到 hf-mirror 尝试 ...")
        repo_id = info["hf_repo"]
        for attempt in range(1, MAX_RETRIES + 1):
            print(f"  [hf-mirror] 第 {attempt}/{MAX_RETRIES} 次尝试 ...")
            if _download_from_huggingface(repo_id, local_dir, use_mirror=True):
                print(f"[完成] {model_key} 下载成功 (via hf-mirror) -> {local_dir}")
                return
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY_SEC * attempt
                print(f"  等待 {wait}s 后重试 ...")
                time.sleep(wait)

    print(f"\n[错误] {model_key} 所有下载源均失败。")
    print("  建议:")
    print("  1. 检查网络连接")
    print("  2. 尝试另一个源: python model_manager.py download {model_key} --source hf")
    print("  3. 手动设置代理: export HTTPS_PROXY=http://127.0.0.1:7890")
    sys.exit(1)


def download_all(source: str = "modelscope"):
    """下载所有模型。"""
    for key in ALL_MODELS:
        download_model(key, source=source)


def main():
    parser = argparse.ArgumentParser(
        description="模型下载与管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python model_manager.py list
  python model_manager.py download qwen3-4b
  python model_manager.py download --all
  python model_manager.py download qwen3-4b --source hf

环境变量:
  LLM_CLASSIFY_MODEL_DIR  自定义模型存储目录
        """,
    )
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
        "--source",
        choices=["modelscope", "hf"],
        default="modelscope",
        help="下载源 (默认: modelscope，国内推荐)",
    )

    args = parser.parse_args()

    if args.command == "list":
        list_models()
    elif args.command == "download":
        if args.all:
            download_all(source=args.source)
        elif args.model:
            download_model(args.model, source=args.source)
        else:
            parser.error("请指定模型名称或使用 --all 下载全部")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
