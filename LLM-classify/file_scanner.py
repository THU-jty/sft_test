"""扫描指定目录，收集所有文件名。"""

import os
from pathlib import Path


def scan_directory(directory: str, recursive: bool = True) -> list[str]:
    """扫描目录下的所有文件，返回文件名列表（不含路径）。

    Args:
        directory: 要扫描的目录路径
        recursive: 是否递归扫描子目录
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"路径不是一个目录: {directory}")

    filenames = []
    if recursive:
        for root, _, files in os.walk(dir_path):
            filenames.extend(files)
    else:
        filenames = [f.name for f in dir_path.iterdir() if f.is_file()]

    if not filenames:
        print(f"[警告] 目录 {directory} 下没有找到任何文件")

    return filenames
