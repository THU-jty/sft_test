"""ONNX 动态量化模块 (INT8 per-channel weight-only，支持 CPU / CUDA / TensorRT EP)。

将 Qwen3 模型导出为 ONNX 格式，使用 onnxruntime.quantization.quantize_dynamic
进行 INT8 动态量化（per-channel, weight-only）。推理时通过 --ep 参数选择 EP：
  - cpu:       纯 CPU 推理
  - cuda:      CUDA EP（MatMulInteger 等 INT8 算子在 GPU 上执行）
  - tensorrt:  TensorRT EP
  - auto:      自动选择最佳 EP（默认，优先级 TensorRT > CUDA > CPU）

此模块完全独立，不影响 main.py 的分类流程。

用法:
  python onnx_quantize.py check                                  # 检查 EP 可用性
  python onnx_quantize.py export qwen3-8b                        # 导出 ONNX + INT8 量化
  python onnx_quantize.py benchmark qwen3-8b --ep cuda           # 用 CUDA EP 做 benchmark
  python onnx_quantize.py infer qwen3-8b "你好" --ep tensorrt    # 用 TensorRT EP 推理
"""

import argparse
import os
import sys
import time
import shutil
import tempfile
from pathlib import Path

from model_manager import ALL_MODELS, is_model_downloaded, get_model_local_path, get_model_dir

# 将所有临时文件也重定向到数据盘，避免撑满系统盘
_TMP_DIR = get_model_dir() / "_tmp"
_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(_TMP_DIR)
os.environ["TEMP"] = str(_TMP_DIR)
os.environ["TMP"] = str(_TMP_DIR)
tempfile.tempdir = str(_TMP_DIR)

# HuggingFace/optimum 的 cache 也重定向到数据盘
os.environ.setdefault("HF_HOME", str(get_model_dir() / "_hf_cache"))

# TensorRT EP 的引擎缓存目录，避免每次重新编译
TRT_ENGINE_CACHE = get_model_dir() / "_trt_engine_cache"


def get_onnx_dir(model_key: str) -> Path:
    return get_model_dir() / f"{model_key}-onnx"


def get_quantized_dir(model_key: str) -> Path:
    return get_model_dir() / f"{model_key}-onnx-int8-dynamic"


def _validate_model(model_key: str):
    if model_key not in ALL_MODELS:
        print(f"[错误] 未知模型: {model_key}")
        sys.exit(1)
    if ALL_MODELS[model_key]["type"] != "llm":
        print(f"[错误] {model_key} 不是 LLM 模型，仅支持对 LLM 模型做 ONNX 量化")
        sys.exit(1)
    if not is_model_downloaded(model_key):
        print(f"[错误] 模型 {model_key} 尚未下载")
        print(f"  请先运行: python model_manager.py download {model_key}")
        sys.exit(1)


def check_providers():
    """检查 ONNX Runtime 可用的 EP 并打印信息。"""
    import onnxruntime as ort

    available = ort.get_available_providers()
    print(f"ONNX Runtime 版本: {ort.__version__}")
    print(f"可用的 Execution Providers:")
    for ep in available:
        marker = "  ✅" if ep in ("TensorrtExecutionProvider", "CUDAExecutionProvider") else "  -"
        print(f"  {marker} {ep}")

    has_trt = "TensorrtExecutionProvider" in available
    has_cuda = "CUDAExecutionProvider" in available

    if has_trt:
        print("\n[OK] TensorRT EP 可用，将获得最佳 GPU 推理性能")
    elif has_cuda:
        print("\n[OK] CUDA EP 可用，可作为 TensorRT 的 fallback")
        print("  如需 TensorRT EP，请安装: pip install onnxruntime-gpu-tensorrt")
    else:
        print("\n[警告] 没有 GPU EP 可用，将 fallback 到 CPU")
        print("  安装 GPU 支持: pip install onnxruntime-gpu")

    return has_trt, has_cuda


def _patch_optimum_qwen3_config():
    """修复 optimum 中 Qwen3 的 NormalizedConfig 使用 GQA 版本。

    optimum < 1.24 对 qwen3 使用 NormalizedTextConfig，会用
    hidden_size/num_attention_heads 推算 head_dim，对于 GQA 模型这是错的。
    需要在导出和推理前都调用。
    """
    try:
        from optimum.utils.normalized_config import NormalizedConfigManager, NormalizedTextConfigWithGQA
        mapping = NormalizedConfigManager._conf
        if "qwen3" in mapping:
            current = mapping["qwen3"]
            if current is not NormalizedTextConfigWithGQA:
                mapping["qwen3"] = NormalizedTextConfigWithGQA
    except (ImportError, AttributeError):
        pass


_patch_optimum_qwen3_config()


EP_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "trt": "TensorrtExecutionProvider",
}

EP_CHOICES = ["auto", "cpu", "cuda", "tensorrt"]


def _get_provider(ep: str = "auto") -> tuple[str, dict]:
    """根据用户指定的 EP 返回 (provider_name, provider_options)。

    ep="auto" 时按 TensorRT > CUDA > CPU 的优先级自动选择。
    """
    import onnxruntime as ort

    available = ort.get_available_providers()

    if ep == "auto":
        if "TensorrtExecutionProvider" in available:
            ep = "tensorrt"
        elif "CUDAExecutionProvider" in available:
            ep = "cuda"
        else:
            ep = "cpu"

    ep_name = EP_ALIASES.get(ep, ep)

    if ep_name not in available and ep_name != "CPUExecutionProvider":
        print(f"[警告] 请求的 EP '{ep_name}' 不可用，可用: {available}")
        print(f"  将 fallback 到 CPUExecutionProvider")
        return "CPUExecutionProvider", {}

    if ep_name == "TensorrtExecutionProvider":
        TRT_ENGINE_CACHE.mkdir(parents=True, exist_ok=True)
        options = {
            "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,  # 4 GB
            "trt_fp16_enable": True,
            "trt_int8_enable": True,
            "trt_int8_use_native_calibration_table": False,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(TRT_ENGINE_CACHE),
        }
        return ep_name, options

    return ep_name, {}


def _get_mem_mb() -> float:
    """获取当前进程 RSS 内存（MB）。"""
    import psutil
    return psutil.Process().memory_info().rss / 1024**2


def _find_onnx_data_files(onnx_path: str) -> list[str]:
    """找到 ONNX 文件对应的 external data 文件（支持 .data 和 _data 命名）。"""
    candidates = [
        onnx_path + ".data",       # model.onnx.data
        onnx_path + "_data",       # model.onnx_data  (optimum 风格)
    ]
    return [f for f in candidates if os.path.exists(f)]


def _onnx_file_total_size_mb(onnx_path: str) -> float:
    """计算 ONNX 文件及其 external data 的总大小 (MB)。"""
    total = os.path.getsize(onnx_path)
    for data_file in _find_onnx_data_files(onnx_path):
        total += os.path.getsize(data_file)
    return total / (1024 ** 2)


def _quantize_model(src_path: str, dst_path: str):
    """使用 onnxruntime 官方 API 做 INT8 动态量化 (per-channel, weight-only)。"""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    src_size = _onnx_file_total_size_mb(src_path)
    has_external = len(_find_onnx_data_files(src_path)) > 0
    use_external = src_size > 1024 or has_external

    print(f"    模型大小: {src_size:.0f} MB | external data: {use_external}")
    print(f"    正在量化 (per-channel, INT8, weight-only) ...")
    print(f"    大模型量化需要大量内存和时间，请耐心等待 ...")
    sys.stdout.flush()

    t0 = time.perf_counter()
    quantize_dynamic(
        model_input=src_path,
        model_output=dst_path,
        per_channel=True,
        weight_type=QuantType.QInt8,
        use_external_data_format=use_external,
        extra_options={"MatMulConstBOnly": True},
    )
    elapsed = time.perf_counter() - t0

    dst_size = _onnx_file_total_size_mb(dst_path)
    print(f"    量化完成: {src_size:.0f} MB -> {dst_size:.0f} MB | 耗时: {elapsed:.0f}s")


def export_onnx(model_key: str):
    """导出 ONNX + INT8 动态量化。"""
    _validate_model(model_key)

    onnx_dir = get_onnx_dir(model_key)
    quantized_dir = get_quantized_dir(model_key)
    local_path = str(get_model_local_path(model_key))

    # Step 1: 导出 ONNX
    print(f"[Step 1] 导出 ONNX: {model_key}")
    print(f"  源模型: {local_path}")
    print(f"  ONNX 输出: {onnx_dir}")

    from optimum.onnxruntime import ORTModelForCausalLM

    if onnx_dir.exists():
        print(f"  ONNX 模型已存在，跳过导出")
    else:
        onnx_dir.mkdir(parents=True, exist_ok=True)
        print(f"  正在导出（可能需要几分钟）...")
        print(f"  临时目录: {_TMP_DIR}")
        ort_model = ORTModelForCausalLM.from_pretrained(
            local_path,
            export=True,
            trust_remote_code=True,
            cache_dir=str(get_model_dir() / "_hf_cache"),
        )
        ort_model.save_pretrained(str(onnx_dir))
        del ort_model

        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  ONNX 导出完成")

    _copy_tokenizer_files(local_path, str(onnx_dir))

    # Step 2: INT8 动态量化
    print(f"\n[Step 2] INT8 动态量化 (per-channel, weight-only)")
    print(f"  量化输出: {quantized_dir}")

    if quantized_dir.exists():
        print(f"  量化模型已存在，跳过量化")
    else:
        quantized_dir.mkdir(parents=True, exist_ok=True)

        onnx_files = list(onnx_dir.glob("*.onnx"))
        if not onnx_files:
            print(f"  [错误] 在 {onnx_dir} 下没有找到 .onnx 文件")
            sys.exit(1)

        for onnx_file in onnx_files:
            dst = str(quantized_dir / onnx_file.name)
            print(f"  量化: {onnx_file.name}")
            _quantize_model(str(onnx_file), dst)

        # 复制非模型文件（配置、tokenizer 等），跳过 onnx 模型及其 data
        onnx_names = {of.name for of in onnx_files}
        for of in onnx_files:
            for df in _find_onnx_data_files(str(of)):
                onnx_names.add(os.path.basename(df))

        for f in onnx_dir.iterdir():
            if f.name in onnx_names:
                continue
            dst_f = quantized_dir / f.name
            if not dst_f.exists():
                if f.is_file():
                    shutil.copy2(f, dst_f)
                elif f.is_dir():
                    shutil.copytree(f, dst_f)

        print(f"  INT8 动态量化完成")

    _copy_tokenizer_files(local_path, str(quantized_dir))

    # Step 3: 大小对比
    _print_size_comparison(model_key, local_path, str(onnx_dir), str(quantized_dir))

    # Step 4: 检查推理 EP
    print()
    check_providers()


def _copy_tokenizer_files(src_dir: str, dst_dir: str):
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json", "merges.txt",
        "generation_config.json", "config.json",
    ]
    for fname in tokenizer_files:
        src = Path(src_dir) / fname
        dst = Path(dst_dir) / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def _get_dir_size_mb(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def _get_onnx_model_size_mb(dir_path: str) -> float:
    """只统计目录中 ONNX 模型文件 (.onnx + 对应 external data) 的总大小。"""
    p = Path(dir_path)
    if not p.exists():
        return 0
    total = 0
    for f in p.iterdir():
        if f.suffix == ".onnx":
            total += f.stat().st_size
            for df in _find_onnx_data_files(str(f)):
                total += os.path.getsize(df)
    return total / (1024 * 1024)


def _print_size_comparison(model_key: str, original: str, onnx: str, quantized: str):
    orig_size = _get_dir_size_mb(original)
    onnx_size = _get_onnx_model_size_mb(onnx)
    quant_size = _get_onnx_model_size_mb(quantized)

    print(f"\n{'=' * 60}")
    print(f"模型大小对比: {model_key}")
    print(f"{'=' * 60}")
    print(f"  原始 PyTorch:        {orig_size:>10.1f} MB")
    print(f"  ONNX FP32:           {onnx_size:>10.1f} MB")
    print(f"  ONNX INT8 量化:      {quant_size:>10.1f} MB")
    if orig_size > 0 and quant_size > 0:
        ratio = quant_size / orig_size * 100
        print(f"  压缩比:              {ratio:>9.1f}%")
    print(f"{'=' * 60}")


def _load_ort_model(quantized_dir: Path, ep: str = "auto"):
    """加载 ONNX 量化模型，使用指定的 EP。"""
    from optimum.onnxruntime import ORTModelForCausalLM

    ep_name, ep_options = _get_provider(ep)
    print(f"  使用 EP: {ep_name}")

    model = ORTModelForCausalLM.from_pretrained(
        str(quantized_dir),
        trust_remote_code=True,
        provider=ep_name,
        provider_options=ep_options,
    )
    return model, ep_name


def benchmark(model_key: str, prompt: str = "你好，请用一句话介绍人工智能。", ep: str = "auto"):
    """对比 PyTorch (GPU) vs ONNX INT8 的显存和速度。"""
    _validate_model(model_key)

    quantized_dir = get_quantized_dir(model_key)
    if not quantized_dir.exists():
        print(f"[错误] 量化模型不存在，请先运行: python onnx_quantize.py export {model_key}")
        sys.exit(1)

    local_path = str(get_model_local_path(model_key))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    import torch
    has_cuda = torch.cuda.is_available()

    ep_name, _ = _get_provider(ep)
    uses_gpu = ep_name in ("CUDAExecutionProvider", "TensorrtExecutionProvider")

    print(f"{'=' * 60}")
    print(f"Benchmark: {model_key}")
    print(f"  Prompt: {prompt}")
    print(f"  ONNX EP: {ep_name}")
    print(f"{'=' * 60}")

    # --- 1. 原始 PyTorch 模型 (bfloat16, GPU) ---
    pt_peak = 0.0
    pt_time = 0.0
    pt_token_count = 0
    pt_response = ""
    mem_loaded = 0.0

    if has_cuda:
        print(f"\n[1/2] PyTorch 原始模型 (bfloat16, GPU)")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            local_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        model.eval()

        mem_loaded = torch.cuda.memory_allocated() / 1024**2
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=5, do_sample=False)

        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        pt_time = time.perf_counter() - start

        pt_peak = torch.cuda.max_memory_allocated() / 1024**2
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        pt_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pt_token_count = len(new_tokens)

        del model, inputs, outputs
        torch.cuda.empty_cache()

        print(f"  显存 (模型加载后): {mem_loaded:.1f} MB")
        print(f"  显存 (推理峰值):   {pt_peak:.1f} MB")
        print(f"  推理耗时:          {pt_time:.2f}s")
        print(f"  生成 token 数:     {pt_token_count}")
        print(f"  吞吐:              {pt_token_count / pt_time:.1f} tokens/s")
        print(f"  输出: {pt_response[:100]}...")
    else:
        print(f"\n[1/2] PyTorch 原始模型 -- 跳过（无 CUDA）")

    # --- 2. ONNX INT8 ---
    print(f"\n[2/2] ONNX INT8 量化模型 ({ep_name})")

    if has_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start_load = time.perf_counter()
    ort_model, ep_name = _load_ort_model(quantized_dir, ep=ep)
    load_time = time.perf_counter() - start_load

    mem_after_load = torch.cuda.memory_allocated() / 1024**2 if has_cuda else 0.0
    inputs_ort = tokenizer(text, return_tensors="pt")
    if uses_gpu:
        inputs_ort = {k: v.to("cuda") for k, v in inputs_ort.items()}

    print(f"  预热中...")
    ort_model.generate(**inputs_ort, max_new_tokens=5, do_sample=False)

    if has_cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    outputs_ort = ort_model.generate(**inputs_ort, max_new_tokens=100, do_sample=False)
    ort_time = time.perf_counter() - start

    ort_peak = torch.cuda.max_memory_allocated() / 1024**2 if has_cuda else 0.0
    new_tokens_ort = outputs_ort[0][inputs_ort["input_ids"].shape[1]:]
    ort_response = tokenizer.decode(new_tokens_ort, skip_special_tokens=True)
    ort_token_count = len(new_tokens_ort)

    del ort_model

    if uses_gpu and has_cuda:
        print(f"  显存 (模型加载后): {mem_after_load:.1f} MB")
        print(f"  显存 (推理峰值):   {ort_peak:.1f} MB")
    elif not uses_gpu:
        rss = _get_mem_mb()
        print(f"  内存 (RSS):        {rss:.1f} MB")
    print(f"  模型加载耗时:      {load_time:.2f}s")
    print(f"  推理耗时:          {ort_time:.2f}s")
    print(f"  生成 token 数:     {ort_token_count}")
    print(f"  吞吐:              {ort_token_count / ort_time:.1f} tokens/s")
    print(f"  输出: {ort_response[:100]}...")

    # --- 汇总 ---
    ort_label = f"ONNX INT8 ({ep_name.replace('ExecutionProvider', '')})"
    print(f"\n{'=' * 65}")
    print(f"汇总对比")
    print(f"{'=' * 65}")

    if has_cuda and pt_time > 0:
        print(f"  {'指标':<20} {'PyTorch (GPU)':>18} {ort_label:>22}")
        print(f"  {'-' * 60}")
        if uses_gpu:
            print(f"  {'显存峰值':<18} {pt_peak:>15.1f} MB {ort_peak:>19.1f} MB")
        print(f"  {'推理耗时':<18} {pt_time:>16.2f}s {ort_time:>20.2f}s")
        print(f"  {'吞吐量':<18} {pt_token_count/pt_time:>12.1f} tok/s {ort_token_count/ort_time:>16.1f} tok/s")
        print(f"  {'生成 token 数':<15} {pt_token_count:>18} {ort_token_count:>22}")

        if pt_peak > 0 and ort_peak > 0 and uses_gpu:
            mem_save = (1 - ort_peak / pt_peak) * 100
            print(f"\n  显存节省: {mem_save:.1f}%")
        if pt_time > 0 and ort_time > 0:
            speedup = pt_time / ort_time
            print(f"  速度提升: {speedup:.2f}x")
    else:
        print(f"  {ort_label}")
        print(f"  {'-' * 40}")
        print(f"  推理耗时:      {ort_time:.2f}s")
        print(f"  吞吐量:        {ort_token_count / ort_time:.1f} tok/s")
        print(f"  生成 token 数: {ort_token_count}")

    print(f"{'=' * 65}")


def infer(model_key: str, prompt: str, ep: str = "auto"):
    """使用量化后的 ONNX 模型推理。"""
    _validate_model(model_key)

    quantized_dir = get_quantized_dir(model_key)
    if not quantized_dir.exists():
        print(f"[错误] 量化模型不存在，请先运行: python onnx_quantize.py export {model_key}")
        sys.exit(1)

    local_path = str(get_model_local_path(model_key))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

    print(f"[加载 ONNX INT8 模型] {model_key}")
    ort_model, ep_name = _load_ort_model(quantized_dir, ep=ep)
    uses_gpu = ep_name in ("CUDAExecutionProvider", "TensorrtExecutionProvider")

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt")
    if uses_gpu:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    start = time.perf_counter()
    outputs = ort_model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9)
    elapsed = time.perf_counter() - start

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"  EP: {ep_name}")
    print(f"  耗时: {elapsed:.2f}s | token 数: {len(new_tokens)} | 吞吐: {len(new_tokens)/elapsed:.1f} tok/s")
    print(f"  输出:\n{response}")


def main():
    llm_keys = [k for k, v in ALL_MODELS.items() if v["type"] == "llm"]

    parser = argparse.ArgumentParser(
        description="ONNX 量化工具 (INT8 动态量化，支持 CPU / CUDA / TensorRT EP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
量化方式: quantize_dynamic (INT8, per-channel, weight-only)
推理后端: 通过 --ep 指定，可选 auto / cpu / cuda / tensorrt
         auto 模式自动选择最佳: TensorRT > CUDA > CPU

示例:
  python onnx_quantize.py check                                  # 检查 EP 可用性
  python onnx_quantize.py export qwen3-8b                        # 导出 + 量化
  python onnx_quantize.py benchmark qwen3-8b                     # 自动选择 EP
  python onnx_quantize.py benchmark qwen3-8b --ep cuda           # 指定 CUDA EP
  python onnx_quantize.py benchmark qwen3-8b --ep cpu            # 指定 CPU EP
  python onnx_quantize.py infer qwen3-8b "你好" --ep tensorrt    # 指定 TensorRT EP
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    subparsers.add_parser("check", help="检查 TensorRT / CUDA EP 是否可用")

    export_parser = subparsers.add_parser("export", help="导出 ONNX + INT8 动态量化")
    export_parser.add_argument("model", choices=llm_keys, help="要量化的模型")

    bench_parser = subparsers.add_parser("benchmark", help="对比量化前后的显存和速度")
    bench_parser.add_argument("model", choices=llm_keys, help="要测试的模型")
    bench_parser.add_argument("--prompt", default="你好，请用一句话介绍人工智能。", help="测试 prompt")
    bench_parser.add_argument("--ep", choices=EP_CHOICES, default="auto",
                              help="Execution Provider: auto/cpu/cuda/tensorrt (默认 auto)")

    infer_parser = subparsers.add_parser("infer", help="用量化模型推理")
    infer_parser.add_argument("model", choices=llm_keys, help="要使用的模型")
    infer_parser.add_argument("prompt", help="输入的 prompt")
    infer_parser.add_argument("--ep", choices=EP_CHOICES, default="auto",
                              help="Execution Provider: auto/cpu/cuda/tensorrt (默认 auto)")

    args = parser.parse_args()

    if args.command == "check":
        check_providers()
    elif args.command == "export":
        export_onnx(args.model)
    elif args.command == "benchmark":
        benchmark(args.model, prompt=args.prompt, ep=args.ep)
    elif args.command == "infer":
        infer(args.model, args.prompt, ep=args.ep)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
