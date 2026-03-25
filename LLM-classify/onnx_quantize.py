"""ONNX 动态量化模块 (QDQ 格式，支持 CPU / CUDA / TensorRT EP)。

将 Qwen3 模型导出为 ONNX 格式，使用 QDQ（QuantizeLinear/DequantizeLinear）格式
进行 INT8 动态量化。推理时通过 --ep 参数选择 Execution Provider：
  - cpu:       纯 CPU 推理
  - cuda:      CUDA EP（GPU 上跑 FP32 计算，权重 INT8 存储节省显存）
  - tensorrt:  TensorRT EP（真正利用 INT8 tensor core 加速）
  - auto:      自动选择最佳 EP（默认，优先级 TensorRT > CUDA > CPU）

此模块完全独立，不影响 main.py 的分类流程。

用法:
  python onnx_quantize.py check                                  # 检查 EP 可用性
  python onnx_quantize.py export qwen3-8b                        # 导出 ONNX + QDQ 量化
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
    return get_model_dir() / f"{model_key}-onnx-qdq-int8"


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


def _quantize_incremental(src_path: str, dst_path: str):
    """增量式 QDQ 动态量化：逐个权重量化，控制峰值内存。

    不使用 onnxruntime.quantization.quantize_dynamic（它会一次性加载整个图），
    而是手动遍历 ONNX 模型中的 initializer（权重），对 MatMul/Gemm 的权重
    逐个做 INT8 量化并插入 QDQ 节点。
    """
    import gc
    import numpy as np
    import onnx
    from onnx import numpy_helper, TensorProto, helper

    mem_start = _get_mem_mb()
    print(f"    内存 (开始): {mem_start:.0f} MB")

    # 用 load 的 external data 模式，避免把所有 tensor 全部读入内存
    print(f"    加载 ONNX 模型结构 ...")
    model = onnx.load(src_path, load_external_data=False)

    # 收集 external data 的基目录
    model_dir = os.path.dirname(src_path)

    # 找出所有 MatMul / Gemm 节点的权重 initializer 名称
    matmul_weight_names = set()
    for node in model.graph.node:
        if node.op_type in ("MatMul", "Gemm"):
            # 第二个输入通常是权重（常量）
            if len(node.input) >= 2:
                matmul_weight_names.add(node.input[1])

    # 过滤出真正是 initializer 的权重
    initializer_map = {init.name: init for init in model.graph.initializer}
    target_weights = [name for name in matmul_weight_names if name in initializer_map]
    target_weights.sort()

    total = len(target_weights)
    print(f"    找到 {total} 个 MatMul/Gemm 权重需要量化")

    if total == 0:
        print(f"    没有需要量化的权重，直接复制原始模型")
        onnx.save_model(model, dst_path)
        return

    quantized_count = 0
    new_initializers = []
    nodes_to_add = []

    for i, weight_name in enumerate(target_weights):
        init = initializer_map[weight_name]

        # 加载单个权重的数据
        try:
            # 尝试从 external data 加载
            if init.data_location == TensorProto.EXTERNAL:
                for entry in init.external_data:
                    if entry.key == "location":
                        data_file = os.path.join(model_dir, entry.value)
                        offset = 0
                        length = 0
                        for e in init.external_data:
                            if e.key == "offset":
                                offset = int(e.value)
                            elif e.key == "length":
                                length = int(e.value)
                        with open(data_file, "rb") as f:
                            f.seek(offset)
                            raw = f.read(length if length > 0 else -1)
                        weight = np.frombuffer(raw, dtype=np.float16 if init.data_type == TensorProto.FLOAT16 else np.float32)
                        weight = weight.reshape([d for d in init.dims])
                        break
                else:
                    weight = numpy_helper.to_array(init)
            else:
                weight = numpy_helper.to_array(init)
        except Exception as e:
            print(f"    [{i+1}/{total}] 跳过 {weight_name}: 无法读取 ({e})")
            continue

        # 转为 float32 做量化计算
        weight_f32 = weight.astype(np.float32)

        # 动态量化: 对称量化到 INT8
        abs_max = np.max(np.abs(weight_f32))
        if abs_max == 0:
            scale = np.float32(1.0)
        else:
            scale = np.float32(abs_max / 127.0)
        zero_point = np.int8(0)
        weight_int8 = np.clip(np.round(weight_f32 / scale), -128, 127).astype(np.int8)

        # 创建 QDQ 节点对应的 tensor
        scale_name = f"{weight_name}_scale"
        zp_name = f"{weight_name}_zero_point"
        dq_output_name = f"{weight_name}_dequantized"

        scale_tensor = numpy_helper.from_array(np.array(scale), name=scale_name)
        zp_tensor = numpy_helper.from_array(np.array(zero_point), name=zp_name)
        quant_weight_tensor = numpy_helper.from_array(weight_int8, name=f"{weight_name}_quantized")

        new_initializers.extend([quant_weight_tensor, scale_tensor, zp_tensor])

        # DequantizeLinear 节点
        dq_node = helper.make_node(
            "DequantizeLinear",
            inputs=[f"{weight_name}_quantized", scale_name, zp_name],
            outputs=[dq_output_name],
            name=f"{weight_name}_DequantizeLinear",
        )
        nodes_to_add.append(dq_node)

        # 替换原始节点中的输入引用
        for node in model.graph.node:
            for j, inp in enumerate(node.input):
                if inp == weight_name:
                    node.input[j] = dq_output_name

        # 释放原始权重数据
        del weight, weight_f32, weight_int8
        quantized_count += 1

        if (i + 1) % 10 == 0 or (i + 1) == total:
            gc.collect()
            mem_now = _get_mem_mb()
            print(f"    [{i+1}/{total}] 已量化 {quantized_count} 个权重 | 内存: {mem_now:.0f} MB")

    # 移除已量化的原始 initializer
    remaining_inits = [init for init in model.graph.initializer if init.name not in matmul_weight_names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(remaining_inits)
    model.graph.initializer.extend(new_initializers)

    # 插入 DequantizeLinear 节点到图的最前面
    for n in reversed(nodes_to_add):
        model.graph.node.insert(0, n)

    print(f"    保存量化模型 ...")
    mem_before_save = _get_mem_mb()
    print(f"    内存 (保存前): {mem_before_save:.0f} MB")

    onnx.save_model(
        model,
        dst_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(dst_path) + ".data",
        size_threshold=1024,
    )

    del model
    gc.collect()

    mem_end = _get_mem_mb()
    print(f"    内存 (完成): {mem_end:.0f} MB | 峰值增量: {max(mem_before_save, mem_end) - mem_start:.0f} MB")
    print(f"    量化完成: {quantized_count}/{total} 个权重")


def export_onnx(model_key: str):
    """导出 ONNX + QDQ 动态量化 (INT8)。"""
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

    # Step 2: QDQ 动态量化
    print(f"\n[Step 2] QDQ 动态量化 (INT8)")
    print(f"  格式: QDQ (QuantizeLinear / DequantizeLinear)")
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
            _quantize_incremental(str(onnx_file), dst)
            print(f"    -> {dst}")

        # 复制非 onnx 文件（配置等）
        for f in onnx_dir.iterdir():
            if f.suffix != ".onnx":
                dst_f = quantized_dir / f.name
                if not dst_f.exists():
                    if f.is_file():
                        shutil.copy2(f, dst_f)
                    elif f.is_dir():
                        shutil.copytree(f, dst_f)

        print(f"  QDQ 量化完成")

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


def _print_size_comparison(model_key: str, original: str, onnx: str, quantized: str):
    orig_size = _get_dir_size_mb(original)
    onnx_size = _get_dir_size_mb(onnx) if Path(onnx).exists() else 0
    quant_size = _get_dir_size_mb(quantized) if Path(quantized).exists() else 0

    print(f"\n{'=' * 60}")
    print(f"模型大小对比: {model_key}")
    print(f"{'=' * 60}")
    print(f"  原始 PyTorch:        {orig_size:>10.1f} MB")
    print(f"  ONNX FP32:           {onnx_size:>10.1f} MB")
    print(f"  ONNX QDQ INT8 量化:  {quant_size:>10.1f} MB")
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
    """对比 PyTorch (GPU) vs ONNX QDQ INT8 的显存和速度。"""
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

    # --- 2. ONNX QDQ INT8 ---
    print(f"\n[2/2] ONNX QDQ INT8 量化模型 ({ep_name})")

    if has_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start_load = time.perf_counter()
    ort_model, ep_name = _load_ort_model(quantized_dir, ep=ep)
    load_time = time.perf_counter() - start_load

    mem_after_load = torch.cuda.memory_allocated() / 1024**2 if has_cuda else 0.0
    inputs_ort = tokenizer(text, return_tensors="pt")

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
    ort_label = f"ONNX QDQ ({ep_name.replace('ExecutionProvider', '')})"
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
    """使用 QDQ 量化后的 ONNX 模型推理。"""
    _validate_model(model_key)

    quantized_dir = get_quantized_dir(model_key)
    if not quantized_dir.exists():
        print(f"[错误] 量化模型不存在，请先运行: python onnx_quantize.py export {model_key}")
        sys.exit(1)

    local_path = str(get_model_local_path(model_key))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

    print(f"[加载 ONNX QDQ INT8 模型] {model_key}")
    ort_model, ep_name = _load_ort_model(quantized_dir, ep=ep)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt")

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
        description="ONNX 量化工具 (QDQ 格式，支持 CPU / CUDA / TensorRT EP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
量化格式: QDQ (QuantizeLinear / DequantizeLinear)
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

    export_parser = subparsers.add_parser("export", help="导出 ONNX + QDQ 量化")
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
