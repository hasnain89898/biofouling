#!/usr/bin/env python3
"""
ONNX Model Exporter & Edge Deployment Validator for Industrial Biofouling
-------------------------------------------------------------------------
Features:
1. Exports PyTorch BiofoulingClassifier to ONNX format with dynamic batching.
2. Validates ONNX graph integrity with onnx.checker.
3. Performs strict numerical parity verification between PyTorch and ONNX Runtime (atol < 1e-4).
4. Benchmarks inference latency and throughput for industrial edge gateways (NVIDIA Jetson / x86).

Author: Principal AI Research Engineer
"""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe Windows stdout encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.model import BiofoulingClassifier, create_biofouling_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BiofoulingONNXExporter")


def export_to_onnx(
    checkpoint_path: Path,
    output_onnx_path: Path,
    image_size: int = 380,
    opset_version: int = 18,
) -> Tuple[Path, Dict[str, float]]:
    """
    Export PyTorch model checkpoint to ONNX format with dynamic batching.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    device = torch.device("cpu")  # Export on CPU for universal portability
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_name = checkpoint.get("model_name", "efficientnet_b3")
    drop_rate = checkpoint.get("drop_rate", 0.4)
    logger.info(f"Loaded checkpoint {checkpoint_path} (Model: {model_name}, Dropout: {drop_rate})")

    model = create_biofouling_model(
        model_name=model_name,
        num_classes=4,
        pretrained=False,
        drop_rate=drop_rate,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    input_names = ["input"]
    output_names = ["logits"]
    dynamic_axes = {
        "input": {0: "batch_size"},
        "logits": {0: "batch_size"},
    }

    logger.info(f"Exporting model to ONNX: {output_onnx_path} (opset={opset_version}, input=[batch, 3, {image_size}, {image_size}])...")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    file_size_mb = output_onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"Successfully generated ONNX artifact: {output_onnx_path} ({file_size_mb:.2f} MB)")

    # 1. Structural Check
    onnx_model = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX Graph Structure Validation: PASSED (Graph is well-formed)")

    # 2. Numerical Parity Verification
    logger.info("Running Numerical Parity Verification across dynamic batch sizes (1, 4, 8)...")
    ort_session = ort.InferenceSession(str(output_onnx_path), providers=["CPUExecutionProvider"])

    max_discrepancies = []
    for test_bs in [1, 4, 8]:
        test_x = torch.randn(test_bs, 3, image_size, image_size)
        with torch.no_grad():
            pt_out = model(test_x).numpy()

        ort_inputs = {ort_session.get_inputs()[0].name: test_x.numpy()}
        ort_out = ort_session.run(None, ort_inputs)[0]

        diff = np.max(np.abs(pt_out - ort_out))
        max_discrepancies.append(diff)
        logger.info(f"  - Batch Size {test_bs}: Max Absolute Logit Difference = {diff:.6e}")
        assert diff < 1e-4, f"Discrepancy too large at batch size {test_bs}: {diff}"

    logger.info("Numerical Parity Verification: PASSED (All predictions match within 1e-4 tolerance)")

    # 3. Latency Benchmarking (50 warmup iterations, 100 test iterations)
    logger.info("Benchmarking Inference Latency (ONNX Runtime vs PyTorch on CPU)...")
    bench_bs = 1
    bench_x_np = np.random.randn(bench_bs, 3, image_size, image_size).astype(np.float32)
    bench_x_pt = torch.from_numpy(bench_x_np)

    # Warmup
    for _ in range(20):
        _ = model(bench_x_pt)
        _ = ort_session.run(None, {ort_session.get_inputs()[0].name: bench_x_np})

    # PyTorch CPU timing
    iters = 50
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(bench_x_pt)
    pt_latency_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    # ONNX Runtime CPU timing
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = ort_session.run(None, {ort_session.get_inputs()[0].name: bench_x_np})
    ort_latency_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    logger.info(
        f"Benchmark (Batch Size {bench_bs}, {image_size}x{image_size}): "
        f"PyTorch CPU = {pt_latency_ms:.2f} ms | ONNX Runtime = {ort_latency_ms:.2f} ms "
        f"({(pt_latency_ms / max(1e-3, ort_latency_ms)):.2f}x speedup)"
    )

    metrics = {
        "file_size_mb": file_size_mb,
        "max_discrepancy": float(max(max_discrepancies)),
        "pytorch_latency_ms": pt_latency_ms,
        "onnx_latency_ms": ort_latency_ms,
    }

    # Also copy to checkpoints directory if outputting to root
    if output_onnx_path.resolve() != (PROJECT_ROOT / "checkpoints" / "biofouling_model.onnx").resolve():
        compat_path = PROJECT_ROOT / "checkpoints" / "biofouling_model.onnx"
        compat_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_onnx_path, compat_path)
        logger.info(f"Synchronized copy to: {compat_path}")

    return output_onnx_path, metrics


def main():
    parser = argparse.ArgumentParser(description="Export Biofouling Model to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_v4.pth", help="PyTorch checkpoint path")
    parser.add_argument("--output", type=str, default="biofouling_model.onnx", help="Output ONNX path")
    parser.add_argument("--image-size", type=int, default=380, help="Input image resolution")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")

    args = parser.parse_args()

    ckpt_p = Path(args.checkpoint)
    if not ckpt_p.exists():
        # Fallback to best_model_v3 or best_model if v4 not found
        for alt in ["checkpoints/best_model_v3.pth", "checkpoints/best_model.pth"]:
            if Path(alt).exists():
                ckpt_p = Path(alt)
                break

    export_to_onnx(
        checkpoint_path=ckpt_p,
        output_onnx_path=Path(args.output),
        image_size=args.image_size,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
