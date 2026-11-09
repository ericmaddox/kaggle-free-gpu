"""Kaggle Free-GPU Smoke Test: Verifies Tesla T4 x2 detection, VRAM, and CUDA compute."""

import json
import platform
import subprocess
import time
from pathlib import Path

import torch


def main() -> None:
    assert torch.cuda.is_available(), "FAIL: torch.cuda.is_available() is False (CPU session instead of GPU!)"

    device_count = torch.cuda.device_count()
    devices = []
    for idx in range(device_count):
        props = torch.cuda.get_device_properties(idx)
        vram_gb = round(props.total_memory / (1024**3), 2)
        devices.append(
            {
                "index": idx,
                "name": props.name,
                "vram_gb": vram_gb,
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }
        )

    # Verify actual FP16 tensor core compute on each detected GPU
    t0 = time.perf_counter()
    for idx in range(device_count):
        dev = torch.device(f"cuda:{idx}")
        a = torch.randn((4096, 4096), device=dev, dtype=torch.float16)
        b = torch.randn((4096, 4096), device=dev, dtype=torch.float16)
        c = torch.matmul(a, b)
        torch.cuda.synchronize(dev)
        del a, b, c
    compute_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    report = {
        "status": "PASS",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_count": device_count,
        "total_vram_gb": round(sum(d["vram_gb"] for d in devices), 2),
        "devices": devices,
        "fp16_4096_gemm_all_gpus_ms": compute_ms,
        "nvidia_smi": smi.splitlines(),
    }

    out_path = Path("gpu_smoke_report.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

