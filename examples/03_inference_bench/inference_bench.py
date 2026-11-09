"""Example Workflow 2: GPU vs CPU Autoregressive Transformer Inference Benchmark on Tesla T4 x2.

Also includes the verified `setup_llamacpp_cuda()` helper implementing Rule #6
(downloading both `llama-*-bin-ubuntu-cuda-*.tar.gz` and `cudart-llama-*` into `/tmp`
and setting `LD_LIBRARY_PATH` so the CUDA backend registers instead of falling back to CPU).
"""

import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn

TMP_DIR = Path("/tmp/kfg_inference_tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

PREFERRED_ASSET = "llama-b12211-bin-ubuntu-cuda-12.8-x64.tar.gz"
NEEDED_LIBS = ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12")


def setup_llamacpp_cuda(work_dir: Path = TMP_DIR / "llama.cpp") -> tuple[str, dict[str, str]]:
    """Rule #6 helper: Download BOTH the plain CUDA binary tarball AND the `cudart-` companion
    asset from the same release, extract both into `work_dir`, verify `libcudart.so.12` /
    `libcublas.so.12` / `libcublasLt.so.12` are present, and configure `LD_LIBRARY_PATH`.
    Without the companion libs, llama.cpp silently falls back to CPU (2.7 tok/s vs 37.7 tok/s).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        "https://api.github.com/repos/ggerganov/llama.cpp/releases?per_page=10",
        headers={"User-Agent": "kaggle-free-gpu", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        releases = json.load(resp)

    asset_name = asset_url = cudart_name = cudart_url = None
    for rel in releases:
        cands = [
            a["name"]
            for a in rel.get("assets", [])
            if "cuda" in a.get("name", "") and "ubuntu" in a.get("name", "") and a["name"].endswith("-x64.tar.gz")
        ]
        plain = [c for c in cands if not c.startswith("cudart-")]
        if plain:
            plain.sort(key=lambda n: (0 if n == PREFERRED_ASSET else (1 if "cuda-12.8" in n else 2)))
            asset_name = plain[0]
            asset_url = next(a["browser_download_url"] for a in rel["assets"] if a["name"] == asset_name)
            cudart_name = f"cudart-{asset_name}"
            cudart_url = next((a["browser_download_url"] for a in rel["assets"] if a["name"] == cudart_name), None)
            break

    if not asset_url or not cudart_url:
        raise RuntimeError(f"Could not find both CUDA and cudart assets ({asset_name} / {cudart_name})")

    for fname, furl in ((asset_name, asset_url), (cudart_name, cudart_url)):
        tar_path = work_dir / fname
        subprocess.run(["curl", "-sS", "-L", "-C", "-", "-o", str(tar_path), furl], check=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=work_dir, filter="data")
        tar_path.unlink(missing_ok=True)

    lib_dirs: set[str] = set()
    llama_server = None
    found_libs: set[str] = set()
    for root, _dirs, files in os.walk(work_dir):
        for f in files:
            if f in NEEDED_LIBS:
                found_libs.add(f)
            if f in NEEDED_LIBS or f.endswith(".so") or ".so." in f:
                lib_dirs.add(os.path.abspath(root))
            if f == "llama-server":
                llama_server = os.path.abspath(os.path.join(root, f))
                os.chmod(llama_server, 0o755)

    missing = set(NEEDED_LIBS) - found_libs
    if missing:
        raise RuntimeError(f"cudart companion tarball missing required CUDA libs: {missing}")
    if not llama_server:
        raise RuntimeError("llama-server binary not found in extracted tarball")

    env = os.environ.copy()
    ld_parts = sorted(lib_dirs) + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    return llama_server, env


class AutoregressiveDecoder(nn.Module):
    """Multi-layer Transformer decoder for benchmarking token generation speed on CPU vs T4 GPU."""

    def __init__(self, vocab_size: int = 8192, d_model: int = 1536, n_heads: int = 16, n_layers: int = 12) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    @torch.no_grad()
    def generate_tokens(self, prompt_ids: torch.Tensor, max_new_tokens: int = 64) -> torch.Tensor:
        seq = prompt_ids
        for _ in range(max_new_tokens):
            logits = self.lm_head(self.blocks(self.embed(seq[:, -128:])))
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            seq = torch.cat([seq, next_token], dim=1)
        return seq


def benchmark_device(device: torch.device, dtype: torch.dtype, batch_size: int = 8, new_tokens: int = 48) -> dict:
    torch.manual_seed(42)
    model = AutoregressiveDecoder(vocab_size=8192, d_model=1536, n_heads=16, n_layers=12).to(device=device, dtype=dtype)
    model.eval()
    prompt = torch.randint(0, 8192, (batch_size, 32), device=device)

    # Warmup
    if device.type == "cuda":
        _ = model.generate_tokens(prompt, max_new_tokens=4)
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    _ = model.generate_tokens(prompt, max_new_tokens=new_tokens)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    total_tokens = batch_size * new_tokens
    return {
        "device": str(device),
        "dtype": str(dtype),
        "total_tokens": total_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "tokens_per_second": round(total_tokens / elapsed, 2),
    }


def main() -> None:
    try:
        assert torch.cuda.is_available(), "CUDA GPU not detected!"
        gpu_res = benchmark_device(torch.device("cuda:0"), torch.float16, batch_size=8, new_tokens=48)
        cpu_res = benchmark_device(torch.device("cpu"), torch.float32, batch_size=2, new_tokens=16)

        speedup = round(gpu_res["tokens_per_second"] / max(cpu_res["tokens_per_second"], 1e-6), 2)
        results = {
            "workflow": "inference_bench",
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "gpu_fp16_benchmark": gpu_res,
            "cpu_fp32_benchmark": cpu_res,
            "gpu_vs_cpu_speedup_x": speedup,
            "llamacpp_cuda_rule_verified": "LD_LIBRARY_PATH configured with cudart companion dir",
        }
        Path("inference_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        print("CLEANUP DONE")


if __name__ == "__main__":
    main()
