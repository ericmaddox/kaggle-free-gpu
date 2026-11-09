# kaggle-free-gpu

<p align="center">
  <img src="https://raw.githubusercontent.com/ericmaddox/kaggle-free-gpu/main/assets/hero.jpg" alt="Kaggle Free-GPU Enterprise Compute" width="100%">
</p>

[![PyPI - Version](https://img.shields.io/pypi/v/kaggle-free-gpu?logo=pypi&logoColor=white&color=blue)](https://pypi.org/project/kaggle-free-gpu/)
[![PyPI - Status](https://img.shields.io/pypi/status/kaggle-free-gpu?color=blue)](https://pypi.org/project/kaggle-free-gpu/)
[![GitHub Release](https://img.shields.io/github/v/release/ericmaddox/kaggle-free-gpu?color=blue&logo=github)](https://github.com/ericmaddox/kaggle-free-gpu/releases)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![MCP Protocol](https://img.shields.io/badge/MCP-1.3.0%2B-purple.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-12%20passed-brightgreen.svg)](tests/)

**MCP Server, CLI, and Agent Skill for Zero-Cost Tesla T4 x2 GPU Execution via the Kaggle REST API.**

`kaggle-free-gpu` provides coding agents (Claude Code, Cursor, Antigravity) and ML engineers with programmatic access to Kaggle's dual NVIDIA Tesla T4 GPUs (2 x 15,360 MiB VRAM, Python 3.13, PyTorch 2.11+cu130) over direct REST and RPC calls to `https://www.kaggle.com/api/v1`.

---

## Overview & Engineering Rationale

Coding agents can author PyTorch training loops, QLoRA fine-tuning pipelines, and CUDA/GGUF inference harnesses, but local agent execution environments rarely have dedicated GPU hardware. Kaggle provides up to 30 GPU-hours per rolling 7-day window on dual Tesla T4 accelerators at zero cost. However, automating Kaggle kernel execution from an autonomous agent fails with standard tooling due to undocumented API invariants:

1. **Broken Official CLI Host:** Kaggle CLI 1.8 routes requests to `api.kaggle.com`, which fails for kernel session operations. All requests must target `https://www.kaggle.com/api/v1` directly.
2. **Title-Derived Slug Resolution:** Kaggle derives the canonical URL slug of a kernel from `newTitle`, ignoring the slug portion of `id` in `kernel-metadata.json`. Calling status or output endpoints with an `id`-based slug that diverges from the title-derived slug returns `HTTP 403 Forbidden`. Attempting to update an existing kernel when `id` does not match the title-derived slug is interpreted as a new kernel creation and fails with `HTTP 409 Conflict`.
3. **Undocumented Service Endpoints:** Standard REST paths such as `/kernels/status/{owner}/{slug}` and `/kernels/output/{owner}/{slug}` do not exist. Session status, output retrieval, and kernel deletion require JSON POST requests to `kernels.KernelsApiService` RPC endpoints.
4. **Working Directory Output Saturation:** `ListKernelSessionOutput` returns signed URLs for every file left in `/kaggle/working`. Leaving multi-gigabyte model checkpoints or GGUF weights in `/kaggle/working` causes output retrieval to fail with `IncompleteRead`.
5. **Silent CPU Fallback in `llama.cpp` CUDA Releases:** Prebuilt `llama.cpp` Ubuntu CUDA tarballs do not bundle `libcudart.so.12`, `libcublas.so.12`, or `libcublasLt.so.12`. Unless the companion `cudart-llama-*` archive from the same release is extracted into the same directory and added to `LD_LIBRARY_PATH`, the CUDA backend fails to register without emitting an error and executes on CPU (~2.7 tok/s vs ~37.7 tok/s on Tesla T4).

`kaggle-free-gpu` encodes these invariants into pre-push static validation, `HEAD` `Content-Length` output guards, a local rolling 7-day GPU quota ledger, a Model Context Protocol (MCP) server, and a terminal CLI (`kfg`).

---

## Hardware & Runtime Specifications

Verified live on Kaggle GPU sessions (`machineShape: NvidiaTeslaT4`):

| Parameter | Specification |
| :--- | :--- |
| **Accelerators** | 2 x NVIDIA Tesla T4 (Compute Capability 7.5, 40 SMs per GPU) |
| **VRAM** | 2 x 15,360 MiB (30,720 MiB total / 29.12 GiB usable) |
| **NVIDIA Driver / CUDA** | Driver `580.178.04` / CUDA `13.0` |
| **Python / PyTorch** | Python `3.13.16` / `torch 2.11.0+cu130` |
| **Weekly Quota** | ~30 GPU-hours per rolling 7-day window (~15 wall-clock hours on T4 x2) |
| **Maximum Session Duration** | 12 hours per execution |

---

## Installation

### Using `uv` (Recommended)

```bash
# Install CLI (`kaggle-free-gpu` / `kfg`) and MCP server (`kaggle-free-gpu-mcp`) globally:
uv tool install git+https://github.com/ericmaddox/kaggle-free-gpu.git

# Or install in a local development checkout:
git clone https://github.com/ericmaddox/kaggle-free-gpu.git
cd kaggle-free-gpu
uv sync --extra dev
```

### Using `pip`

```bash
pip install git+https://github.com/ericmaddox/kaggle-free-gpu.git
```

---

## Authentication

Generate a Kaggle API token at [https://www.kaggle.com/settings](https://www.kaggle.com/settings). Credentials are resolved in the following order and are never logged or written to repository files:

1. `KAGGLE_API_TOKEN` environment variable (`Bearer KGAT_...`)
2. `~/.kaggle/access_token` file (`Bearer KGAT_...`)
3. `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables (`Basic` auth)
4. `~/.kaggle/kaggle.json` file (`Basic` auth)

```bash
mkdir -p ~/.kaggle
echo "KGAT_your_token_here" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

---

## MCP Server Configuration

Register `kaggle-free-gpu-mcp` in your agent's MCP configuration (`mcp_config.json`, `.mcp.json`, or `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "kaggle-free-gpu": {
      "command": "kaggle-free-gpu-mcp",
      "env": {
        "KAGGLE_API_TOKEN": "KGAT_your_token_here"
      }
    }
  }
}
```

### MCP Tools Reference

| Tool | Signature | Description |
| :--- | :--- | :--- |
| `push_kernel` | `(dir: str, allow_mismatch: bool = False)` | Validates `kernel-metadata.json` (`id` slug vs title-derived slug, code file existence, privacy flag), pushes via `POST /api/v1/kernels/push` (`application/json`), reports rejected taxonomy tags (`invalid_tags`), and records the run in the local quota ledger. |
| `kernel_status` | `(owner_slug: str)` | Queries `POST /api/v1/kernels.KernelsApiService/GetKernelSessionStatus` and returns `QUEUED`, `RUNNING`, `COMPLETE`, or `ERROR`. |
| `kernel_output` | `(owner_slug: str, out_dir: str, max_size_mb: int = 500, force_large: bool = False)` | Queries `POST /api/v1/kernels.KernelsApiService/ListKernelSessionOutput`, inspects `Content-Length` via `HEAD` before downloading files from `/kaggle/working`, writes `kernel.log`, and updates the quota ledger with measured runtime from the execution log. |
| `list_kernels` | `(mine: bool = True, user: str \| None = None, search: str \| None = None, page_size: int = 20)` | Lists kernels via `GET /api/v1/kernels/list` (uses `group=profile` when `mine=True`). |
| `delete_kernel` | `(owner_slug: str, confirm: bool = False)` | Permanently deletes a kernel via `POST /api/v1/kernels.KernelsApiService/DeleteKernel`. Refuses execution unless `confirm=True` is explicitly passed after user confirmation. |
| `quota_estimate` | `()` | Estimates rolling 7-day GPU-hours and T4 x2 wall-hours remaining from `~/.kaggle-free-gpu/usage.json`. Note: Kaggle provides no API endpoint for live quota; check Settings -> GPU Quota on the Kaggle web UI for the exact balance. |

---

## CLI Usage (`kfg` / `kaggle-free-gpu`)

```bash
# 1. Estimate remaining rolling 7-day GPU quota
uv run kfg quota

# 2. Validate and push a kernel directory, poll until COMPLETE, and download outputs
uv run kfg push examples/01_gpu_smoke_test --wait --out-dir outputs/smoke

# 3. Check execution status of a kernel
uv run kfg status <owner>/kfg-gpu-smoke-test

# 4. Pull output files and execution logs with a 500 MB per-file safety guard
uv run kfg output <owner>/kfg-gpu-smoke-test --out-dir outputs/smoke --max-size-mb 500

# 5. List your own kernels
uv run kfg list --mine

# 6. Permanently delete a kernel (interactive confirmation or --yes)
uv run kfg delete <owner>/kfg-gpu-smoke-test
```

> **Note on `<owner>` placeholder resolution:** All `kernel-metadata.json` files in `examples/` use `"id": "<owner>/<slug>"`. `kaggle-free-gpu` automatically resolves `<owner>` from your authenticated Kaggle account at push time.

---

## Benchmarks: CPU vs. Kaggle Tesla T4 x2

Measured live on Kaggle (`2 x Tesla T4`, Python `3.13.16`, `torch 2.11.0+cu130`) across the repository's reference workflows:

| Workload | CPU Baseline | Kaggle Tesla T4 ($0) | Speedup / Outcome |
| :--- | :--- | :--- | :--- |
| **Autoregressive Transformer Decoding** (`examples/03_inference_bench`) | `4.99 tok/s` (`6.42s` for 32 tokens, FP32) | **`456.07 tok/s`** (`0.84s` for 384 tokens, FP16) | **91.4x speedup** |
| **9B Q4_K_M GGUF Inference (`llama.cpp`)** (Rule #6 verification) | `2.70 tok/s` (without `cudart` companion libs) | **`37.70 tok/s`** (with `cudart` + `LD_LIBRARY_PATH`) | **14.0x speedup** |
| **QLoRA 4-Bit Base + LoRA Fine-Tuning** (`examples/02_qlora_finetune`) | Impractical on CPU | **`19,092.1 tok/s`** (`6.44s` for 60 steps; loss `8.9066 -> 0.0005`) | **60 steps in 6.44s** |
| **Batch 32K x 768 Cosine Top-K + SVD** (`examples/04_data_processing`) | `11,806.2 docs/s` (`2.71s`) | **`58,595.5 docs/s`** (`0.55s` for 256M cosine pairs + PCA) | **4.96x speedup** |

### CLI Execution Transcript

```text
$ uv run kfg push examples/02_qlora_finetune --wait --out-dir outputs/qlora
+--------------------------- Kaggle Free-GPU Push ----------------------------+
| Pushed successfully!                                                        |
| Ref: <owner>/kfg-qlora-finetune-demo                                        |
| Title: KFG QLoRA Finetune Demo                                              |
| Version: 3                                                                  |
| GPU Enabled: True | Private: True                                           |
| URL: https://www.kaggle.com/code/<owner>/kfg-qlora-finetune-demo            |
+-----------------------------------------------------------------------------+
[0s]  <owner>/kfg-qlora-finetune-demo -> QUEUED
[10s] <owner>/kfg-qlora-finetune-demo -> RUNNING
[50s] <owner>/kfg-qlora-finetune-demo -> COMPLETE
Output downloaded to: outputs/qlora (runtime: 32.99s)
+---------------------------------- Stdout -----------------------------------+
| {                                                                           |
|   "workflow": "qlora_finetune",                                             |
|   "gpu_name": "Tesla T4",                                                   |
|   "gpu_count": 2,                                                           |
|   "trainable_lora_params": 1878016,                                         |
|   "frozen_4bit_packed_mb": 38.0,                                            |
|   "steps": 60,                                                              |
|   "initial_loss": 8.9066,                                                   |
|   "final_loss": 0.0005,                                                     |
|   "gpu_train_seconds": 6.436,                                               |
|   "tokens_per_second": 19092.1,                                             |
|   "saved_adapter_file": "lora_adapter.pt"                                   |
| }                                                                           |
| CLEANUP DONE                                                                |
+-----------------------------------------------------------------------------+
```

---

## Reference Workflows (`examples/`)

- **[`examples/01_gpu_smoke_test`](examples/01_gpu_smoke_test):** Queries `torch.cuda` device properties and `nvidia-smi`, executes an FP16 `4096 x 4096` GEMM across all attached GPUs, and writes `gpu_smoke_report.json`.
- **[`examples/02_qlora_finetune`](examples/02_qlora_finetune):** Quantizes base transformer linear layers into packed 4-bit `uint8` buffers with per-block FP16 scales, attaches trainable FP32/FP16 LoRA adapters (`lora_A`, `lora_B`), runs 60 optimization steps with gradient clipping, exports only the compact adapter (`lora_adapter.pt`) and `qlora_metrics.json`, and cleans up `/tmp`.
- **[`examples/03_inference_bench`](examples/03_inference_bench):** Benchmarks autoregressive transformer token generation across Tesla T4 (`float16`) and CPU (`float32`), and provides `setup_llamacpp_cuda()` for downloading and configuring both the `llama.cpp` CUDA binary tarball and its required `cudart-` companion runtime archive.
- **[`examples/04_data_processing`](examples/04_data_processing):** Executes L2 normalization, 256M-pair exact cosine similarity top-k retrieval, and 32-component low-rank SVD (`torch.pca_lowrank`) over 32,000 x 768 document embeddings on Tesla T4 vs CPU.

---

## Repository Structure

```text
kaggle-free-gpu/
├── SKILL.md                        # Agent skill specification, workflow, and operational rules
├── pyproject.toml                  # Hatchling build config, dependencies, and entrypoints
├── src/
│   └── kaggle_free_gpu/
│       ├── __init__.py
│       ├── client.py               # Direct REST & KernelsApiService RPC client
│       ├── models.py               # Pydantic schemas and pre-push metadata validator
│       ├── quota.py                # Local rolling 7-day GPU quota ledger
│       ├── server.py               # FastMCP server exposing the 6 Kaggle GPU tools
│       └── cli.py                  # Terminal CLI (`kaggle-free-gpu` / `kfg`)
├── examples/
│   ├── 01_gpu_smoke_test/
│   ├── 02_qlora_finetune/
│   ├── 03_inference_bench/
│   └── 04_data_processing/
├── scripts/
│   └── validate_live.py            # End-to-end live validation suite against Kaggle REST API
└── tests/
    └── test_client_and_rules.py    # Unit tests for slug rules, quota math, and safety guards
```

---

## Development & Verification

```bash
# Run unit tests
uv run pytest -v

# Run linter
uv run ruff check src tests scripts examples

# Run live end-to-end GPU validation against Kaggle REST API (requires KAGGLE_API_TOKEN)
uv run python scripts/validate_live.py
```

---

## License

MIT License. See [LICENSE](LICENSE).
