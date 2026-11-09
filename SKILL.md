---
name: kaggle-free-gpu
description: Run GPU workloads (QLoRA fine-tuning, LLM/GGUF inference benchmarks, batch embeddings, CUDA verification) at zero cost on Kaggle's free Tesla T4 x2 GPUs (~30 GB total VRAM) using the kaggle-free-gpu MCP server or CLI.
---

# Kaggle Free-GPU Skill (`kaggle-free-gpu`)

Provide coding agents with direct access to **Kaggle's free NVIDIA Tesla T4 x2 GPUs** (2 x 15,360 MiB VRAM, Python 3.13, PyTorch 2.11+cu130, internet enabled) via `https://www.kaggle.com/api/v1`.

---

## 1. Workload Selection: Tesla T4 x2 GPU vs. CPU

| Workload | Use Kaggle Free-GPU (`enable_gpu: true`) | Use Local / Kaggle CPU (`enable_gpu: false`) |
| :--- | :--- | :--- |
| **Fine-Tuning (QLoRA / LoRA / PEFT)** | **Always.** Requires 10-30 GB VRAM and FP16/INT4 tensor core execution. | Never -- orders of magnitude slower or out-of-memory. |
| **LLM / GGUF / Transformer Inference Benchmarks** | **Always.** T4 x2 executes 7B-14B Q4_K_M models at ~35-45 tok/s vs ~2-3 tok/s on CPU. | Only for trivial (<500M parameter) smoke checks. |
| **Batch Embeddings & Matrix Operations** | **Yes** when encoding >1,000 texts/images or running high-dimensional similarity/SVD. | Small CSV/JSON preprocessing (<100 MB), regex, ETL. |
| **CUDA Kernel / Multi-GPU Verification** | **Always.** Kaggle provisions dual Tesla T4 accelerators (`cuda:0`, `cuda:1`). | Unit tests that mock `torch.cuda`. |

### Quota Accounting
- **Weekly Budget:** ~30 GPU-hours per rolling 7-day window (hours age out 7 days after consumption; does not reset on a fixed weekday).
- **Dual-GPU Burn Rate:** Kaggle GPU sessions attach **Tesla T4 x2**, which consumes **2 GPU-hours per wall-clock hour** (~15 wall-hours/week on dual T4s).
- **Single Session Limit:** 12 hours maximum runtime per kernel execution.
- **CPU Sessions:** Unlimited weekly hours (`enable_gpu: false`).
- **Zero Financial Risk:** When quota is exhausted, nothing is billed; new GPU sessions are refused until rolling usage ages out.

---

## 2. Standard Agent Execution Workflow

Execute the following 6-step loop for every GPU task:

1. **Check Quota (`quota_estimate`)**
   - Call `quota_estimate()` (or `kfg quota`) prior to pushing.
   - Because Kaggle provides **no live API endpoint for remaining GPU quota**, report the local estimate from `~/.kaggle-free-gpu/usage.json` and remind the user that the authoritative balance is under **Settings -> GPU Quota** (`https://www.kaggle.com/settings`).
2. **Prepare `kernel-metadata.json` and Script in a Dedicated Directory**
   - Set `"is_private": true`, `"enable_gpu": true`, `"enable_internet": true`.
   - Ensure `"id"` (`"<owner>/<slug>"`) matches the **title-derived slug** exactly (Rule 1).
   - Ensure the script writes compact outputs (`.json`, `.csv`, LoRA adapters) to `/kaggle/working` and **deletes all multi-hundred-MB or multi-GB intermediate files before exiting** (Rule 5).
3. **Push Kernel (`push_kernel`)**
   - Call `push_kernel(dir)` (or `kfg push <dir>`).
   - If pre-push validation fails, fix the reported metadata/slug error before retrying.
   - Inspect `invalid_tags` in the response and remove any rejected tags (Rule 9).
4. **Poll Status (`kernel_status`)**
   - Poll `kernel_status("<owner>/<title-derived-slug>")` every 10-20 seconds until status reaches `COMPLETE` or `ERROR`.
5. **Retrieve Output (`kernel_output`)**
   - Call `kernel_output("<owner>/<title-derived-slug>", out_dir)` to download output artifacts and `kernel.log` (`stdout`/`stderr`).
   - This automatically updates the local GPU quota ledger with the exact runtime recorded in Kaggle's session log.
6. **Report Results & Clean Up**
   - Present the GPU metrics and measured runtime to the user.
   - **Retain kernels by default:** Only invoke `delete_kernel` if the user explicitly requests deletion, and **always obtain explicit user confirmation first** because kernel deletion is permanent.

---

## 3. Mandatory Operational Rules (Verified API Gotchas)

Every rule below was verified live against `https://www.kaggle.com/api/v1` on Tesla T4 x2 kernels:

### Rule 1: Slug Derivation -- `id` Must Match the Title-Derived Slug
- **Kaggle derives the kernel URL slug from `title`, NOT from the metadata `id`.**
- Calling `kernel_status` or `kernel_output` with an `id`-based slug that differs from the title-derived slug returns **HTTP 403 Forbidden**.
- Pushing an update with an existing title when `id` does not match that kernel's existing title-derived slug is treated as a *create* operation and fails with **HTTP 409 Conflict**.
- **Requirement:** If `title` is `"KFG QLoRA Finetune Demo"`, `id` **must** be `"<owner>/kfg-qlora-finetune-demo"`.

### Rule 2: Push Requests Require a JSON Body (`Content-Type: application/json`)
- Sending `multipart/form-data` to `/api/v1/kernels/push` fails with **HTTP 400 (`JToken` error)**.
- Always send a JSON body with `Content-Type: application/json` via `push_kernel`.

### Rule 3: Do Not Use the Official Kaggle CLI 1.8 Host (`api.kaggle.com`)
- Kaggle CLI 1.8 targets `api.kaggle.com`, which is invalid for these endpoints.
- Always target `https://www.kaggle.com/api/v1` via `kaggle-free-gpu` MCP tools or the `kfg` CLI.

### Rule 4: Use `KernelsApiService` RPC Endpoints for Status, Output, and Delete
- Plain REST paths `/kernels/status/{owner}/{slug}` and `/kernels/output/{owner}/{slug}` **do not exist**.
- Status, output, and delete operations must use:
  - `POST /api/v1/kernels.KernelsApiService/GetKernelSessionStatus`
  - `POST /api/v1/kernels.KernelsApiService/ListKernelSessionOutput`
  - `POST /api/v1/kernels.KernelsApiService/DeleteKernel`
- Listing the authenticated user's own kernels via `GET /api/v1/kernels/list` requires `group=profile` (`mine=true` and `group=my` return HTTP 400).

### Rule 5: Delete Large Intermediates Before Script Termination
- `kernel_output` enumerates every file remaining in `/kaggle/working`. Leaving a 5.6 GB `.gguf` or base checkpoint in `/kaggle/working` causes output downloads to fail (`IncompleteRead`).
- **Requirement:** Store transient downloads in `/tmp`, or explicitly remove all large weights, tarballs, and unpacked trees in a `try...finally` block prior to script exit.

### Rule 6: `llama.cpp` CUDA on Kaggle Requires BOTH Release Tarballs and `LD_LIBRARY_PATH`
- Downloading only `llama-<tag>-bin-ubuntu-cuda-12.x-x64.tar.gz` is insufficient. The CUDA runtime libraries (`libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`) are packaged separately in `cudart-llama-<tag>-bin-ubuntu-cuda-12.x-x64.tar.gz`. Without extracting both archives into the same directory and prepending that path to `LD_LIBRARY_PATH`, the CUDA backend fails to register and `llama.cpp` runs on CPU at ~2.7 tok/s with no error instead of ~37.7 tok/s on Tesla T4.
- **Requirement:** Download and extract both the `llama-...` binary tarball and the `cudart-llama-...` companion tarball, configure `LD_LIBRARY_PATH`, pass `-ngl 99`, and verify `CUDA0`/`CUDA1` buffer allocation in the logs.

### Rule 7: Keep Kernels Private by Default
- Always set `"is_private": true` in `kernel-metadata.json` unless the user explicitly instructs you to publish the kernel.

### Rule 8: Every Push Initiates a New Execution and Consumes Quota
- Pushing an update to an existing kernel creates a new version and immediately starts a GPU session. Validate code locally before pushing.

### Rule 9: Only Use Valid Kaggle Taxonomy Tags (`keywords` / `categoryIds`)
- Kaggle validates tags against its internal taxonomy and drops unrecognized tags in the push response under `invalidTags`.
- Verified valid tags: `["gpu", "nlp"]` (tags such as `"deep learning"`, `"pytorch"`, and `"text generation"` are rejected in `/kernels/push` and returned in `invalidTags`). Always inspect `invalid_tags` in the `push_kernel` result.

---

## 4. Safety Rules

1. **Free Tier Only:** Never provision paid cloud instances or remote-control tunnel workarounds.
2. **Credential Hygiene:** Never hardcode, print, log, or commit `KAGGLE_API_TOKEN` or `KAGGLE_KEY`. Load credentials exclusively via environment variables (`KAGGLE_API_TOKEN`) or `~/.kaggle/access_token` / `~/.kaggle/kaggle.json`.
3. **Permanent Deletion Requires User Confirmation:** Never call `delete_kernel(..., confirm=True)` without first asking the user and receiving explicit confirmation.

---

## 5. Reference Workflows

### Workflow 1: QLoRA / LoRA Fine-Tuning (`examples/02_qlora_finetune`)

**`kernel-metadata.json`**
```json
{
  "id": "<owner>/kfg-qlora-finetune-demo",
  "title": "KFG QLoRA Finetune Demo",
  "code_file": "qlora_finetune.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "keywords": ["gpu", "nlp"]
}
```

**`qlora_finetune.py` (skeleton)**
```python
import json, os, shutil
from pathlib import Path
import torch

TMP_DIR = Path("/tmp/hf_cache")
TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(TMP_DIR)

try:
    assert torch.cuda.is_available(), "CUDA GPU not available"
    # Run 4-bit base + trainable LoRA adapter fine-tuning on cuda:0
    # Save only compact adapter state_dict and metrics JSON to /kaggle/working
    Path("qlora_metrics.json").write_text(json.dumps({
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
    }, indent=2))
finally:
    shutil.rmtree(TMP_DIR, ignore_errors=True)
```

---

### Workflow 2: `llama.cpp` + PyTorch GPU Inference Benchmark (`examples/03_inference_bench`)

**`kernel-metadata.json`**
```json
{
  "id": "<owner>/kfg-inference-bench-demo",
  "title": "KFG Inference Bench Demo",
  "code_file": "inference_bench.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "keywords": ["gpu", "nlp"]
}
```

**`inference_bench.py` (Rule 5 & Rule 6 helper for `llama.cpp` CUDA)**
```python
import json, os, subprocess, tarfile, urllib.request

PREFERRED_ASSET = "llama-b12211-bin-ubuntu-cuda-12.8-x64.tar.gz"
NEEDED_LIBS = ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12")

def setup_llamacpp_cuda(work_dir: str = "/tmp/llama.cpp") -> tuple[str, dict[str, str]]:
    """Download both the plain CUDA binary tarball and the cudart- companion tarball,
    extract both into work_dir, verify libcudart/libcublas exist, and set LD_LIBRARY_PATH.
    """
    os.makedirs(work_dir, exist_ok=True)
    req = urllib.request.Request(
        "https://api.github.com/repos/ggerganov/llama.cpp/releases?per_page=10",
        headers={"User-Agent": "kaggle-free-gpu", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        releases = json.load(resp)

    asset_name = asset_url = cudart_name = cudart_url = None
    for rel in releases:
        cands = [
            a["name"] for a in rel.get("assets", [])
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

    assert asset_url and cudart_url, f"Missing CUDA or cudart companion asset ({asset_name} / {cudart_name})"
    for fname, furl in ((asset_name, asset_url), (cudart_name, cudart_url)):
        subprocess.run(["curl", "-sS", "-L", "-C", "-", "-o", fname, furl], check=True)
        with tarfile.open(fname, "r:gz") as tar:
            tar.extractall(path=work_dir, filter="data")
        os.remove(fname)

    lib_dirs: set[str] = set()
    llama_server = None
    for root, _dirs, files in os.walk(work_dir):
        for f in files:
            if f in NEEDED_LIBS or f.endswith(".so") or ".so." in f:
                lib_dirs.add(os.path.abspath(root))
            if f == "llama-server":
                llama_server = os.path.abspath(os.path.join(root, f))
                os.chmod(llama_server, 0o755)

    assert llama_server, "llama-server binary not found in extracted archive"
    env = os.environ.copy()
    ld_parts = sorted(lib_dirs) + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    return llama_server, env
```

---

### Workflow 3: GPU Batch Data Processing (`examples/04_data_processing`)

**`kernel-metadata.json`**
```json
{
  "id": "<owner>/kfg-data-processing-demo",
  "title": "KFG Data Processing Demo",
  "code_file": "data_processing.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "keywords": ["gpu", "nlp"]
}
```

**`data_processing.py` (skeleton)**
```python
import json
from pathlib import Path
import torch
import torch.nn.functional as F

assert torch.cuda.is_available(), "Expected Tesla T4 GPU"
device = torch.device("cuda:0")
embeddings = F.normalize(torch.randn((32000, 768), device=device), p=2, dim=-1)
sim_matrix = torch.matmul(embeddings[:8000], embeddings.T)
top_vals, top_idx = torch.topk(sim_matrix, k=5, dim=-1)
u, s, v = torch.pca_lowrank(embeddings, q=32, center=True)
Path("data_processing_results.json").write_text(json.dumps({"top_singular_values": s[:5].tolist()}, indent=2))
```
