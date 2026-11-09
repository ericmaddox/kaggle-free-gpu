"""Example Workflow 3: GPU-Accelerated Batch Data Processing, Exact k-NN Cosine Search & SVD on T4 x2.

Demonstrates processing 40,000 768-dim document embeddings (all-pairs cosine similarity top-k
retrieval + low-rank SVD projection) across Kaggle's Tesla T4 GPU vs CPU, then saving only
the compact results JSON and cleaning up raw tensors (Rule #5).
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def run_pipeline(device: torch.device, n_docs: int = 32000, n_queries: int = 8000, dim: int = 768, top_k: int = 5) -> dict:
    torch.manual_seed(42)
    embeddings = torch.randn((n_docs, dim), device=device, dtype=torch.float32)

    # Warmup CUDA context on GPU so we measure steady-state pipeline throughput
    if device.type == "cuda":
        _ = torch.matmul(embeddings[:64], embeddings[:64].T)
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()

    # 1. Normalize 32,000 x 768 embeddings
    normed = F.normalize(embeddings, p=2, dim=-1)

    # 2. Compute exact cosine similarity for 8,000 query vectors against all 32,000 docs (256M pairs)
    queries = normed[:n_queries]
    sim_matrix = torch.matmul(queries, normed.T)
    top_vals, top_idx = torch.topk(sim_matrix, k=top_k, dim=-1)

    # 3. Compute truncated SVD / PCA projection to 32 dimensions
    u, s, v = torch.pca_lowrank(normed, q=32, center=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    sample_matches = [
        {"query_id": i, "top_doc_ids": top_idx[i].tolist(), "top_scores": [round(float(x), 4) for x in top_vals[i].tolist()]}
        for i in range(3)
    ]
    top_singular_values = [round(float(x), 3) for x in s[:5].tolist()]

    del embeddings, normed, queries, sim_matrix, u, s, v
    return {
        "device": str(device),
        "n_docs": n_docs,
        "embedding_dim": dim,
        "elapsed_seconds": round(elapsed, 4),
        "docs_per_second": round(n_docs / elapsed, 1),
        "top_5_singular_values": top_singular_values,
        "sample_top_k_matches": sample_matches,
    }


def main() -> None:
    assert torch.cuda.is_available(), "CUDA GPU required!"
    gpu_metrics = run_pipeline(torch.device("cuda:0"))
    cpu_metrics = run_pipeline(torch.device("cpu"))

    speedup = round(cpu_metrics["elapsed_seconds"] / max(gpu_metrics["elapsed_seconds"], 1e-6), 2)
    summary = {
        "workflow": "data_processing",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "gpu_pipeline": gpu_metrics,
        "cpu_pipeline": cpu_metrics,
        "gpu_vs_cpu_speedup_x": speedup,
    }
    Path("data_processing_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("CLEANUP DONE")


if __name__ == "__main__":
    main()

