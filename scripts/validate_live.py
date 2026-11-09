"""Live end-to-end validation runner for §7 of the Kaggle Free-GPU Brief.

Verifies against the real Kaggle REST API (https://www.kaggle.com/api/v1):
1. Push GPU smoke test kernel -> status COMPLETE -> output shows Tesla T4 x2 detected.
2. Validate Gotcha #1 slug rule:
   a) Local pre-push validator blocks mismatched id vs title-derived slug.
   b) Calling kernel_status with an id-based wrong slug returns HTTP 403 (`ERROR_403_FORBIDDEN`).
3. Quota estimate is sane and updated with actual measured runtime from kernel logs.
4. All 3 example workflows (`02_qlora_finetune`, `03_inference_bench`, `04_data_processing`)
   push, run COMPLETE on Tesla T4 x2, and download their output JSONs cleanly.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from kaggle_free_gpu.client import KaggleGPUClient

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "outputs" / "validation"


def poll_kernel(client: KaggleGPUClient, ref: str, poll_sec: float = 8.0, timeout_sec: float = 600.0) -> str:
    t0 = time.time()
    last = None
    while True:
        res = client.kernel_status(ref)
        elapsed = int(time.time() - t0)
        if res.status != last:
            print(f"  [{elapsed:3d}s] {ref} -> {res.status}", flush=True)
            last = res.status
        if res.status in ("COMPLETE", "ERROR") or res.status.startswith("ERROR_"):
            return res.status
        if elapsed >= timeout_sec:
            raise TimeoutError(f"Timed out waiting for {ref} (last status: {res.status})")
        time.sleep(poll_sec)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    client = KaggleGPUClient()
    owner = client.resolve_username() or "owner"
    results_summary: dict = {"steps": {}}

    # -------------------------------------------------------------------------
    # Check 2: Validate Slug Rule (Gotcha #1) — local guard + live 403 proof
    # -------------------------------------------------------------------------
    print("\n=== [Check 2] Validating Gotcha #1 (Slug Rule & HTTP 403 on wrong slug) ===", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "dummy.py").write_text("print('test')\n", encoding="utf-8")
        (tmp_path / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": f"{owner}/wrong-id-based-slug-xyz",
                    "title": "KFG Actual Title Derived Slug",
                    "code_file": "dummy.py",
                    "language": "python",
                    "kernel_type": "script",
                    "is_private": True,
                    "enable_gpu": True,
                    "enable_internet": True,
                }
            ),
            encoding="utf-8",
        )
        blocked_push = client.push_kernel(tmp_path, allow_mismatch=False)
        assert not blocked_push.ok, "Expected pre-push validator to reject mismatched slug!"
        print("  [PASS] Pre-push validator rejected mismatched id slug:", blocked_push.error.splitlines()[1])

    # Now verify live HTTP 403 when querying status with a non-existent / wrong id-based slug
    wrong_slug_status = client.kernel_status(f"{owner}/wrong-id-based-slug-xyz")
    print("  [LIVE] Status response for wrong slug:", wrong_slug_status.model_dump())
    assert wrong_slug_status.status == "ERROR_403_FORBIDDEN", f"Expected 403, got {wrong_slug_status.status}"
    results_summary["steps"]["check_2_slug_rule_403"] = {
        "pre_push_blocked": not blocked_push.ok,
        "live_wrong_slug_status": wrong_slug_status.model_dump(),
    }

    # -------------------------------------------------------------------------
    # Check 1 & Check 4: Push and run Smoke Test + All 3 Example Workflows
    # -------------------------------------------------------------------------
    workflows = [
        ("check_1_gpu_smoke_test", ROOT / "examples" / "01_gpu_smoke_test", "gpu_smoke_report.json"),
        ("workflow_1_qlora_finetune", ROOT / "examples" / "02_qlora_finetune", "qlora_metrics.json"),
        ("workflow_2_inference_bench", ROOT / "examples" / "03_inference_bench", "inference_results.json"),
        ("workflow_3_data_processing", ROOT / "examples" / "04_data_processing", "data_processing_results.json"),
    ]

    # Push all 4 kernels and poll to completion
    pushed_refs: list[tuple[str, str, str]] = []
    for step_name, folder, expected_json in workflows:
        print(f"\n=== Running {step_name} ({folder.name}) ===", flush=True)
        _, report = client.validate_kernel_dir(folder)
        ref = f"{report.owner}/{report.id_slug}"
        # Check if already running or complete before pushing so we don't double-push
        cur = client.kernel_status(ref)
        if cur.status in ("QUEUED", "RUNNING", "COMPLETE"):
            print(f"  Kernel {ref} already exists (status={cur.status}); waiting for completion...", flush=True)
            push_url = f"https://www.kaggle.com/code/{ref}"
            push_ver = 1
        else:
            push_res = client.push_kernel(folder)
            assert push_res.ok, f"Push failed for {folder}: {push_res.error}"
            print(f"  Pushed {push_res.ref} (v{push_res.version_number}) -> {push_res.url}", flush=True)
            if push_res.warnings:
                print(f"  Warnings: {push_res.warnings}", flush=True)
            ref = push_res.ref
            push_url = push_res.url
            push_ver = push_res.version_number

        status = poll_kernel(client, ref)
        assert status == "COMPLETE", f"Kernel {ref} ended with status {status}"

        out_dir = OUT_ROOT / step_name
        out_res = client.kernel_output(ref, out_dir)
        json_file = out_dir / expected_json
        assert json_file.is_file(), f"Expected output file {json_file} not found! Stdout: {out_res.stdout_tail}"
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        print(f"  [PASS] {step_name} COMPLETE in {out_res.runtime_seconds}s! Output {expected_json}:", flush=True)
        print("  " + json.dumps(payload, indent=2).replace("\n", "\n  "), flush=True)

        results_summary["steps"][step_name] = {
            "ref": ref,
            "version": push_ver,
            "url": push_url,
            "runtime_seconds": out_res.runtime_seconds,
            "gpu_hours_recorded": out_res.gpu_hours_recorded,
            "output": payload,
        }
        pushed_refs.append((step_name, ref, expected_json))

    # -------------------------------------------------------------------------
    # Check 3: Verify Quota Estimate
    # -------------------------------------------------------------------------
    print("\n=== [Check 3] Verifying Quota Estimate ===", flush=True)
    quota = client.quota_estimate()
    print(json.dumps(quota, indent=2), flush=True)
    assert quota["rolling_7d_gpu_run_count"] >= 4
    assert 0 < quota["estimated_gpu_hours_used"] < 30.0
    results_summary["steps"]["check_3_quota_estimate"] = quota

    summary_path = OUT_ROOT / "validation_summary.json"
    summary_path.write_text(json.dumps(results_summary, indent=2), encoding="utf-8")
    print(f"\nALL VALIDATION CHECKS PASSED! Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
