"""FastMCP Model Context Protocol (MCP) Server for Kaggle Free-GPU."""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from kaggle_free_gpu.client import KaggleGPUClient

logger = logging.getLogger("kaggle_free_gpu.server")

mcp = FastMCP("kaggle-free-gpu")


def get_client() -> KaggleGPUClient:
    """Create a KaggleGPUClient instance using configured environment or ~/.kaggle credentials."""
    return KaggleGPUClient()


@mcp.tool()
def push_kernel(
    dir: str,
    allow_mismatch: bool = False,
) -> str:
    """Validate kernel-metadata.json + code file in `dir` and push the kernel to Kaggle's Tesla T4 x2 GPUs.

    Validates id/title/slug consistency (Gotcha #1: Kaggle derives the URL slug from the TITLE,
    not metadata `id`), checks that the code file exists and is non-empty, sends a JSON push
    request (Gotcha #2), reports any dropped `invalidTags` (Gotcha #9), and records the run
    in the local rolling 7-day GPU quota ledger (Gotcha #8). Refuses to push if validation fails.

    Args:
        dir: Path to folder containing `kernel-metadata.json` and the script/notebook file.
        allow_mismatch: Only set True when intentionally testing Gotcha #1 slug mismatch behavior.

    Returns:
        JSON string with `ok`, `ref`, `url`, `version_number`, `invalid_tags`, `warnings`, or `error`.
    """
    client = get_client()
    result = client.push_kernel(dir_path=dir, allow_mismatch=allow_mismatch)
    return json.dumps(result.model_dump(), indent=2)


@mcp.tool()
def kernel_status(owner_slug: str) -> str:
    """Check the execution status of a Kaggle kernel (`QUEUED`, `RUNNING`, `COMPLETE`, or `ERROR`).

    Uses `POST /api/v1/kernels.KernelsApiService/GetKernelSessionStatus` (Gotcha #4).
    Note: `owner_slug` MUST use the title-derived slug (`owner/title-slug`), otherwise Kaggle
    returns HTTP 403 (Gotcha #1).

    Args:
        owner_slug: Kernel reference in `<owner>/<slug>` format.

    Returns:
        JSON string with `ref`, `status`, `failure_message`, and optional `hint`.
    """
    client = get_client()
    result = client.kernel_status(owner_slug=owner_slug)
    return json.dumps(result.model_dump(), indent=2)


@mcp.tool()
def kernel_output(
    owner_slug: str,
    out_dir: str,
    max_size_mb: int = 500,
    force_large: bool = False,
) -> str:
    """Download and extract kernel output files and execution logs into `out_dir`.

    Uses `POST /api/v1/kernels.KernelsApiService/ListKernelSessionOutput` (Gotcha #4).
    Checks `Content-Length` on every file in `/kaggle/working` before downloading (Gotcha #5)
    and skips/warns if any file exceeds `max_size_mb` unless `force_large=True`.
    Also writes `kernel.log` and updates the local GPU quota ledger with exact measured runtime.

    Args:
        owner_slug: Kernel reference in `<owner>/<slug>` format.
        out_dir: Local directory path where output files and `kernel.log` will be saved.
        max_size_mb: Maximum per-file size in MB before warning/skipping (default 500 MB).
        force_large: Set True to download files larger than `max_size_mb`.

    Returns:
        JSON string with downloaded `files`, `runtime_seconds`, `gpu_hours_recorded`, `stdout_tail`, `stderr_tail`, and `warnings`.
    """
    client = get_client()
    result = client.kernel_output(
        owner_slug=owner_slug,
        out_dir=out_dir,
        max_size_mb=max_size_mb,
        force_large=force_large,
    )
    return json.dumps(result.model_dump(), indent=2)


@mcp.tool()
def list_kernels(
    mine: bool = True,
    user: str | None = None,
    search: str | None = None,
    page_size: int = 20,
) -> str:
    """List Kaggle kernels (your own private/public kernels by default, or filter by user/search).

    Uses `GET /api/v1/kernels/list` with `group=profile` when `mine=True`.

    Args:
        mine: If True (default), list the authenticated user's own kernels.
        user: Optional Kaggle username to filter by.
        search: Optional search keyword query.
        page_size: Maximum number of results to return (default 20).

    Returns:
        JSON string containing a list of kernel summaries (`ref`, `title`, `last_run_time`, etc.).
    """
    client = get_client()
    items = client.list_kernels(mine=mine, user=user, search=search, page_size=page_size)
    return json.dumps(items, indent=2)


@mcp.tool()
def delete_kernel(
    owner_slug: str,
    confirm: bool = False,
) -> str:
    """Permanently delete a Kaggle kernel (`POST /api/v1/kernels.KernelsApiService/DeleteKernel`).

    SAFETY RULE: Deletion is permanent and unrecoverable. You MUST confirm with the user
    before calling this tool with `confirm=True`. Calling with `confirm=False` returns a
    confirmation prompt without deleting.

    Args:
        owner_slug: Kernel reference in `<owner>/<slug>` format.
        confirm: Must be set to True ONLY after the user has explicitly confirmed deletion.

    Returns:
        JSON string confirming deletion or requesting user confirmation.
    """
    client = get_client()
    result = client.delete_kernel(owner_slug=owner_slug, confirm=confirm)
    return json.dumps(result, indent=2)


@mcp.tool()
def quota_estimate() -> str:
    """Estimate remaining weekly Kaggle GPU quota from local push history (~30 GPU-hours/week rolling window).

    Tesla T4 x2 consumes 2 GPU-hours per wall-clock hour (~15 wall-hours/week on dual T4s).
    Because Kaggle provides no live API endpoint for remaining quota, always remind the user
    to check Settings -> GPU Quota at https://www.kaggle.com/settings for the exact number.

    Returns:
        JSON string with rolling 7-day GPU-hours used, remaining GPU-hours, remaining T4 x2 wall-hours, and recent runs.
    """
    client = get_client()
    return json.dumps(client.quota_estimate(), indent=2)


def main() -> None:
    """Entrypoint for the kaggle-free-gpu-mcp server."""
    mcp.run()


if __name__ == "__main__":
    main()

