"""Command-line interface for Kaggle Free-GPU (`kaggle-free-gpu` / `kfg`)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.table import Table

from kaggle_free_gpu.client import KaggleAPIError, KaggleAuthError, KaggleGPUClient

console = Console()
_ANSI_CTRL_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _safe_text(val: object) -> str:
    """Strip ANSI/OSC terminal escape codes and escape Rich markup tags from untrusted text."""
    cleaned = _ANSI_CTRL_RE.sub("", str(val if val is not None else ""))
    return rich_escape(cleaned)


def _poll_until_done(client: KaggleGPUClient, ref: str, poll_interval: float = 10.0, timeout_sec: float = 3600.0) -> str:
    """Poll kernel_status until COMPLETE or ERROR."""
    start = time.time()
    last_status = ""
    while True:
        res = client.kernel_status(ref)
        status = res.status
        elapsed = int(time.time() - start)
        if status != last_status:
            console.print(f"[cyan][{elapsed}s][/cyan] {_safe_text(ref)} -> [bold]{_safe_text(status)}[/bold]")
            last_status = status
        if status in ("COMPLETE", "ERROR") or status.startswith("ERROR_"):
            if res.failure_message:
                console.print(f"[red]Failure:[/red] {_safe_text(res.failure_message)}")
            if res.hint:
                console.print(f"[yellow]Hint:[/yellow] {_safe_text(res.hint)}")
            return status
        if elapsed >= timeout_sec:
            console.print(f"[yellow]Timeout reached ({timeout_sec}s) while status is {_safe_text(status)}.[/yellow]")
            return status
        time.sleep(poll_interval)


def cmd_push(args: argparse.Namespace) -> int:
    client = KaggleGPUClient()
    res = client.push_kernel(dir_path=args.dir, allow_mismatch=args.allow_mismatch)
    if args.json:
        print(json.dumps(res.model_dump(), indent=2))
        return 0 if res.ok else 1

    for w in res.warnings:
        console.print(f"[yellow]Warning:[/yellow] {_safe_text(w)}")

    if not res.ok:
        console.print(Panel(_safe_text(res.error or "Push failed"), title="[red]Push Rejected[/red]", border_style="red"))
        return 1

    console.print(
        Panel(
            f"[bold green]Pushed successfully![/bold green]\n"
            f"Ref: [cyan]{_safe_text(res.ref)}[/cyan]\n"
            f"Title: {_safe_text(res.title)}\n"
            f"Version: {_safe_text(res.version_number)}\n"
            f"GPU Enabled: {res.enable_gpu} | Private: {res.is_private}\n"
            f"URL: {_safe_text(res.url)}",
            title="Kaggle Free-GPU Push",
            border_style="green",
        )
    )

    if args.wait:
        final_status = _poll_until_done(client, res.ref, poll_interval=args.poll_interval)
        if final_status != "COMPLETE":
            return 1
        if args.out_dir:
            out_res = client.kernel_output(res.ref, args.out_dir)
            console.print(f"[green]Output downloaded to:[/green] {_safe_text(out_res.out_dir)} (runtime: {out_res.runtime_seconds}s)")
            if out_res.stdout_tail:
                console.print(Panel(_safe_text(out_res.stdout_tail.strip()), title="Stdout", border_style="blue"))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    client = KaggleGPUClient()
    if args.wait:
        final_status = _poll_until_done(client, args.ref, poll_interval=args.poll_interval)
        return 0 if final_status == "COMPLETE" else 1

    res = client.kernel_status(args.ref)
    if args.json:
        print(json.dumps(res.model_dump(), indent=2))
        return 0 if not res.status.startswith("ERROR") else 1

    color = "green" if res.status == "COMPLETE" else ("red" if "ERROR" in res.status else "yellow")
    console.print(f"[bold]{_safe_text(res.ref)}[/bold]: [{color}]{_safe_text(res.status)}[/{color}]")
    if res.failure_message:
        console.print(f"[red]Failure:[/red] {_safe_text(res.failure_message)}")
    if res.hint:
        console.print(f"[yellow]Hint:[/yellow] {_safe_text(res.hint)}")
    return 0 if not res.status.startswith("ERROR") else 1


def cmd_output(args: argparse.Namespace) -> int:
    client = KaggleGPUClient()
    res = client.kernel_output(
        owner_slug=args.ref,
        out_dir=args.out_dir,
        max_size_mb=args.max_size_mb,
        force_large=args.force_large,
    )
    if args.json:
        print(json.dumps(res.model_dump(), indent=2))
        return 0

    for w in res.warnings:
        console.print(f"[yellow]Warning:[/yellow] {_safe_text(w)}")
    if res.hint:
        console.print(f"[yellow]Hint:[/yellow] {_safe_text(res.hint)}")

    console.print(
        f"[bold green]Pulled output for {_safe_text(res.ref)}[/bold green] -> [cyan]{_safe_text(res.out_dir)}[/cyan] "
        f"(runtime: {res.runtime_seconds}s, GPU-hrs: {res.gpu_hours_recorded})"
    )
    if res.files:
        table = Table(title="Working Directory Output Files")
        table.add_column("File", style="cyan")
        table.add_column("Size (bytes)", justify="right")
        table.add_column("Downloaded")
        for f in res.files:
            table.add_row(
                _safe_text(f.file_name),
                str(f.size_bytes) if f.size_bytes is not None else "?",
                "[green]Yes[/green]" if f.downloaded else f"[red]Skipped[/red] ({_safe_text(f.skipped_reason)})",
            )
        console.print(table)

    if res.stdout_tail:
        console.print(Panel(_safe_text(res.stdout_tail.strip()), title="Stdout (Tail)", border_style="blue"))
    if res.stderr_tail.strip():
        console.print(Panel(_safe_text(res.stderr_tail.strip()), title="Stderr (Tail)", border_style="yellow"))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    client = KaggleGPUClient()
    mine = not (args.user or args.search) or args.mine
    items = client.list_kernels(
        mine=mine,
        user=args.user,
        search=args.search,
        page_size=args.page_size,
    )
    if args.json:
        print(json.dumps(items, indent=2))
        return 0

    table = Table(title="Kaggle Kernels")
    table.add_column("Ref (owner/slug)", style="cyan")
    table.add_column("Title")
    table.add_column("Author")
    table.add_column("Last Run (UTC)")
    for item in items:
        table.add_row(
            _safe_text(item["ref"]),
            _safe_text(item["title"]),
            _safe_text(item["author"]),
            _safe_text(item["last_run_time"]),
        )
    console.print(table)
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    client = KaggleGPUClient()
    confirmed = args.yes
    if not confirmed:
        console.print(
            f"[bold red]WARNING:[/bold red] Deleting kernel [cyan]{_safe_text(args.ref)}[/cyan] is "
            "[bold]PERMANENT and UNRECOVERABLE[/bold]."
        )
        ans = input(f"Type '{args.ref}' to confirm permanent deletion: ").strip()
        if ans != args.ref:
            console.print("[yellow]Deletion aborted.[/yellow]")
            return 1
        confirmed = True

    res = client.delete_kernel(args.ref, confirm=confirmed)
    if args.json:
        print(json.dumps(res, indent=2))
        return 0 if res.get("deleted") else 1

    if res.get("deleted"):
        console.print(f"[green]Deleted kernel {_safe_text(args.ref)}.[/green]")
        return 0
    console.print(f"[red]{_safe_text(res.get('message'))}[/red]")
    return 1


def cmd_quota(args: argparse.Namespace) -> int:
    client = KaggleGPUClient()
    if args.sync_used_gpu_hours is not None:
        client.quota_ledger.record_manual_adjustment(
            gpu_hours_used=args.sync_used_gpu_hours,
            note="CLI manual quota sync",
        )
    est = client.quota_estimate()
    if args.json:
        print(json.dumps(est, indent=2))
        return 0

    console.print(
        Panel(
            f"Weekly Budget: [bold]{est['weekly_budget_gpu_hours']} GPU-hours[/bold] (rolling 7-day window)\n"
            f"T4 x2 Multiplier: [bold]{est['t4_x2_multiplier']}x[/bold] (2 GPU-hours per wall-clock hour)\n"
            f"Estimated Used (7d): [yellow]{est['estimated_gpu_hours_used']} GPU-hours[/yellow] "
            f"({est['estimated_wall_hours_used_t4_x2']} wall-hours on T4 x2)\n"
            f"Estimated Remaining: [green]{est['estimated_gpu_hours_remaining']} GPU-hours[/green] "
            f"(~{est['estimated_wall_hours_remaining_t4_x2']} wall-hours on T4 x2)\n"
            f"Single Session Cap: {est['single_session_cap_hours']} hours\n\n"
            f"[dim]{est['notice']}[/dim]",
            title="Kaggle Free-GPU Quota Estimate",
            border_style="cyan",
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle-free-gpu",
        description="Run GPU workloads on Kaggle's free Tesla T4 x2 GPUs from the terminal or any coding agent.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # push
    p_push = sub.add_parser("push", help="Validate and push a kernel folder to Kaggle GPU")
    p_push.add_argument("dir", type=Path, help="Folder containing kernel-metadata.json and script")
    p_push.add_argument("--wait", action="store_true", help="Poll status until COMPLETE or ERROR")
    p_push.add_argument("--poll-interval", type=float, default=10.0, help="Seconds between status polls")
    p_push.add_argument("--out-dir", type=Path, default=None, help="Download output to this folder once complete")
    p_push.add_argument("--allow-mismatch", action="store_true", help="Bypass Gotcha #1 slug mismatch guard (for testing)")
    p_push.add_argument("--json", action="store_true", help="Output raw JSON")
    p_push.set_defaults(func=cmd_push)

    # status
    p_status = sub.add_parser("status", help="Check kernel execution status (QUEUED/RUNNING/COMPLETE/ERROR)")
    p_status.add_argument("ref", help="Kernel reference (<owner>/<title-derived-slug>)")
    p_status.add_argument("--wait", action="store_true", help="Poll status until COMPLETE or ERROR")
    p_status.add_argument("--poll-interval", type=float, default=10.0, help="Seconds between status polls")
    p_status.add_argument("--json", action="store_true", help="Output raw JSON")
    p_status.set_defaults(func=cmd_status)

    # output
    p_out = sub.add_parser("output", help="Download kernel output files and execution logs")
    p_out.add_argument("ref", help="Kernel reference (<owner>/<title-derived-slug>)")
    p_out.add_argument("--out-dir", type=Path, default=Path("./outputs"), help="Destination directory")
    p_out.add_argument("--max-size-mb", type=int, default=500, help="Skip files larger than this size in MB")
    p_out.add_argument("--force-large", action="store_true", help="Download files even if they exceed --max-size-mb")
    p_out.add_argument("--json", action="store_true", help="Output raw JSON")
    p_out.set_defaults(func=cmd_output)

    # list
    p_list = sub.add_parser("list", help="List Kaggle kernels")
    p_list.add_argument("--mine", action="store_true", help="List your own kernels (default if no user/search)")
    p_list.add_argument("--user", default=None, help="Filter by Kaggle username")
    p_list.add_argument("--search", default=None, help="Search query")
    p_list.add_argument("--page-size", type=int, default=20, help="Results per page")
    p_list.add_argument("--json", action="store_true", help="Output raw JSON")
    p_list.set_defaults(func=cmd_list)

    # delete
    p_del = sub.add_parser("delete", help="Permanently delete a Kaggle kernel (requires confirmation)")
    p_del.add_argument("ref", help="Kernel reference (<owner>/<title-derived-slug>)")
    p_del.add_argument("--yes", action="store_true", help="Confirm permanent deletion without interactive prompt")
    p_del.add_argument("--json", action="store_true", help="Output raw JSON")
    p_del.set_defaults(func=cmd_delete)

    # quota
    p_quota = sub.add_parser("quota", help="Estimate rolling 7-day GPU quota from local history")
    p_quota.add_argument("--sync-used-gpu-hours", type=float, default=None, help="Record manual GPU hours used")
    p_quota.add_argument("--json", action="store_true", help="Output raw JSON")
    p_quota.set_defaults(func=cmd_quota)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        rc = args.func(args)
        sys.exit(rc)
    except (KaggleAuthError, KaggleAPIError, FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Error:[/bold red] {_safe_text(exc)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

