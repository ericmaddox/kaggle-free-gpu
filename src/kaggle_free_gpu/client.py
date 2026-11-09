"""Direct REST & RPC client for Kaggle Free-GPU compute.

Verified live on 2026-11-08 against https://www.kaggle.com/api/v1.
Does NOT use the broken api.kaggle.com host from Kaggle CLI 1.8.
"""

from __future__ import annotations

import base64
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from kaggle_free_gpu.models import (
    KernelMetadata,
    KernelOutputResult,
    KernelStatusResult,
    OutputFileInfo,
    PushKernelResult,
    ValidationReport,
    slugify_title,
)
from kaggle_free_gpu.quota import QuotaLedger

BASE_URL = "https://www.kaggle.com/api/v1"


class KaggleAuthError(RuntimeError):
    """Raised when no Kaggle credentials can be found or authentication fails."""


class KaggleAPIError(RuntimeError):
    """Raised when the Kaggle API returns an HTTP or application error."""

    def __init__(self, status_code: int, message: str, hint: str | None = None) -> None:
        self.status_code = status_code
        self.message = message
        self.hint = hint
        full = f"HTTP {status_code}: {message}"
        if hint:
            full = f"{full} | Hint: {hint}"
        super().__init__(full)


def _sanitize_credential(value: str, label: str) -> str:
    """Strip outer whitespace and reject control characters (prevents HTTP header / CRLF injection)."""
    cleaned = str(value).strip()
    if not cleaned or any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise KaggleAuthError(f"Invalid characters in Kaggle credential '{label}'.")
    return cleaned


def resolve_auth_headers(
    api_token: str | None = None,
    username: str | None = None,
    key: str | None = None,
) -> tuple[dict[str, str], str | None]:
    """Resolve Kaggle auth headers without ever logging or printing secret tokens.

    Supports:
    1. Explicit `api_token` (`KGAT_...`) or `username` + `key`
    2. `KAGGLE_API_TOKEN` environment variable (`Bearer`)
    3. `~/.kaggle/access_token` file (`Bearer`)
    4. `KAGGLE_USERNAME` + `KAGGLE_KEY` environment variables (`Basic` or `Bearer` if `KGAT_`)
    5. `~/.kaggle/kaggle.json` file (`Basic` or `Bearer` if `KGAT_`)

    Returns:
        Tuple of (headers_dict, known_username_or_none)
    """
    raw_env_user = username or os.environ.get("KAGGLE_USERNAME")
    env_username = _sanitize_credential(raw_env_user, "username") if raw_env_user else None

    # 1. Explicit arguments take highest priority
    if api_token:
        tok = _sanitize_credential(api_token, "api_token")
        return {"Authorization": f"Bearer {tok}"}, env_username
    if key and key.startswith("KGAT_"):
        tok = _sanitize_credential(key, "key")
        return {"Authorization": f"Bearer {tok}"}, env_username
    if username and key:
        u = _sanitize_credential(username, "username")
        k = _sanitize_credential(key, "key")
        raw = f"{u}:{k}".encode()
        b64 = base64.b64encode(raw).decode("ascii")
        return {"Authorization": f"Basic {b64}"}, u

    # 2. Environment variable or ~/.kaggle/access_token
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        access_token_path = Path.home() / ".kaggle" / "access_token"
        if access_token_path.is_file():
            token = access_token_path.read_text(encoding="utf-8").strip()

    if token:
        tok = _sanitize_credential(token, "access_token")
        return {"Authorization": f"Bearer {tok}"}, env_username

    env_key = key or os.environ.get("KAGGLE_KEY")
    if env_key and env_key.startswith("KGAT_"):
        tok = _sanitize_credential(env_key, "KAGGLE_KEY")
        return {"Authorization": f"Bearer {tok}"}, env_username

    if env_username and env_key:
        k = _sanitize_credential(env_key, "KAGGLE_KEY")
        raw = f"{env_username}:{k}".encode()
        b64 = base64.b64encode(raw).decode("ascii")
        return {"Authorization": f"Basic {b64}"}, env_username

    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json_path.is_file():
        try:
            data = json.loads(kaggle_json_path.read_text(encoding="utf-8"))
            file_token = data.get("token") or data.get("access_token")
            raw_file_user = data.get("username") or env_username
            file_user = _sanitize_credential(raw_file_user, "username") if raw_file_user else None
            file_key = data.get("key")
            if file_token:
                tok = _sanitize_credential(str(file_token), "token")
                return {"Authorization": f"Bearer {tok}"}, file_user
            if file_key and str(file_key).startswith("KGAT_"):
                tok = _sanitize_credential(str(file_key), "key")
                return {"Authorization": f"Bearer {tok}"}, file_user
            if file_user and file_key:
                k = _sanitize_credential(str(file_key), "key")
                raw = f"{file_user}:{k}".encode()
                b64 = base64.b64encode(raw).decode("ascii")
                return {"Authorization": f"Basic {b64}"}, file_user
        except KaggleAuthError:
            raise
        except Exception as exc:
            raise KaggleAuthError(
                f"Failed to parse ~/.kaggle/kaggle.json ({type(exc).__name__})."
            ) from exc

    raise KaggleAuthError(
        "No Kaggle credentials found. Provide one of:\n"
        "  1. Set KAGGLE_API_TOKEN=KGAT_... or save it to ~/.kaggle/access_token\n"
        "  2. Set KAGGLE_USERNAME and KAGGLE_KEY or save them to ~/.kaggle/kaggle.json"
    )


_SAFE_SLUG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
_MAX_METADATA_BYTES = 1024 * 1024  # 1 MB cap for kernel-metadata.json
_MAX_ZIP_FILES = 10_000
_ALLOWED_API_HOSTS = {"www.kaggle.com", "kaggle.com"}
_ALLOWED_DOWNLOAD_HOST_SUFFIXES = (
    ".kaggleusercontent.com",
    "kaggleusercontent.com",
    ".kaggle.com",
    "kaggle.com",
    "storage.googleapis.com",
)


def split_owner_slug(owner_slug: str) -> tuple[str, str]:
    """Split 'owner/slug' (or '/code/owner/slug') and validate both parts against injection."""
    cleaned = owner_slug.strip().strip("/")
    if cleaned.startswith("code/"):
        cleaned = cleaned[len("code/") :].strip("/")
    if "/" not in cleaned or len(cleaned.split("/")) != 2:
        raise ValueError(
            f"Invalid kernel reference '{owner_slug}'. Expected format: '<owner>/<slug>'."
        )
    owner, slug = cleaned.split("/", 1)
    if not owner or not slug:
        raise ValueError(
            f"Invalid kernel reference '{owner_slug}'. Both owner and slug must be non-empty."
        )
    if (
        ".." in owner
        or ".." in slug
        or not _SAFE_SLUG_RE.match(owner)
        or not _SAFE_SLUG_RE.match(slug)
    ):
        raise ValueError(
            f"Invalid characters in kernel reference '{owner_slug}'. Only alphanumeric, '-', '_', and '.' are allowed."
        )
    return owner, slug


def _validate_base_url(base_url: str) -> str:
    """Ensure base_url uses HTTPS and targets the official Kaggle host (prevents credential leak)."""
    if "api.kaggle.com" in base_url:
        raise ValueError(
            "Gotcha #3 violation: Do NOT use 'api.kaggle.com' (broken host in Kaggle CLI 1.8). "
            "Use 'https://www.kaggle.com/api/v1'."
        )
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in _ALLOWED_API_HOSTS:
        raise ValueError(
            f"Security error: Untrusted or non-HTTPS base_url '{base_url}'. "
            "Credentials may only be sent to 'https://www.kaggle.com/api/v1'."
        )
    return base_url.rstrip("/")


def _is_trusted_download_url(url: str) -> bool:
    """Validate that a signed output download URL uses HTTPS and an official Kaggle/GCS domain."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    return any(
        host == suffix.lstrip(".") or (suffix.startswith(".") and host.endswith(suffix))
        for suffix in _ALLOWED_DOWNLOAD_HOST_SUFFIXES
    )


def _safe_extract_zip(zip_path: Path, target_dir: Path, max_total_bytes: int) -> None:
    """Safely extract a zip archive while blocking Zip Slip, symlinks, and Zip Bombs."""
    resolved_target = target_dir.resolve()
    total_extracted = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        if len(members) > _MAX_ZIP_FILES:
            raise ValueError(
                f"Security error: Zip archive contains too many entries ({len(members)} > {_MAX_ZIP_FILES})."
            )
        for info in members:
            # Reject symlinks inside zip archive (POSIX S_IFLNK = 0o120000)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError(f"Security error: Refusing to extract symlink '{info.filename}' from zip archive.")

            normalized_name = info.filename.replace("\\", "/")
            rel_member = Path(normalized_name)
            if (
                "\\" in info.filename
                or rel_member.is_absolute()
                or rel_member.drive
                or ".." in rel_member.parts
            ):
                raise ValueError(f"Security error: Zip Slip path traversal detected in '{info.filename}'.")

            dest = (resolved_target / rel_member).resolve()
            if not dest.is_relative_to(resolved_target):
                raise ValueError(f"Security error: Zip member '{info.filename}' resolves outside target directory.")
            if dest.is_symlink():
                raise ValueError(f"Security error: Refusing to overwrite symlink at '{dest}'.")

            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src_f, dest.open("wb") as dst_f:
                while True:
                    chunk = src_f.read(65536)
                    if not chunk:
                        break
                    total_extracted += len(chunk)
                    if total_extracted > max_total_bytes:
                        dst_f.close()
                        dest.unlink(missing_ok=True)
                        raise ValueError(
                            f"Security error: Zip archive exceeded maximum uncompressed size ({max_total_bytes} bytes)."
                        )
                    dst_f.write(chunk)


class KaggleGPUClient:
    """Client for Kaggle Free-GPU kernel operations over https://www.kaggle.com/api/v1."""

    def __init__(
        self,
        api_token: str | None = None,
        username: str | None = None,
        key: str | None = None,
        base_url: str = BASE_URL,
        quota_ledger: QuotaLedger | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = _validate_base_url(base_url)
        self._auth_headers, self.username = resolve_auth_headers(
            api_token=api_token,
            username=username,
            key=key,
        )
        self.quota_ledger = quota_ledger or QuotaLedger()
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            **self._auth_headers,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def resolve_username(self) -> str | None:
        """Resolve the authenticated user's Kaggle username if not already set."""
        if self.username:
            return self.username
        try:
            items = self.list_kernels(mine=True, page_size=1)
            if items and "/" in items[0].get("ref", ""):
                self.username = items[0]["ref"].split("/", 1)[0]
        except Exception:
            pass
        return self.username

    def validate_kernel_dir(
        self,
        dir_path: str | Path,
        allow_mismatch: bool = False,
    ) -> tuple[KernelMetadata, ValidationReport]:
        """Load and validate kernel-metadata.json from a directory."""
        folder = Path(dir_path).expanduser().resolve()
        meta_path = (folder / "kernel-metadata.json").resolve()
        if not meta_path.is_relative_to(folder) or not meta_path.is_file():
            raise FileNotFoundError(
                f"Missing kernel-metadata.json in directory: {folder}"
            )
        if meta_path.stat().st_size > _MAX_METADATA_BYTES:
            raise ValueError(
                f"kernel-metadata.json exceeds maximum allowed size ({_MAX_METADATA_BYTES} bytes)."
            )
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = KernelMetadata.model_validate(raw)
        if meta.id.startswith(("<owner>/", "your-username/", "owner/")) and not self.username:
            self.resolve_username()
        report = meta.validate_for_push(
            folder=folder,
            expected_owner=self.username,
            allow_mismatch=allow_mismatch,
        )
        return meta, report

    def push_kernel(
        self,
        dir_path: str | Path,
        allow_mismatch: bool = False,
    ) -> PushKernelResult:
        """Validate kernel-metadata.json and push the kernel to Kaggle via JSON body.

        Gotchas enforced:
        - #1: Metadata `id` slug must match title-derived slug (`slugify_title(title)`).
        - #2: Sends JSON body (`Content-Type: application/json`), never multipart form data.
        - #3: Uses `https://www.kaggle.com/api/v1/kernels/push`.
        - #7: Warns if `is_private` is False.
        - #8: Records push in local GPU quota ledger (each push starts a new version & run).
        - #9: Reports `invalidTags` returned by Kaggle's taxonomy check.
        """
        folder = Path(dir_path).expanduser().resolve()
        meta, report = self.validate_kernel_dir(folder, allow_mismatch=allow_mismatch)

        if not report.valid:
            return PushKernelResult(
                ok=False,
                ref=meta.id,
                title=meta.title,
                url="",
                enable_gpu=meta.enable_gpu,
                is_private=meta.is_private,
                warnings=report.warnings,
                error="Pre-push validation failed:\n- " + "\n- ".join(report.errors),
            )

        code_path = Path(report.code_path)
        code_text = code_path.read_text(encoding="utf-8")

        resolved_slug = f"{report.owner}/{report.id_slug}"
        # For notebooks (.ipynb), Kaggle expects the raw notebook JSON string in `text`.
        payload: dict[str, Any] = {
            "slug": resolved_slug,
            "newTitle": meta.title,
            "text": code_text,
            "language": meta.language,
            "kernelType": meta.kernel_type,
            "isPrivate": meta.is_private,
            "enableGpu": meta.enable_gpu,
            "enableTpu": meta.enable_tpu,
            "enableInternet": meta.enable_internet,
            "datasetDataSources": meta.dataset_sources,
            "competitionDataSources": meta.competition_sources,
            "kernelDataSources": meta.kernel_sources,
            "modelDataSources": meta.model_sources,
            "categoryIds": meta.all_tags,
        }

        url = f"{self.base_url}/kernels/push"
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            resp = client.post(url, headers=self._headers(), json=payload)

        if resp.status_code >= 400:
            body_text = resp.text[:500]
            hint = None
            if resp.status_code == 409:
                hint = (
                    "HTTP 409 Conflict (Gotcha #1): A kernel with this title already exists, "
                    "Wait or ensure metadata 'id' matches the existing kernel's title-derived slug "
                    f"('{report.owner}/{slugify_title(meta.title)}') to update it, or choose a new title."
                )
            elif resp.status_code == 403:
                hint = (
                    "HTTP 403 Forbidden: Check that the owner in metadata 'id' matches your "
                    "Kaggle username and your API token has kernel write permissions."
                )
            return PushKernelResult(
                ok=False,
                ref=meta.id,
                title=meta.title,
                url="",
                enable_gpu=meta.enable_gpu,
                is_private=meta.is_private,
                warnings=report.warnings,
                error=f"HTTP {resp.status_code} from /kernels/push: {body_text}"
                + (f"\nHint: {hint}" if hint else ""),
            )

        data = resp.json()
        api_error = data.get("error")
        invalid_tags = data.get("invalidTags") or []
        warnings = list(report.warnings)
        if invalid_tags:
            warnings.append(
                f"Gotcha #9: Kaggle dropped invalid taxonomy tags: {invalid_tags}. "
                "Only tags in Kaggle's official taxonomy stick."
            )

        if api_error:
            return PushKernelResult(
                ok=False,
                ref=data.get("ref") or meta.id,
                title=meta.title,
                url=data.get("url") or "",
                version_number=data.get("versionNumber"),
                invalid_tags=invalid_tags,
                enable_gpu=meta.enable_gpu,
                is_private=meta.is_private,
                warnings=warnings,
                error=f"Kaggle API error: {api_error}",
            )

        raw_ref = data.get("ref") or f"{report.owner}/{report.title_slug}"
        o, s = split_owner_slug(raw_ref)
        actual_ref = f"{o}/{s}"
        actual_url = data.get("url") or f"https://www.kaggle.com/code/{actual_ref}"
        version_number = data.get("versionNumber")

        # Record push in rolling 7-day GPU quota ledger (Gotcha #8)
        self.quota_ledger.record_push(
            ref=actual_ref,
            title=meta.title,
            version_number=version_number,
            enable_gpu=meta.enable_gpu,
        )

        return PushKernelResult(
            ok=True,
            ref=actual_ref,
            title=meta.title,
            url=actual_url,
            version_number=version_number,
            invalid_tags=invalid_tags,
            enable_gpu=meta.enable_gpu,
            is_private=meta.is_private,
            warnings=warnings,
        )

    def kernel_status(self, owner_slug: str) -> KernelStatusResult:
        """Check kernel session status via KernelsApiService/GetKernelSessionStatus.

        Gotcha #4: Plain REST path /kernels/status/{owner}/{slug} does NOT exist.
        Gotcha #1: Calling with an id-based slug that doesn't match the title-derived slug returns 403.
        """
        owner, slug = split_owner_slug(owner_slug)
        ref = f"{owner}/{slug}"
        url = f"{self.base_url}/kernels.KernelsApiService/GetKernelSessionStatus"
        payload = {"userName": owner, "kernelSlug": slug}

        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            resp = client.post(url, headers=self._headers(), json=payload)

        if resp.status_code == 403:
            return KernelStatusResult(
                ref=ref,
                status="ERROR_403_FORBIDDEN",
                failure_message=f"HTTP 403 Forbidden for '{ref}'.",
                hint=(
                    "Gotcha #1: Kaggle derives the kernel slug from the TITLE, not the metadata 'id'. "
                    "Ensure you are passing '<owner>/<title-derived-slug>' (check with list_kernels)."
                ),
            )
        if resp.status_code >= 400:
            raise KaggleAPIError(
                status_code=resp.status_code,
                message=resp.text[:500],
            )

        data = resp.json()
        raw_status = str(data.get("status") or "UNKNOWN").upper()
        # Normalize enum prefixes if Kaggle ever returns KERNEL_WORKER_STATUS_COMPLETE
        for prefix in ("KERNEL_WORKER_STATUS_", "KERNEL_SESSION_STATUS_"):
            if raw_status.startswith(prefix):
                raw_status = raw_status[len(prefix) :]

        failure_msg = data.get("failureMessage") or data.get("errorMessage")

        if raw_status in ("COMPLETE", "ERROR"):
            self.quota_ledger.update_run_duration(ref=ref, status=raw_status)

        return KernelStatusResult(
            ref=ref,
            status=raw_status,
            failure_message=failure_msg,
        )

    def kernel_output(
        self,
        owner_slug: str,
        out_dir: str | Path,
        max_size_mb: int = 500,
        force_large: bool = False,
    ) -> KernelOutputResult:
        """Download kernel output files and logs via KernelsApiService/ListKernelSessionOutput.

        Gotcha #4: Uses service endpoint ListKernelSessionOutput.
        Gotcha #5: Checks Content-Length via HEAD before downloading each file in /kaggle/working
        so a multi-GB model left behind never causes an IncompleteRead crash.
        Also saves stdout/stderr logs and updates the local GPU quota ledger with exact runtime!
        """
        owner, slug = split_owner_slug(owner_slug)
        ref = f"{owner}/{slug}"
        target_dir = Path(out_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        url = f"{self.base_url}/kernels.KernelsApiService/ListKernelSessionOutput"
        payload = {"userName": owner, "kernelSlug": slug}

        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            resp = client.post(url, headers=self._headers(), json=payload)

            if resp.status_code == 403:
                return KernelOutputResult(
                    ref=ref,
                    out_dir=str(target_dir),
                    warnings=["HTTP 403 Forbidden when listing kernel output."],
                    hint=(
                        "Gotcha #1: Kaggle derives the kernel slug from the TITLE, not metadata 'id'. "
                        "Verify the title-derived slug via list_kernels."
                    ),
                )
            if resp.status_code >= 400:
                raise KaggleAPIError(resp.status_code, resp.text[:500])

            data = resp.json()
            raw_files = data.get("files") or []
            raw_log_str = data.get("log")

            warnings: list[str] = []
            runtime_seconds: float | None = None
            gpu_hours_recorded: float | None = None
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []
            log_file_path: str | None = None

            # 1. Process execution log if present
            if raw_log_str:
                try:
                    log_events = json.loads(raw_log_str) if isinstance(raw_log_str, str) else raw_log_str
                    if isinstance(log_events, list):
                        for ev in log_events:
                            stream = ev.get("stream_name", "stdout")
                            text = ev.get("data", "")
                            t = ev.get("time")
                            if isinstance(t, (int, float)):
                                runtime_seconds = max(runtime_seconds or 0.0, float(t))
                            if stream == "stderr":
                                stderr_chunks.append(text)
                            else:
                                stdout_chunks.append(text)

                        log_path = target_dir / "kernel.log"
                        stdout_full = "".join(stdout_chunks)
                        stderr_full = "".join(stderr_chunks)
                        combined = stdout_full
                        if stderr_full.strip():
                            combined += "\n--- STDERR ---\n" + stderr_full
                        log_path.write_text(combined, encoding="utf-8")
                        log_file_path = str(log_path)
                except Exception as exc:
                    warnings.append(f"Could not parse kernel log JSON: {exc}")

            if runtime_seconds is not None:
                updated = self.quota_ledger.update_run_duration(
                    ref=ref,
                    wall_seconds=runtime_seconds,
                    source="measured_log",
                    status="COMPLETE",
                )
                if updated:
                    gpu_hours_recorded = round(updated.gpu_hours_used, 4)

            # 2. Inspect and download output files with size guard (Gotcha #5)
            max_bytes = max_size_mb * 1024 * 1024
            total_planned_bytes = 0
            file_infos: list[OutputFileInfo] = []

            for f_item in raw_files:
                file_name = f_item.get("fileName") or ""
                file_url = f_item.get("url") or ""
                if not file_name or not file_url:
                    continue

                if not _is_trusted_download_url(file_url):
                    msg = f"Security warning: Refusing to download '{file_name}' from untrusted URL '{file_url}'."
                    warnings.append(msg)
                    file_infos.append(
                        OutputFileInfo(
                            file_name=file_name,
                            downloaded=False,
                            skipped_reason=msg,
                        )
                    )
                    continue

                # Check size via HEAD request first (using unauthenticated request to signed CDN URL)
                size_bytes: int | None = None
                try:
                    head_resp = httpx.head(file_url, timeout=self.timeout, follow_redirects=False)
                    cl = head_resp.headers.get("Content-Length")
                    if cl is not None and cl.isdigit():
                        size_bytes = int(cl)
                except Exception:
                    size_bytes = None

                if size_bytes is not None and size_bytes > max_bytes and not force_large:
                    size_mb = size_bytes / (1024 * 1024)
                    msg = (
                        f"Skipped '{file_name}' ({size_mb:.1f} MB > max_size_mb={max_size_mb} MB). "
                        "Gotcha #5: Large intermediates left in /kaggle/working cause download failures. "
                        "Delete large model files inside your kernel script before exit, or pass force_large=True."
                    )
                    warnings.append(msg)
                    file_infos.append(
                        OutputFileInfo(
                            file_name=file_name,
                            size_bytes=size_bytes,
                            downloaded=False,
                            skipped_reason=msg,
                        )
                    )
                    continue

                if size_bytes is not None:
                    total_planned_bytes += size_bytes
                    if total_planned_bytes > max_bytes * 2 and not force_large:
                        msg = (
                            f"Skipped '{file_name}' because cumulative output size exceeded {max_size_mb * 2} MB."
                        )
                        warnings.append(msg)
                        file_infos.append(
                            OutputFileInfo(
                                file_name=file_name,
                                size_bytes=size_bytes,
                                downloaded=False,
                                skipped_reason=msg,
                            )
                        )
                        continue

                # Prevent path traversal (strict containment inside target_dir; reject traversal paths outright)
                raw_rel = Path(file_name.replace("\\", "/"))
                dest_path = (target_dir / raw_rel).resolve()
                if (
                    "\\" in file_name
                    or raw_rel.is_absolute()
                    or raw_rel.drive
                    or ".." in raw_rel.parts
                    or dest_path == target_dir
                    or dest_path.is_symlink()
                    or not dest_path.is_relative_to(target_dir)
                ):
                    msg = f"Security warning: Skipped '{file_name}' due to unsafe or traversal file path."
                    warnings.append(msg)
                    file_infos.append(
                        OutputFileInfo(
                            file_name=file_name,
                            downloaded=False,
                            skipped_reason=msg,
                        )
                    )
                    continue
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Stream download with mid-stream byte budget enforcement (in case Content-Length was omitted)
                downloaded_bytes = 0
                aborted_mid_stream = False
                with httpx.stream("GET", file_url, timeout=self.timeout, follow_redirects=False) as dl_resp:
                    dl_resp.raise_for_status()
                    with dest_path.open("wb") as out_f:
                        for chunk in dl_resp.iter_bytes(chunk_size=65536):
                            downloaded_bytes += len(chunk)
                            if size_bytes is None:
                                total_planned_bytes += len(chunk)
                            if not force_large and (
                                downloaded_bytes > max_bytes or total_planned_bytes > max_bytes * 2
                            ):
                                aborted_mid_stream = True
                                break
                            out_f.write(chunk)

                if aborted_mid_stream:
                    dest_path.unlink(missing_ok=True)
                    msg = (
                        f"Aborted download of '{file_name}' after exceeding max_size_mb={max_size_mb} MB mid-stream (Gotcha #5)."
                    )
                    warnings.append(msg)
                    file_infos.append(
                        OutputFileInfo(
                            file_name=file_name,
                            size_bytes=downloaded_bytes,
                            downloaded=False,
                            skipped_reason=msg,
                        )
                    )
                    continue

                # If it's a zip archive, extract it safely into target_dir (blocking Zip Slip & Zip Bombs)
                if dest_path.suffix.lower() == ".zip" and zipfile.is_zipfile(dest_path):
                    _safe_extract_zip(dest_path, target_dir, max_total_bytes=max_bytes * 2)

                file_infos.append(
                    OutputFileInfo(
                        file_name=file_name,
                        size_bytes=size_bytes or downloaded_bytes,
                        downloaded=True,
                        saved_path=str(dest_path),
                    )
                )

            stdout_tail = "".join(stdout_chunks)[-4000:]
            stderr_tail = "".join(stderr_chunks)[-2000:]

            return KernelOutputResult(
                ref=ref,
                out_dir=str(target_dir),
                runtime_seconds=round(runtime_seconds, 2) if runtime_seconds is not None else None,
                gpu_hours_recorded=gpu_hours_recorded,
                files=file_infos,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                log_file=log_file_path,
                warnings=warnings,
            )

    def list_kernels(
        self,
        mine: bool = True,
        user: str | None = None,
        search: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> list[dict[str, Any]]:
        """List kernels via GET /api/v1/kernels/list.

        Note: In Kaggle's REST API, listing the authenticated user's own kernels uses
        `group=profile` (passing `mine=true` or `group=my` returns HTTP 400).
        """
        params: dict[str, Any] = {
            "page": page,
            "pageSize": page_size,
        }
        if mine:
            params["group"] = "profile"
        if user:
            params["user"] = user
        if search:
            params["search"] = search

        url = f"{self.base_url}/kernels/list"
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            resp = client.get(url, headers=self._headers(), params=params)

        if resp.status_code >= 400:
            raise KaggleAPIError(resp.status_code, resp.text[:500])

        items = resp.json()
        if not isinstance(items, list):
            return []

        results: list[dict[str, Any]] = []
        for item in items:
            ref = item.get("ref") or ""
            if mine and not self.username and "/" in ref:
                self.username = ref.split("/", 1)[0]
            results.append(
                {
                    "ref": ref,
                    "title": item.get("title") or "",
                    "author": item.get("author") or "",
                    "last_run_time": item.get("lastRunTime") or "",
                    "enable_gpu": item.get("enableGpu"),
                    "is_private": item.get("isPrivate"),
                    "total_votes": item.get("totalVotes", 0),
                }
            )
        return results

    def delete_kernel(self, owner_slug: str, confirm: bool = False) -> dict[str, Any]:
        """Permanently delete a kernel via KernelsApiService/DeleteKernel.

        Safety rule: Requires explicit user confirmation (`confirm=True`) every time.
        """
        owner, slug = split_owner_slug(owner_slug)
        ref = f"{owner}/{slug}"

        if not confirm:
            return {
                "deleted": False,
                "ref": ref,
                "confirmation_required": True,
                "message": (
                    f"Deletion of '{ref}' is PERMANENT and UNRECOVERABLE. "
                    "Ask the user for explicit confirmation first, then call delete_kernel "
                    "with confirm=True."
                ),
            }

        url = f"{self.base_url}/kernels.KernelsApiService/DeleteKernel"
        payload = {"userName": owner, "kernelSlug": slug}

        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            resp = client.post(url, headers=self._headers(), json=payload)

        if resp.status_code >= 400:
            raise KaggleAPIError(resp.status_code, resp.text[:500])

        return {
            "deleted": True,
            "ref": ref,
            "response": resp.json() if resp.text.strip() else {},
        }

    def quota_estimate(self) -> dict[str, Any]:
        """Return rolling 7-day GPU quota estimate from local push history."""
        return self.quota_ledger.estimate()
