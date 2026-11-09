"""Pydantic models and validation rules for Kaggle Free-GPU."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def slugify_title(title: str) -> str:
    """Derive Kaggle's URL slug from a kernel title.

    Gotcha #1: Kaggle derives the kernel URL slug from the TITLE, not the metadata `id`.
    Calling status/output with an id-based slug that differs from the title-derived slug
    returns HTTP 403. Updating an existing kernel requires `id` to match the existing
    title-derived slug exactly, otherwise the push is treated as a create and HTTP 409s.
    """
    normalized = unicodedata.normalize("NFKD", title)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return slug


class ValidationReport(BaseModel):
    """Result of validating kernel-metadata.json and folder contents before pushing."""

    valid: bool
    owner: str = ""
    id_slug: str = ""
    title_slug: str = ""
    code_path: str = ""
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class KernelMetadata(BaseModel):
    """Schema for Kaggle's kernel-metadata.json."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str = Field(..., description="Owner and slug in format 'owner/slug'")
    title: str = Field(..., description="Kernel display title (Kaggle derives URL slug from this)")
    code_file: str = Field(..., description="Relative path to script (.py) or notebook (.ipynb)")
    language: Literal["python", "r", "rmarkdown"] = "python"
    kernel_type: Literal["script", "notebook"] = "script"
    is_private: bool = True
    enable_gpu: bool = True
    enable_tpu: bool = False
    enable_internet: bool = True
    dataset_sources: list[str] = Field(default_factory=list)
    competition_sources: list[str] = Field(default_factory=list)
    kernel_sources: list[str] = Field(default_factory=list)
    model_sources: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    category_ids: list[str] = Field(default_factory=list, alias="categoryIds")

    @property
    def all_tags(self) -> list[str]:
        """Combine keywords and categoryIds deduplicated."""
        seen: set[str] = set()
        result: list[str] = []
        for tag in [*self.keywords, *self.category_ids]:
            if tag not in seen:
                seen.add(tag)
                result.append(tag)
        return result

    def validate_for_push(
        self,
        folder: Path,
        expected_owner: str | None = None,
        allow_mismatch: bool = False,
    ) -> ValidationReport:
        """Validate metadata and code file against all Kaggle gotchas before pushing."""
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Check id format: owner/slug
        if "/" not in self.id or len(self.id.split("/")) != 2:
            errors.append(
                f"Invalid 'id' format '{self.id}': must be '<owner>/<slug>' (e.g. 'myuser/my-kernel-title')."
            )
            owner, id_slug = "", self.id
        else:
            owner, id_slug = self.id.split("/", 1)
            owner = owner.strip()
            id_slug = id_slug.strip()
            if owner in ("<owner>", "your-username", "owner") and expected_owner:
                owner = expected_owner
            if not owner or not id_slug:
                errors.append(f"Invalid 'id' '{self.id}': both owner and slug must be non-empty.")
            elif owner in ("<owner>", "your-username", "owner"):
                errors.append(
                    "Metadata 'id' uses placeholder '<owner>'. Set KAGGLE_USERNAME or replace '<owner>' with your Kaggle username in kernel-metadata.json."
                )

        if expected_owner and owner and owner not in ("<owner>", "your-username", "owner") and owner.lower() != expected_owner.lower():
            warnings.append(
                f"Metadata owner '{owner}' differs from authenticated account '{expected_owner}'."
            )

        # 2. Gotcha #1: Kaggle derives the kernel URL slug from TITLE, not metadata id
        title_slug = slugify_title(self.title)
        if not title_slug:
            errors.append(f"Title '{self.title}' produces an empty slug.")
        elif id_slug != title_slug:
            msg = (
                f"Slug mismatch (Gotcha #1): metadata id slug is '{id_slug}', "
                f"but title '{self.title}' slugifies to '{title_slug}'. "
                f"Kaggle derives the active kernel slug from TITLE, not 'id'. "
                f"Calling status/output with '{owner}/{id_slug}' will fail with HTTP 403, "
                f"and updating an existing kernel with mismatched id/title will fail with HTTP 409. "
                f"Fix: set \"id\": \"{owner}/{title_slug}\" in kernel-metadata.json."
            )
            if allow_mismatch:
                warnings.append(msg)
            else:
                errors.append(msg)

        # 3. Check code_file existence, non-empty, and strict containment within folder (prevent path traversal)
        resolved_folder = folder.resolve()
        raw_code_path = Path(self.code_file)
        code_path = (resolved_folder / raw_code_path).resolve()
        if raw_code_path.is_absolute() or raw_code_path.drive or not code_path.is_relative_to(resolved_folder):
            errors.append(
                f"Security error: code_file '{self.code_file}' resolves outside the kernel directory '{resolved_folder}'."
            )
        elif not code_path.is_file():
            errors.append(
                f"Code file '{self.code_file}' does not exist in '{folder}'."
            )
        else:
            content = code_path.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                errors.append(f"Code file '{self.code_file}' is empty.")

        # 4. Gotcha #7: Keep kernels private by default
        if not self.is_private:
            warnings.append(
                "Kernel has 'is_private': false (Gotcha #7). Kernels should stay private unless the user explicitly requested publishing."
            )

        # 5. GPU check reminder
        if not self.enable_gpu:
            warnings.append(
                "Kernel has 'enable_gpu': false -- it will run on CPU only, not Tesla T4 x2."
            )

        return ValidationReport(
            valid=len(errors) == 0,
            owner=owner,
            id_slug=id_slug,
            title_slug=title_slug,
            code_path=str(code_path),
            errors=errors,
            warnings=warnings,
        )


class PushKernelResult(BaseModel):
    """Structured response from pushing a kernel."""

    ok: bool
    ref: str
    title: str
    url: str
    version_number: int | None = None
    invalid_tags: list[str] = Field(default_factory=list)
    enable_gpu: bool = True
    is_private: bool = True
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class KernelStatusResult(BaseModel):
    """Structured response from checking kernel session status."""

    ref: str
    status: str
    failure_message: str | None = None
    hint: str | None = None


class OutputFileInfo(BaseModel):
    """Metadata for a single output file in /kaggle/working."""

    file_name: str
    size_bytes: int | None = None
    downloaded: bool = False
    saved_path: str | None = None
    skipped_reason: str | None = None


class KernelOutputResult(BaseModel):
    """Structured response from pulling kernel session output and logs."""

    ref: str
    out_dir: str
    runtime_seconds: float | None = None
    gpu_hours_recorded: float | None = None
    files: list[OutputFileInfo] = Field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""
    log_file: str | None = None
    warnings: list[str] = Field(default_factory=list)
    hint: str | None = None
