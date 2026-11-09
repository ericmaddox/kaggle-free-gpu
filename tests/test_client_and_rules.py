"""Unit tests for Kaggle Free-GPU models, gotcha validation rules, client RPCs, MCP tools, and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kaggle_free_gpu import server
from kaggle_free_gpu.cli import build_parser
from kaggle_free_gpu.client import (
    KaggleAuthError,
    KaggleGPUClient,
    _safe_extract_zip,
    resolve_auth_headers,
    split_owner_slug,
)
from kaggle_free_gpu.models import KernelMetadata, slugify_title
from kaggle_free_gpu.quota import QuotaLedger


def test_slugify_title_gotcha_1() -> None:
    assert slugify_title("DeepSeek heretic 9B test v7") == "deepseek-heretic-9b-test-v7"
    assert slugify_title("[0.485+ SOTA] CASMI26: From Peaks to Structures") == "0-485-sota-casmi26-from-peaks-to-structures"
    assert slugify_title("KFG QLoRA Finetune Demo") == "kfg-qlora-finetune-demo"


def test_pre_push_validation_catches_slug_mismatch(tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('hello gpu')\n", encoding="utf-8")

    meta = KernelMetadata(
        id="testuser/wrong-id-slug",
        title="Actual Kernel Title",
        code_file="train.py",
    )
    report = meta.validate_for_push(tmp_path)
    assert not report.valid
    assert any("Slug mismatch (Gotcha #1)" in err for err in report.errors)
    assert report.title_slug == "actual-kernel-title"


def test_pre_push_validation_passes_matching_slug(tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('hello gpu')\n", encoding="utf-8")

    meta = KernelMetadata(
        id="testuser/actual-kernel-title",
        title="Actual Kernel Title",
        code_file="train.py",
        is_private=True,
        enable_gpu=True,
        keywords=["gpu", "nlp", "gpu"],
    )
    assert meta.all_tags == ["gpu", "nlp"]
    report = meta.validate_for_push(tmp_path)
    assert report.valid
    assert report.errors == []


def test_pre_push_validation_resolves_owner_placeholder(tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('hello gpu')\n", encoding="utf-8")

    meta = KernelMetadata(
        id="<owner>/actual-kernel-title",
        title="Actual Kernel Title",
        code_file="train.py",
        is_private=False,
        enable_gpu=False,
    )
    report = meta.validate_for_push(tmp_path, expected_owner="resolved_user")
    assert report.valid
    assert report.owner == "resolved_user"
    assert report.id_slug == "actual-kernel-title"
    assert len(report.warnings) == 2


def test_gotcha_3_rejects_broken_cli_host() -> None:
    with pytest.raises(ValueError, match="Gotcha #3"):
        KaggleGPUClient(api_token="KGAT_test", base_url="https://api.kaggle.com/v1")


def test_delete_kernel_requires_explicit_confirmation(tmp_path: Path) -> None:
    client = KaggleGPUClient(api_token="KGAT_test", quota_ledger=QuotaLedger(tmp_path / "usage.json"))
    res = client.delete_kernel("testuser/some-kernel", confirm=False)
    assert res["deleted"] is False
    assert res["confirmation_required"] is True


def test_quota_ledger_rolling_math(tmp_path: Path) -> None:
    ledger = QuotaLedger(tmp_path / "usage.json")
    ledger.record_push("/code/testuser/kfg-gpu-smoke-test", title="Smoke", version_number=1, enable_gpu=True)
    # Update with 180 seconds (3 minutes = 0.05 wall hours = 0.10 T4 x2 GPU hours)
    ledger.update_run_duration("testuser/kfg-gpu-smoke-test", wall_seconds=180.0, source="measured_log", status="COMPLETE")

    est = ledger.estimate()
    assert est["weekly_budget_gpu_hours"] == 30.0
    assert est["t4_x2_multiplier"] == 2.0
    assert est["estimated_gpu_hours_used"] == pytest.approx(0.1, rel=1e-3)
    assert est["estimated_wall_hours_used_t4_x2"] == pytest.approx(0.05, rel=1e-3)
    assert est["estimated_gpu_hours_remaining"] == pytest.approx(29.9, rel=1e-3)
    assert "Settings -> GPU Quota" in est["notice"]


def test_resolve_auth_headers_token_and_basic() -> None:
    headers, user = resolve_auth_headers(api_token="KGAT_dummy123", username="testuser")
    assert headers["Authorization"] == "Bearer KGAT_dummy123"
    assert user == "testuser"

    basic_headers, basic_user = resolve_auth_headers(username="alice", key="hexkey123")
    assert basic_headers["Authorization"].startswith("Basic ")
    assert basic_user == "alice"


def test_split_owner_slug() -> None:
    assert split_owner_slug("testuser/kfg-gpu-smoke-test") == ("testuser", "kfg-gpu-smoke-test")
    assert split_owner_slug("/code/testuser/kfg-gpu-smoke-test") == ("testuser", "kfg-gpu-smoke-test")
    with pytest.raises(ValueError):
        split_owner_slug("invalid_slug_without_slash")


def test_client_push_status_output_and_size_guard(tmp_path: Path) -> None:
    ledger = QuotaLedger(tmp_path / "usage.json")
    client = KaggleGPUClient(api_token="KGAT_test", username="testuser", quota_ledger=ledger)

    kernel_dir = tmp_path / "my_kernel"
    kernel_dir.mkdir()
    (kernel_dir / "script.py").write_text("print('ok')\n", encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "<owner>/my-gpu-test",
                "title": "My GPU Test",
                "code_file": "script.py",
                "keywords": ["gpu", "invalid_tag_xyz"],
            }
        ),
        encoding="utf-8",
    )

    mock_push_resp = MagicMock()
    mock_push_resp.status_code = 200
    mock_push_resp.json.return_value = {
        "ref": "/code/testuser/my-gpu-test",
        "url": "https://www.kaggle.com/code/testuser/my-gpu-test",
        "versionNumber": 1,
        "invalidTags": ["invalid_tag_xyz"],
    }

    with patch("httpx.Client.post", return_value=mock_push_resp) as mock_post:
        push_res = client.push_kernel(kernel_dir)
        assert push_res.ok is True
        assert push_res.ref == "testuser/my-gpu-test"
        assert push_res.invalid_tags == ["invalid_tag_xyz"]
        assert any("Gotcha #9" in w for w in push_res.warnings)
        sent_json = mock_post.call_args.kwargs["json"]
        assert sent_json["slug"] == "testuser/my-gpu-test"
        assert sent_json["enableGpu"] is True

    # Test 403 status handling (Gotcha #1)
    mock_403_resp = MagicMock()
    mock_403_resp.status_code = 403
    with patch("httpx.Client.post", return_value=mock_403_resp):
        st_403 = client.kernel_status("testuser/wrong-slug")
        assert st_403.status == "ERROR_403_FORBIDDEN"
        assert st_403.hint and "Gotcha #1" in st_403.hint

    # Test output size guard skipping > max_size_mb (Gotcha #5)
    mock_out_resp = MagicMock()
    mock_out_resp.status_code = 200
    mock_out_resp.json.return_value = {
        "files": [
            {"fileName": "huge_model.gguf", "url": "https://www.kaggleusercontent.com/kf/123/huge_model.gguf"},
        ],
        "log": json.dumps([{"stream_name": "stdout", "time": 42.5, "data": "done\n"}]),
    }
    mock_head_resp = MagicMock()
    mock_head_resp.headers = {"Content-Length": str(600 * 1024 * 1024)}  # 600 MB

    out_dir = tmp_path / "out"
    with (
        patch("httpx.Client.post", return_value=mock_out_resp),
        patch("httpx.Client.head", return_value=mock_head_resp),
    ):
        out_res = client.kernel_output("testuser/my-gpu-test", out_dir, max_size_mb=500, force_large=False)
        assert out_res.runtime_seconds == 42.5
        assert len(out_res.files) == 1
        assert out_res.files[0].downloaded is False
        assert any("Gotcha #5" in w for w in out_res.warnings)


def test_security_controls_path_traversal_ssrf_and_zip_slip(tmp_path: Path) -> None:
    import zipfile

    # 1. Block code_file path traversal outside kernel folder
    kernel_dir = tmp_path / "kdir"
    kernel_dir.mkdir()
    outside_secret = tmp_path / "secret.py"
    outside_secret.write_text("SECRET = 123\n", encoding="utf-8")

    meta = KernelMetadata(
        id="testuser/my-test",
        title="My Test",
        code_file="../secret.py",
    )
    report = meta.validate_for_push(kernel_dir)
    assert not report.valid
    assert any("Security error" in err for err in report.errors)

    # 2. Reject cleartext HTTP or non-Kaggle base_url (prevent credential leak)
    with pytest.raises(ValueError, match="Security error"):
        KaggleGPUClient(api_token="KGAT_test", base_url="http://www.kaggle.com/api/v1")
    with pytest.raises(ValueError, match="Security error"):
        KaggleGPUClient(api_token="KGAT_test", base_url="https://evil.example.com/api/v1")

    # 3. Reject slug injection characters
    with pytest.raises(ValueError, match="Invalid characters"):
        split_owner_slug("testuser/bad;slug?query=1")

    # 4. Reject SSRF / untrusted download URL in kernel_output
    ledger = QuotaLedger(tmp_path / "usage.json")
    client = KaggleGPUClient(api_token="KGAT_test", username="testuser", quota_ledger=ledger)
    mock_ssrf_resp = MagicMock()
    mock_ssrf_resp.status_code = 200
    mock_ssrf_resp.json.return_value = {
        "files": [
            {"fileName": "meta.json", "url": "http://169.254.169.254/latest/meta-data/"},
        ],
        "log": "[]",
    }
    with patch("httpx.Client.post", return_value=mock_ssrf_resp):
        ssrf_out = client.kernel_output("testuser/my-test", tmp_path / "out_ssrf")
        assert ssrf_out.files[0].downloaded is False
        assert any("Security warning" in w for w in ssrf_out.warnings)

    # 5. Reject Zip Slip archive member escaping target_dir
    bad_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../../escaped.txt", "pwned")
    with pytest.raises(ValueError, match="Zip Slip"):
        _safe_extract_zip(bad_zip, tmp_path / "extract_target", max_total_bytes=1024 * 1024)

    # 6. Reject CRLF / header injection in credentials
    with pytest.raises(KaggleAuthError, match="Invalid characters in Kaggle credential"):
        resolve_auth_headers(api_token="KGAT_test\r\nX-Injected: evil")

    # 7. Reject invalid characters in metadata 'id'
    bad_meta = KernelMetadata(
        id="testuser/bad$(id)",
        title="Bad ID Title",
        code_file="script.py",
    )
    bad_report = bad_meta.validate_for_push(kernel_dir)
    assert not bad_report.valid
    assert any("Invalid characters in metadata 'id'" in err for err in bad_report.errors)


def test_mcp_server_tools_and_cli_parser(tmp_path: Path) -> None:
    ledger = QuotaLedger(tmp_path / "usage.json")
    dummy_client = KaggleGPUClient(api_token="KGAT_test", username="testuser", quota_ledger=ledger)

    with patch.object(server, "get_client", return_value=dummy_client):
        del_json = json.loads(server.delete_kernel("testuser/my-gpu-test", confirm=False))
        assert del_json["deleted"] is False
        assert del_json["confirmation_required"] is True

        quota_json = json.loads(server.quota_estimate())
        assert quota_json["weekly_budget_gpu_hours"] == 30.0

    parser = build_parser()
    args = parser.parse_args(["push", "examples/01_gpu_smoke_test", "--wait"])
    assert args.command == "push"
    assert args.wait is True
