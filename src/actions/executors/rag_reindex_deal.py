from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from actions.engine.models import ActionError, ActionPayload, ArtifactMetadata, now_utc_iso
from actions.executors._artifacts import resolve_action_artifact_dir
from actions.executors.base import ActionExecutionError, ActionExecutor, ExecutionContext, ExecutionResult

try:
    from zakops_secret_scan import find_secrets_in_text
except Exception:  # pragma: no cover
    find_secrets_in_text = None  # type: ignore


INCLUDE_EXTENSIONS = {".md", ".txt", ".csv", ".json", ".yaml", ".yml"}


def _dataroom_root() -> Path:
    return Path(os.getenv("DATAROOM_ROOT", "/home/zaks/DataRoom")).resolve()


def _rag_api_url() -> str:
    return str(os.getenv("RAG_API_URL", "http://localhost:8052/rag/add")).strip()


def _generate_synthetic_url(file_path: Path) -> str:
    prefix = (os.getenv("ZAKOPS_RAG_URL_PREFIX") or "https://dataroom.local/DataRoom").rstrip("/")
    try:
        rel = file_path.resolve().relative_to(_dataroom_root())
        return f"{prefix}/{rel.as_posix()}"
    except Exception:
        return f"{prefix}/{file_path.name}"


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_read_text(path: Path, *, max_bytes: int = 500_000) -> str:
    try:
        raw = path.read_bytes()
    except Exception:
        return ""
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    try:
        return raw.decode("utf-8")
    except Exception:
        return raw.decode("utf-8", errors="replace")


def _load_manifest(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict):
        files = data.get("files")
        if isinstance(files, dict):
            return {str(k): str(v) for k, v in files.items() if str(k).strip() and str(v).strip()}
    return {}


def _write_manifest(path: Path, files: Dict[str, str]) -> None:
    payload = {"updated_at": now_utc_iso(), "files": files}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class IndexOutcome:
    indexed: int
    skipped: int
    blocked_secrets: int
    errors: int


class RagReindexDealExecutor(ActionExecutor):
    """
    RAG.REINDEX_DEAL

    Incremental indexing of deal artifacts into the local RAG service.
    """

    action_type = "RAG.REINDEX_DEAL"

    def validate(self, payload: ActionPayload) -> tuple[bool, Optional[str]]:
        # Inputs are flexible; deal_path is resolved via ctx.deal where possible.
        return True, None

    def execute(self, payload: ActionPayload, ctx: ExecutionContext) -> ExecutionResult:
        inputs = payload.inputs or {}
        max_files = int(inputs.get("max_files") or 50)

        deal_folder = ""
        if ctx.deal and str((ctx.deal or {}).get("folder_path") or "").strip():
            deal_folder = str((ctx.deal or {}).get("folder_path") or "").strip()
        elif str(inputs.get("deal_path") or "").strip():
            deal_folder = str(inputs.get("deal_path") or "").strip()

        if not deal_folder:
            raise ActionExecutionError(
                ActionError(
                    code="deal_folder_missing",
                    message="deal folder_path missing (ctx.deal.folder_path or inputs.deal_path required)",
                    category="validation",
                    retryable=False,
                )
            )

        deal_folder_path = Path(deal_folder).expanduser()
        deal_path = (_dataroom_root() / deal_folder_path).resolve() if not deal_folder_path.is_absolute() else deal_folder_path.resolve()
        try:
            deal_path.relative_to(_dataroom_root())
        except ValueError:
            raise ActionExecutionError(
                ActionError(
                    code="deal_path_outside_dataroom",
                    message="Deal path is outside DATAROOM_ROOT",
                    category="validation",
                    retryable=False,
                    details={"deal_path": str(deal_path)},
                )
            )

        # Build candidate file list from inputs.
        candidates: List[Path] = []
        artifact_paths = inputs.get("artifact_paths") or []
        if isinstance(artifact_paths, list):
            for raw in artifact_paths:
                if not isinstance(raw, str) or not raw.strip():
                    continue
                p = Path(raw).expanduser().resolve()
                candidates.append(p)

        bundle_path_raw = str(inputs.get("bundle_path") or "").strip()
        if bundle_path_raw:
            bundle = Path(bundle_path_raw).expanduser().resolve()
            try:
                bundle.relative_to(deal_path)
            except ValueError:
                # best-effort: allow bundle under dataroom root, but still require it to be a deal subtree
                pass
            if bundle.exists() and bundle.is_dir():
                for p in bundle.rglob("*"):
                    if p.is_file():
                        candidates.append(p.resolve())

        # Filter and deduplicate candidates.
        uniq: List[Path] = []
        seen = set()
        for p in candidates:
            if not p.exists() or not p.is_file():
                continue
            if p.suffix.lower() not in INCLUDE_EXTENSIONS:
                continue
            try:
                p.relative_to(_dataroom_root())
            except ValueError:
                continue
            if str(p) in seen:
                continue
            seen.add(str(p))
            uniq.append(p)

        uniq = uniq[: max(1, max_files)]

        manifest_path = deal_path / ".rag_indexed_files.json"
        indexed_manifest = _load_manifest(manifest_path)

        indexed = 0
        skipped = 0
        blocked = 0
        errors = 0

        details: List[Dict[str, Any]] = []

        for p in uniq:
            try:
                rel = p.relative_to(deal_path).as_posix()
            except Exception:
                rel = p.name

            sha = ""
            try:
                sha = _sha256_file(p)
            except Exception:
                errors += 1
                details.append({"path": str(p), "status": "error", "reason": "hash_failed"})
                continue

            if indexed_manifest.get(rel) == sha:
                skipped += 1
                details.append({"path": str(p), "status": "skipped", "reason": "unchanged"})
                continue

            content = _safe_read_text(p)
            if not content.strip():
                skipped += 1
                details.append({"path": str(p), "status": "skipped", "reason": "empty"})
                continue

            # Secret scan gate
            if find_secrets_in_text is not None:
                findings = find_secrets_in_text(content)
                if findings:
                    blocked += 1
                    details.append({"path": str(p), "status": "blocked", "reason": f"secrets:{','.join(findings[:3])}"})
                    continue

            url = _generate_synthetic_url(p)
            metadata = {
                "source": "deal",
                "deal_id": payload.deal_id or inputs.get("deal_id"),
                "path": rel,
                "filename": p.name,
                "modified": int(p.stat().st_mtime),
            }

            try:
                resp = requests.post(_rag_api_url(), json={"url": url, "content": content, "metadata": metadata, "chunk_size": 5000}, timeout=30)
                if resp.status_code != 200:
                    errors += 1
                    details.append({"path": str(p), "status": "error", "reason": f"rag_http_{resp.status_code}"})
                    continue
            except requests.exceptions.ConnectionError:
                raise ActionExecutionError(
                    ActionError(
                        code="rag_api_unavailable",
                        message=f"RAG API not available at {_rag_api_url()}",
                        category="dependency",
                        retryable=True,
                    )
                )
            except Exception as e:
                errors += 1
                details.append({"path": str(p), "status": "error", "reason": f"{type(e).__name__}"})
                continue

            indexed_manifest[rel] = sha
            indexed += 1
            details.append({"path": str(p), "status": "indexed", "url": url})

        _write_manifest(manifest_path, indexed_manifest)

        out_dir = resolve_action_artifact_dir(ctx)
        report_json = out_dir / "rag_reindex_report.json"
        report_md = out_dir / "rag_reindex_report.md"

        report_payload = {
            "deal_id": payload.deal_id or inputs.get("deal_id"),
            "deal_path": str(deal_path),
            "rag_api_url": _rag_api_url(),
            "indexed": indexed,
            "skipped": skipped,
            "blocked_secrets": blocked,
            "errors": errors,
            "details": details[:200],
            "generated_at": now_utc_iso(),
        }
        report_json.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        md_lines = [
            "# RAG Reindex Report",
            "",
            f"- **deal_id:** `{payload.deal_id or inputs.get('deal_id') or ''}`",
            f"- **deal_path:** `{deal_path}`",
            f"- **indexed:** `{indexed}`",
            f"- **skipped:** `{skipped}`",
            f"- **blocked_secrets:** `{blocked}`",
            f"- **errors:** `{errors}`",
            "",
        ]
        report_md.write_text("\n".join(md_lines), encoding="utf-8")

        artifacts = [
            ArtifactMetadata(filename=report_json.name, mime_type="application/json", path=str(report_json), created_at=now_utc_iso()),
            ArtifactMetadata(filename=report_md.name, mime_type="text/markdown", path=str(report_md), created_at=now_utc_iso()),
        ]

        outputs: Dict[str, Any] = {
            "deal_id": payload.deal_id or inputs.get("deal_id"),
            "deal_path": str(deal_path),
            "rag_api_url": _rag_api_url(),
            "indexed": indexed,
            "skipped": skipped,
            "blocked_secrets": blocked,
            "errors": errors,
            "manifest_path": str(manifest_path),
        }

        return ExecutionResult(outputs=outputs, artifacts=artifacts)
