from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from actions.engine.models import ActionError, ActionPayload, ArtifactMetadata, now_utc_iso
from actions.executors.base import ActionExecutionError, ActionExecutor, ExecutionContext, ExecutionResult


def _dataroom_root() -> Path:
    return Path(os.getenv("DATAROOM_ROOT", "/home/zaks/DataRoom")).resolve()


def _enforce_under_dataroom(path: Path, *, code: str = "path_outside_dataroom") -> None:
    try:
        path.relative_to(_dataroom_root())
    except ValueError as e:
        raise ActionExecutionError(
            ActionError(
                code=code,
                message="Path is outside DATAROOM_ROOT",
                category="validation",
                retryable=False,
                details={"path": str(path), "error": str(e)},
            )
        )


def _safe_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip())
    cleaned = re.sub(r"_{2,}", "_", cleaned).strip("_")
    return cleaned[:160] or "file"


def _sha256_path(path: Path, *, max_bytes: int = 50 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                break
            h.update(chunk)
    return h.hexdigest()


def _detect_doc_bucket(filename: str) -> str:
    name = (filename or "").lower()
    if "nda" in name or "non-disclosure" in name or "confidentiality" in name:
        return "nda"
    if "cim" in name or "confidential information memorandum" in name or "investment memorandum" in name:
        return "cim"
    if "teaser" in name or "one-pager" in name or "onepager" in name or "executive summary" in name:
        return "teaser"
    if any(k in name for k in ["financial", "p&l", "pnl", "income", "balance", "cash flow", "trial balance", "t12", "ltm"]):
        return "financials"
    if "qofe" in name or "quality of earnings" in name:
        return "qofe"
    if name.endswith((".xls", ".xlsx", ".csv")):
        return "financials"
    return "other"


DEST_BY_BUCKET = {
    "nda": "01-NDA",
    "cim": "02-CIM",
    "teaser": "02-CIM",
    "financials": "03-Financials",
    "qofe": "06-Analysis",
}


SAFE_EXTS = {
    "pdf",
    "doc",
    "docx",
    "rtf",
    "txt",
    "xls",
    "xlsx",
    "csv",
    "ppt",
    "pptx",
    "zip",
}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class DedupeAndPlaceMaterialsExecutor(ActionExecutor):
    """
    DEAL.DEDUPE_AND_PLACE_MATERIALS

    Derived view builder:
    - Reads a correspondence bundle's `attachments/` and copies safe artifacts into
      top-level deal folders (01-NDA/02-CIM/03-Financials/...) using filename heuristics.
    - Leaves originals intact under the correspondence bundle (audit/provenance).
    - Idempotent via `placed_materials.json` inside the bundle directory.
    """

    action_type = "DEAL.DEDUPE_AND_PLACE_MATERIALS"

    def validate(self, payload: ActionPayload) -> tuple[bool, Optional[str]]:
        inputs = payload.inputs or {}
        bundle_path = str(inputs.get("bundle_path") or "").strip()
        if not bundle_path:
            return False, "Missing required inputs.bundle_path"
        return True, None

    def execute(self, payload: ActionPayload, ctx: ExecutionContext) -> ExecutionResult:
        inputs = payload.inputs or {}

        # Resolve deal folder (prefer ctx.deal).
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

        deal_path = Path(deal_folder).expanduser().resolve()
        _enforce_under_dataroom(deal_path, code="deal_outside_dataroom")

        bundle_dir = Path(str(inputs.get("bundle_path") or "")).expanduser().resolve()
        _enforce_under_dataroom(bundle_dir, code="bundle_outside_dataroom")
        if not bundle_dir.exists() or not bundle_dir.is_dir():
            raise ActionExecutionError(
                ActionError(
                    code="bundle_missing",
                    message="bundle_path does not exist or is not a directory",
                    category="validation",
                    retryable=False,
                    details={"bundle_path": str(bundle_dir)},
                )
            )

        attachments_dir = bundle_dir / "attachments"
        if not attachments_dir.exists() or not attachments_dir.is_dir():
            return ExecutionResult(
                outputs={
                    "deal_id": payload.deal_id or inputs.get("deal_id"),
                    "deal_path": str(deal_path),
                    "bundle_path": str(bundle_dir),
                    "placed": 0,
                    "skipped": 0,
                    "reason": "no_attachments_dir",
                },
                artifacts=[],
            )

        placements_path = bundle_dir / "placed_materials.json"
        existing = _load_json(placements_path)
        placements: List[Dict[str, Any]] = []
        if isinstance(existing, dict) and isinstance(existing.get("placements"), list):
            placements = [p for p in existing.get("placements") if isinstance(p, dict)]

        placed_by_src = {str(p.get("source_rel") or ""): p for p in placements if str(p.get("source_rel") or "").strip()}

        placed = 0
        skipped = 0
        updated = False

        for src in sorted(attachments_dir.iterdir()):
            if not src.is_file():
                continue
            src_rel = f"attachments/{src.name}"
            if src_rel in placed_by_src:
                dst_existing = str(placed_by_src[src_rel].get("dest_rel") or "").strip()
                if dst_existing and (deal_path / dst_existing).exists():
                    skipped += 1
                    continue

            ext = src.suffix.lower().lstrip(".")
            if ext and ext not in SAFE_EXTS:
                skipped += 1
                continue

            bucket = _detect_doc_bucket(src.name)
            dest_root_rel = DEST_BY_BUCKET.get(bucket) or "07-Correspondence/Attachments"
            dest_dir = deal_path / dest_root_rel
            dest_dir.mkdir(parents=True, exist_ok=True)

            sha = _sha256_path(src)
            base_name = _safe_component(src.name)
            dest = dest_dir / base_name
            if dest.exists():
                try:
                    if _sha256_path(dest) == sha:
                        placements.append(
                            {
                                "source_rel": src_rel,
                                "dest_rel": str(Path(dest_root_rel) / dest.name),
                                "bucket": bucket,
                                "sha256": sha,
                                "placed_at": now_utc_iso(),
                                "deduped": True,
                            }
                        )
                        updated = True
                        skipped += 1
                        continue
                except Exception:
                    pass
                stem = dest.stem
                suffix = dest.suffix
                dest = dest_dir / f"{stem}--{sha[:8]}{suffix}"
                if dest.exists():
                    counter = 2
                    while True:
                        candidate = dest_dir / f"{stem}--{sha[:8]}-{counter}{suffix}"
                        if not candidate.exists():
                            dest = candidate
                            break
                        counter += 1

            shutil.copy2(src, dest)
            placements.append(
                {
                    "source_rel": src_rel,
                    "dest_rel": str(Path(dest_root_rel) / dest.name),
                    "bucket": bucket,
                    "sha256": sha,
                    "placed_at": now_utc_iso(),
                    "deduped": False,
                }
            )
            updated = True
            placed += 1

        if updated:
            _write_json(
                placements_path,
                {
                    "deal_id": payload.deal_id or inputs.get("deal_id"),
                    "bundle_path": str(bundle_dir),
                    "updated_at": now_utc_iso(),
                    "placements": placements,
                },
            )

        outputs: Dict[str, Any] = {
            "deal_id": payload.deal_id or inputs.get("deal_id"),
            "deal_path": str(deal_path),
            "bundle_path": str(bundle_dir),
            "placements_path": str(placements_path),
            "placed": placed,
            "skipped": skipped,
        }

        artifacts: List[ArtifactMetadata] = []
        if updated and placements_path.exists():
            artifacts.append(
                ArtifactMetadata(
                    filename=placements_path.name,
                    mime_type="application/json",
                    path=str(placements_path),
                    created_at=now_utc_iso(),
                )
            )

        return ExecutionResult(outputs=outputs, artifacts=artifacts)

