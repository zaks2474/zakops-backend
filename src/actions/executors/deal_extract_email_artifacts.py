from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from actions.engine.models import ActionError, ActionPayload, ArtifactMetadata, now_utc_iso
from actions.executors._artifacts import resolve_action_artifact_dir
from actions.executors.base import ActionExecutionError, ActionExecutor, ExecutionContext, ExecutionResult


_MONEY_RE = re.compile(
    r"(?P<prefix>\\$)?(?P<num>\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?|\\d+(?:\\.\\d+)?)\\s*(?P<scale>mm|m|million|k|thousand|b|bn|billion)?",
    re.IGNORECASE,
)
_METRIC_RE = re.compile(r"\\b(ebitda|arr|revenue|sde|profit|cash flow|ltm)\\b", re.IGNORECASE)


def _safe_read_text(path: Path, *, max_bytes: int) -> str:
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


def _detect_doc_type(filename: str) -> Tuple[str, float]:
    name = (filename or "").lower()
    if "nda" in name or "non-disclosure" in name or "confidentiality" in name:
        return "nda", 0.9
    if "cim" in name or "confidential information memorandum" in name or "investment memorandum" in name:
        return "cim", 0.9
    if "teaser" in name or "one-pager" in name or "onepager" in name or "executive summary" in name:
        return "teaser", 0.85
    if any(k in name for k in ["financial", "p&l", "pnl", "income", "balance", "cash flow", "ltm", "t12", "trial balance"]):
        return "financials", 0.8
    if "qofe" in name or "quality of earnings" in name:
        return "qofe", 0.8
    if any(name.endswith(ext) for ext in [".xls", ".xlsx", ".csv"]):
        return "spreadsheet", 0.7
    if name.endswith(".pdf"):
        return "pdf", 0.5
    return "other", 0.3


def _extract_numbers(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not text:
        return out

    for m in _MONEY_RE.finditer(text):
        num = (m.group("num") or "").strip()
        scale = (m.group("scale") or "").strip().lower() or None
        if not num:
            continue
        # Avoid tiny noise (e.g., "2025" in dates) unless it's prefixed with "$" or scaled.
        if not (m.group("prefix") or scale) and len(num.replace(",", "")) <= 4:
            continue
        out.append({"raw": m.group(0).strip(), "number": num, "scale": scale})
        if len(out) >= 50:
            break
    return out


def _extract_metrics(text: str) -> List[str]:
    if not text:
        return []
    metrics = set(m.group(1).lower() for m in _METRIC_RE.finditer(text))
    return sorted(metrics)


@dataclass(frozen=True)
class ArtifactSignal:
    path: str
    filename: str
    doc_type: str
    confidence: float
    text_preview: str = ""


class ExtractEmailArtifactsExecutor(ActionExecutor):
    """
    DEAL.EXTRACT_EMAIL_ARTIFACTS

    Local-only, best-effort extraction to produce:
    - extracted_summary.md
    - extracted_entities.json
    - detected_doc_types.json
    """

    action_type = "DEAL.EXTRACT_EMAIL_ARTIFACTS"

    def validate(self, payload: ActionPayload) -> tuple[bool, Optional[str]]:
        inputs = payload.inputs or {}
        artifact_paths = inputs.get("artifact_paths")
        if not isinstance(artifact_paths, list) or not artifact_paths:
            return False, "Missing required inputs.artifact_paths (list)"
        return True, None

    def execute(self, payload: ActionPayload, ctx: ExecutionContext) -> ExecutionResult:
        inputs = payload.inputs or {}
        max_read_bytes = int(inputs.get("max_read_bytes") or 200_000)

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

        artifact_paths_raw = inputs.get("artifact_paths") or []
        artifact_paths: List[Path] = []
        for raw in artifact_paths_raw:
            if not isinstance(raw, str) or not raw.strip():
                continue
            artifact_paths.append(Path(raw).expanduser().resolve())

        signals: List[ArtifactSignal] = []
        combined_text = ""

        for ap in artifact_paths:
            if not ap.exists() or not ap.is_file():
                continue
            doc_type, conf = _detect_doc_type(ap.name)

            text_preview = ""
            if ap.suffix.lower() in {".md", ".txt", ".csv"}:
                text_preview = _safe_read_text(ap, max_bytes=max_read_bytes).strip()
            # Keep previews bounded
            if text_preview and len(text_preview) > 2000:
                text_preview = text_preview[:2000] + "…"

            if text_preview:
                combined_text += "\n" + text_preview

            signals.append(
                ArtifactSignal(
                    path=str(ap),
                    filename=ap.name,
                    doc_type=doc_type,
                    confidence=float(conf),
                    text_preview=text_preview,
                )
            )

        numbers = _extract_numbers(combined_text)
        metrics = _extract_metrics(combined_text)

        out_dir = resolve_action_artifact_dir(ctx)
        summary_path = out_dir / "extracted_summary.md"
        entities_path = out_dir / "extracted_entities.json"
        doc_types_path = out_dir / "detected_doc_types.json"

        # Build detected doc types payload
        doc_types_payload = {
            "deal_id": payload.deal_id or inputs.get("deal_id"),
            "action_id": payload.action_id,
            "generated_at": now_utc_iso(),
            "artifacts": [
                {
                    "path": s.path,
                    "filename": s.filename,
                    "doc_type": s.doc_type,
                    "confidence": s.confidence,
                }
                for s in signals
            ],
        }

        entities_payload = {
            "deal_id": payload.deal_id or inputs.get("deal_id"),
            "action_id": payload.action_id,
            "generated_at": now_utc_iso(),
            "metrics_detected": metrics,
            "numbers_detected": numbers,
            "notes": "Best-effort local extraction (text previews only for md/txt/csv).",
        }

        summary_lines = [
            "# Extraction Summary (Local)",
            "",
            f"- **deal_id:** `{payload.deal_id or inputs.get('deal_id') or ''}`",
            f"- **deal_path:** `{deal_path}`",
            f"- **artifacts_scanned:** `{len(signals)}`",
            f"- **metrics_detected:** `{', '.join(metrics) if metrics else ''}`",
            f"- **numbers_detected:** `{len(numbers)}`",
            "",
            "## Detected Doc Types",
            "",
        ]
        for s in signals:
            summary_lines.append(f"- `{s.filename}` → **{s.doc_type}** (confidence {s.confidence:.2f})")
        summary_lines.append("")
        summary_lines.append("_Generated by ZakOps (deal_extract_email_artifacts)._\n")

        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
        entities_path.write_text(json.dumps(entities_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        doc_types_path.write_text(json.dumps(doc_types_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        artifacts = [
            ArtifactMetadata(filename=summary_path.name, mime_type="text/markdown", path=str(summary_path), created_at=now_utc_iso()),
            ArtifactMetadata(filename=entities_path.name, mime_type="application/json", path=str(entities_path), created_at=now_utc_iso()),
            ArtifactMetadata(filename=doc_types_path.name, mime_type="application/json", path=str(doc_types_path), created_at=now_utc_iso()),
        ]

        outputs: Dict[str, Any] = {
            "deal_id": payload.deal_id or inputs.get("deal_id"),
            "deal_path": str(deal_path),
            "artifacts_scanned": len(signals),
            "metrics_detected": metrics,
            "numbers_detected_count": len(numbers),
            "outputs": {
                "extracted_summary_md": str(summary_path),
                "extracted_entities_json": str(entities_path),
                "detected_doc_types_json": str(doc_types_path),
            },
        }

        return ExecutionResult(outputs=outputs, artifacts=artifacts)
