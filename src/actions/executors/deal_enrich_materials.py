from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from actions.engine.models import ActionError, ActionPayload, now_utc_iso
from actions.executors._artifacts import resolve_action_artifact_dir
from actions.executors.base import ActionExecutionError, ActionExecutor, ExecutionContext, ExecutionResult
from integrations.n8n_webhook import emit_auth_required_links_detected


def _dataroom_root() -> Path:
    return Path(os.getenv("DATAROOM_ROOT", "/home/zaks/DataRoom")).resolve()


def _safe_url(url: str) -> str:
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except Exception:
        return (url or "").split("?", 1)[0].split("#", 1)[0]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class LinkRecord:
    url: str
    type: str
    auth_required: bool
    vendor_hint: Optional[str] = None


def _extract_links_from_inputs(inputs: Dict[str, Any]) -> List[LinkRecord]:
    links_in = inputs.get("links")
    out: List[LinkRecord] = []
    if isinstance(links_in, list):
        for item in links_in:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            out.append(
                LinkRecord(
                    url=_safe_url(url),
                    type=str(item.get("type") or "other"),
                    auth_required=bool(item.get("auth_required", True)),
                    vendor_hint=str(item.get("vendor_hint") or "").strip() or None,
                )
            )
    return out


def _extract_links_from_artifacts(artifact_paths: List[Path]) -> List[LinkRecord]:
    links: List[LinkRecord] = []
    for p in artifact_paths:
        if not p.exists() or not p.is_file():
            continue
        if p.suffix.lower() != ".json":
            continue
        data = _load_json(p)
        if not isinstance(data, dict):
            continue
        raw_links = data.get("links")
        if not isinstance(raw_links, list):
            continue
        for item in raw_links:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            links.append(
                LinkRecord(
                    url=_safe_url(url),
                    type=str(item.get("type") or "other"),
                    auth_required=bool(item.get("auth_required", True)),
                    vendor_hint=str(item.get("vendor_hint") or "").strip() or None,
                )
            )
    return links


def _write_link_intake_queue(*, deal_id: Optional[str], links: List[LinkRecord], source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Append auth-required links into a global intake queue for operator follow-up.
    """
    queue_path = _dataroom_root() / ".deal-registry" / "link_intake_queue.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_json(queue_path)
    items: List[Dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        items = [x for x in payload.get("items") if isinstance(x, dict)]

    existing = {(str(i.get("deal_id") or ""), str(i.get("url") or "")) for i in items}
    added = 0
    for l in links:
        if not l.auth_required:
            continue
        key = (str(deal_id or ""), l.url)
        if key in existing:
            continue
        existing.add(key)
        items.append(
            {
                "deal_id": deal_id,
                "url": l.url,
                "type": l.type,
                "auth_required": True,
                "vendor_hint": l.vendor_hint,
                "source": source,
                "added_at": now_utc_iso(),
            }
        )
        added += 1

    _write_json(queue_path, {"updated_at": now_utc_iso(), "items": items})
    return {"queue_path": str(queue_path), "added": added, "total": len(items)}


class EnrichMaterialsExecutor(ActionExecutor):
    """
    DEAL.ENRICH_MATERIALS

    Local-only enrichment pass for a deal's artifacts.

    - Delegates to DEAL.EXTRACT_EMAIL_ARTIFACTS for deterministic extraction.
    - Extracts link inventories (inputs + bundle manifests) and writes auth-required links into
      DataRoom/.deal-registry/link_intake_queue.json for operator follow-up.
    - Optional: best-effort download of public links when enabled via env flag.
    """

    action_type = "DEAL.ENRICH_MATERIALS"

    def validate(self, payload: ActionPayload) -> tuple[bool, Optional[str]]:
        inputs = payload.inputs or {}
        artifact_paths = inputs.get("artifact_paths")
        if not isinstance(artifact_paths, list) or not artifact_paths:
            return False, "Missing required inputs.artifact_paths (list)"
        return True, None

    def execute(self, payload: ActionPayload, ctx: ExecutionContext) -> ExecutionResult:
        from actions.executors.deal_extract_email_artifacts import ExtractEmailArtifactsExecutor

        inputs = payload.inputs or {}
        artifact_paths_raw = inputs.get("artifact_paths") or []
        artifact_paths: List[Path] = []
        for raw in artifact_paths_raw if isinstance(artifact_paths_raw, list) else []:
            if isinstance(raw, str) and raw.strip():
                artifact_paths.append(Path(raw).expanduser().resolve())

        # Link extraction sources:
        # - explicit inputs.links
        # - any manifest/links json files included in artifact_paths
        links: List[LinkRecord] = []
        links.extend(_extract_links_from_inputs(inputs))
        links.extend(_extract_links_from_artifacts(artifact_paths))

        # Optional: bundle_path can be used to read links.json/manifest.json without requiring the caller
        # to pass those paths explicitly.
        bundle_path_raw = str(inputs.get("bundle_path") or "").strip()
        if bundle_path_raw:
            bundle_dir = Path(bundle_path_raw).expanduser().resolve()
            try:
                bundle_dir.relative_to(_dataroom_root())
            except ValueError:
                bundle_dir = None  # type: ignore[assignment]
            if bundle_dir and bundle_dir.exists() and bundle_dir.is_dir():
                for name in ("manifest.json", "links.json"):
                    links.extend(_extract_links_from_artifacts([bundle_dir / name]))

        # Dedup links by url
        deduped: List[LinkRecord] = []
        seen = set()
        for l in links:
            if not l.url or l.url in seen:
                continue
            seen.add(l.url)
            deduped.append(l)

        # Run deterministic extraction pass.
        result = ExtractEmailArtifactsExecutor().execute(payload, ctx)
        outputs = dict(result.outputs or {})
        outputs.setdefault("enrichment_kind", "local_extract_v2")

        # Record auth-required links into global intake queue.
        source = {"action_id": payload.action_id, "action_type": payload.type}
        if bundle_path_raw:
            source["bundle_path"] = bundle_path_raw
        queue_info = _write_link_intake_queue(deal_id=payload.deal_id or inputs.get("deal_id"), links=deduped, source=source)
        outputs["link_intake_queue"] = queue_info
        outputs["links_detected"] = len(deduped)

        # Optional n8n integration (off by default unless ZAKOPS_N8N_WEBHOOK_URL is set).
        try:
            auth_required = sum(1 for l in deduped if l.auth_required)
            if auth_required:
                emit_auth_required_links_detected(deal_id=str(payload.deal_id or inputs.get("deal_id") or ""), count=auth_required)
        except Exception:
            pass

        # Optional: download public links (best-effort; disabled by default).
        downloads: List[Dict[str, Any]] = []
        allow_downloads = (os.getenv("ZAKOPS_ENRICH_DOWNLOAD_PUBLIC_LINKS") or "").strip().lower() in {"1", "true", "yes", "on"}
        if allow_downloads:
            out_dir = resolve_action_artifact_dir(ctx) / "link_snapshots"
            out_dir.mkdir(parents=True, exist_ok=True)
            for l in deduped:
                if l.auth_required:
                    continue
                try:
                    resp = requests.get(l.url, timeout=15)
                    if resp.status_code != 200:
                        downloads.append({"url": l.url, "status": "error", "reason": f"http_{resp.status_code}"})
                        continue
                    fname = (l.type or "link").lower().replace("/", "_")
                    safe = fname if fname else "link"
                    path = out_dir / f"{safe}_{len(downloads)+1}.html"
                    path.write_text(resp.text[:300_000], encoding="utf-8", errors="replace")
                    downloads.append({"url": l.url, "status": "downloaded", "path": str(path)})
                except Exception as e:
                    downloads.append({"url": l.url, "status": "error", "reason": f"{type(e).__name__}"})

        if downloads:
            outputs["public_link_snapshots"] = downloads

        return ExecutionResult(outputs=outputs, artifacts=result.artifacts or [])
