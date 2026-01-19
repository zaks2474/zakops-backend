from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from actions.engine.models import ActionError, ActionPayload, ArtifactMetadata, now_utc_iso
from actions.executors.base import ActionExecutionError, ActionExecutor, ExecutionContext, ExecutionResult


def _dataroom_root() -> Path:
    return Path(os.getenv("DATAROOM_ROOT", "/home/zaks/DataRoom")).resolve()


def _clean_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", (text or "").strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned[:120] or "item"


def _safe_url(url: str) -> str:
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except Exception:
        return (url or "").split("?", 1)[0].split("#", 1)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Link Classification for UI (tracking, unsubscribe, social, etc.)
# ─────────────────────────────────────────────────────────────────────────────

_TRACKING_DOMAINS = {
    "hubspotlinks.com", "hubspotlinksstarter.com", "hubspotstarter.net",
    "list-manage.com", "mailchi.mp", "click.mailchimp.com",
    "sendgrid.net", "pardot.com", "mktossl.com", "activehosted.com",
    "safelinks.protection.outlook.com",
}

_TRACKING_PATTERNS = [
    re.compile(r"^[a-z0-9]+\.na\d*\.hubspotlinks", re.IGNORECASE),
    re.compile(r"^click\.", re.IGNORECASE),
    re.compile(r"^trk\.", re.IGNORECASE),
    re.compile(r"^track\.", re.IGNORECASE),
]

_UNSUBSCRIBE_PATTERNS = [
    re.compile(r"\bunsubscribe\b", re.IGNORECASE),
    re.compile(r"\bpreferences?\b", re.IGNORECASE),
    re.compile(r"\bopt[_-]?out\b", re.IGNORECASE),
    re.compile(r"/hs/manage-preferences/", re.IGNORECASE),
]

_SOCIAL_DOMAINS = {"linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com"}


def _classify_link_category(url: str) -> str:
    """Classify link into category for UI grouping."""
    try:
        from urllib.parse import urlparse
        domain = (urlparse(url).hostname or "").lower()
    except Exception:
        domain = ""

    # Check tracking domains
    for td in _TRACKING_DOMAINS:
        if domain == td or domain.endswith("." + td):
            return "tracking"
    for pat in _TRACKING_PATTERNS:
        if pat.match(domain):
            return "tracking"

    # Check unsubscribe
    url_lower = url.lower()
    for pat in _UNSUBSCRIBE_PATTERNS:
        if pat.search(url_lower):
            return "unsubscribe"

    # Check social
    for sd in _SOCIAL_DOMAINS:
        if domain == sd or domain.endswith("." + sd):
            return "social"

    return "material"  # Default: potentially deal-related


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _find_existing_bundle(correspondence_dir: Path, message_id: str) -> Optional[Path]:
    suffix = _clean_component(message_id)[-12:]
    if not suffix:
        return None
    for existing in correspondence_dir.iterdir():
        if not existing.is_dir():
            continue
        if suffix not in existing.name:
            continue
        manifest = existing / "manifest.json"
        if not manifest.exists() or not manifest.is_file():
            continue
        data = _read_json(manifest)
        if isinstance(data, dict) and str(data.get("message_id") or "").strip() == message_id:
            return existing
    return None


def _copy_quarantine_payload(*, quarantine_dir: Path, dest_dir: Path) -> Tuple[int, int, List[str]]:
    copied = 0
    skipped = 0
    copied_paths: List[str] = []

    if not quarantine_dir.exists() or not quarantine_dir.is_dir():
        return (0, 0, [])

    dest_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(quarantine_dir.iterdir()):
        if not src.is_file():
            skipped += 1
            continue
        if src.name in {"email_body.txt", "email.json"}:
            skipped += 1
            continue
        try:
            dst = dest_dir / src.name
            shutil.copy2(src, dst)
            copied += 1
            copied_paths.append(str(dst))
        except Exception:
            skipped += 1

    return (copied, skipped, copied_paths)


class AppendEmailMaterialsExecutor(ActionExecutor):
    """
    DEAL.APPEND_EMAIL_MATERIALS

    Append-only correspondence bundle creation for follow-up emails mapped to an existing deal.
    """

    action_type = "DEAL.APPEND_EMAIL_MATERIALS"

    def validate(self, payload: ActionPayload) -> tuple[bool, Optional[str]]:
        inputs = payload.inputs or {}
        deal_id = str(inputs.get("deal_id") or payload.deal_id or "").strip()
        message_id = str(inputs.get("message_id") or "").strip()
        subject = str(inputs.get("subject") or "").strip()
        if not deal_id:
            return False, "Missing required deal_id (inputs.deal_id or payload.deal_id)"
        if not message_id:
            return False, "Missing required inputs.message_id"
        if not subject:
            return False, "Missing required inputs.subject"
        return True, None

    def execute(self, payload: ActionPayload, ctx: ExecutionContext) -> ExecutionResult:
        inputs = payload.inputs or {}

        deal_id = str(inputs.get("deal_id") or payload.deal_id or "").strip()
        message_id = str(inputs.get("message_id") or "").strip()
        thread_id = str(inputs.get("thread_id") or "").strip()

        email_from = str(inputs.get("from") or "").strip()
        email_to = str(inputs.get("to") or "").strip()
        email_date = str(inputs.get("date") or "").strip()
        email_subject = str(inputs.get("subject") or "").strip()

        if not deal_id:
            raise ActionExecutionError(
                ActionError(
                    code="invalid_inputs",
                    message="deal_id is required (inputs.deal_id or payload.deal_id)",
                    category="validation",
                    retryable=False,
                )
            )
        if not message_id:
            raise ActionExecutionError(
                ActionError(
                    code="invalid_inputs",
                    message="inputs.message_id is required",
                    category="validation",
                    retryable=False,
                )
            )

        registry = getattr(ctx, "registry", None)
        deal_obj = registry.get_deal(deal_id) if registry else None
        deal_folder = str(getattr(deal_obj, "folder_path", "") or "").strip() if deal_obj else ""
        if not deal_folder:
            raise ActionExecutionError(
                ActionError(
                    code="deal_not_found",
                    message="Deal not found or missing folder_path (append materials cannot create deals)",
                    category="validation",
                    retryable=False,
                    details={"deal_id": deal_id},
                )
            )

        deal_folder_path = Path(deal_folder).expanduser()
        deal_path = (_dataroom_root() / deal_folder_path).resolve() if not deal_folder_path.is_absolute() else deal_folder_path.resolve()
        _enforce_under_dataroom(deal_path, code="deal_path_outside_dataroom")

        correspondence_dir = deal_path / "07-Correspondence"
        correspondence_dir.mkdir(parents=True, exist_ok=True)

        existing = _find_existing_bundle(correspondence_dir, message_id)
        if existing:
            bundle_dir = existing
            deduped = True
        else:
            base = f"{_utc_timestamp()}_{_clean_component(message_id)[-12:]}"
            bundle_dir = correspondence_dir / base
            counter = 2
            while bundle_dir.exists():
                bundle_dir = correspondence_dir / f"{base}_{counter}"
                counter += 1
            bundle_dir.mkdir(parents=True, exist_ok=False)
            deduped = False

        email_md_path = bundle_dir / "email.md"
        email_json_path = bundle_dir / "email.json"
        manifest_path = bundle_dir / "manifest.json"
        links_path = bundle_dir / "links.json"
        pending_auth_links_path = bundle_dir / "pending_auth_links.json"
        attachments_dir = bundle_dir / "attachments"

        quarantine_dir_raw = str(inputs.get("quarantine_dir") or "").strip()
        quarantine_dir = Path(quarantine_dir_raw).expanduser().resolve() if quarantine_dir_raw else None
        if quarantine_dir is None:
            quarantine_dir = (_dataroom_root() / "00-PIPELINE" / "_INBOX_QUARANTINE" / message_id).resolve()
        _enforce_under_dataroom(quarantine_dir, code="quarantine_outside_dataroom")

        email_body = ""
        try:
            body_path = quarantine_dir / "email_body.txt"
            if body_path.exists() and body_path.is_file():
                email_body = body_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            email_body = ""

        # Materialize bundle files only on first creation (append-only).
        copied = 0
        skipped = 0
        copied_paths: List[str] = []
        if not deduped:
            md = "\n".join(
                [
                    "# Email (Appended)",
                    "",
                    f"- Message ID: {message_id}",
                    f"- Thread ID: {thread_id}",
                    f"- From: {email_from}",
                    f"- To: {email_to}",
                    f"- Date: {email_date}",
                    f"- Subject: {email_subject}",
                    "",
                    "---",
                    "",
                    (email_body or "").strip(),
                    "",
                ]
            )
            email_md_path.write_text(md, encoding="utf-8")

            email_json = {
                "message_id": message_id,
                "thread_id": thread_id,
                "from": email_from,
                "to": email_to,
                "date": email_date,
                "subject": email_subject,
                "body": email_body,
            }
            _write_json(email_json_path, email_json)

            links_in = inputs.get("links") or []
            links: List[Dict[str, Any]] = []
            for l in links_in if isinstance(links_in, list) else []:
                if not isinstance(l, dict):
                    continue
                url = str(l.get("url") or "").strip()
                if not url:
                    continue
                links.append(
                    {
                        "type": str(l.get("type") or "other"),
                        "url": _safe_url(url),
                        "auth_required": bool(l.get("auth_required")),
                        "vendor_hint": l.get("vendor_hint"),
                    }
                )

            attachments_in = inputs.get("attachments") or []
            attachments_meta: List[Dict[str, Any]] = []
            for a in attachments_in if isinstance(attachments_in, list) else []:
                if not isinstance(a, dict):
                    continue
                attachments_meta.append(
                    {
                        "filename": a.get("filename"),
                        "mime_type": a.get("mime_type"),
                        "size_bytes": a.get("size_bytes"),
                    }
                )

            manifest: Dict[str, Any] = {
                "deal_id": deal_id,
                "message_id": message_id,
                "thread_id": thread_id,
                "from": email_from,
                "to": email_to,
                "date": email_date,
                "subject": email_subject,
                "classification": inputs.get("classification"),
                "urgency": inputs.get("urgency"),
                "confidence": inputs.get("confidence"),
                "quarantine_dir": str(quarantine_dir),
                "links": links,
                "attachments": attachments_meta,
                "generated_at": now_utc_iso(),
            }
            _write_json(manifest_path, manifest)

            _write_json(links_path, {"links": links, "generated_at": now_utc_iso()})
            _write_json(
                pending_auth_links_path,
                {"links": [l for l in links if bool(l.get("auth_required"))], "generated_at": now_utc_iso()},
            )

            copied, skipped, copied_paths = _copy_quarantine_payload(quarantine_dir=quarantine_dir, dest_dir=attachments_dir)

        # Update deal-level aggregate links (append-only dedup by url).
        # Now includes classification for UI grouping (tracking, unsubscribe, social, material)
        agg_path = correspondence_dir / "links.json"
        agg_payload = _read_json(agg_path)
        agg_links: List[Dict[str, Any]] = []
        if isinstance(agg_payload, dict) and isinstance(agg_payload.get("links"), list):
            agg_links = [l for l in agg_payload.get("links") if isinstance(l, dict)]
        seen_urls = {str(l.get("url") or "") for l in agg_links if str(l.get("url") or "").strip()}

        current_manifest = _read_json(manifest_path)
        bundle_links = []
        if isinstance(current_manifest, dict) and isinstance(current_manifest.get("links"), list):
            bundle_links = [l for l in current_manifest.get("links") if isinstance(l, dict)]
        for l in bundle_links:
            url = str(l.get("url") or "").strip()
            if not url:
                continue
            url = _safe_url(url)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            category = _classify_link_category(url)
            agg_links.append(
                {
                    "url": url,
                    "type": l.get("type") or "other",
                    "category": category,  # tracking | unsubscribe | social | material
                    "auth_required": bool(l.get("auth_required")),
                    "vendor_hint": l.get("vendor_hint"),
                    "source": {"message_id": message_id, "bundle": bundle_dir.name},
                    "added_at": now_utc_iso(),
                }
            )

        # Separate material links (for display) from tracking/unsubscribe/social (collapsed)
        material_links = [l for l in agg_links if l.get("category") == "material"]
        tracking_links = [l for l in agg_links if l.get("category") == "tracking"]
        unsubscribe_links = [l for l in agg_links if l.get("category") == "unsubscribe"]
        social_links = [l for l in agg_links if l.get("category") == "social"]

        agg_output = {
            "deal_id": deal_id,
            "updated_at": now_utc_iso(),
            "links": material_links,  # Primary: only material links shown by default
            "summary": {
                "material_count": len(material_links),
                "tracking_count": len(tracking_links),
                "unsubscribe_count": len(unsubscribe_links),
                "social_count": len(social_links),
                "total_raw": len(agg_links),
            },
            # Collapsed sections (for audit/debug, not shown by default)
            "_tracking_links": tracking_links,
            "_unsubscribe_links": unsubscribe_links,
            "_social_links": social_links,
            "_all_links_raw": agg_links,  # Full audit trail
        }
        _write_json(agg_path, agg_output)

        # Build next_actions for Phase 6 chaining.
        artifact_paths: List[str] = [str(email_md_path)]
        if attachments_dir.exists():
            for p in sorted(attachments_dir.iterdir()):
                if p.is_file():
                    artifact_paths.append(str(p))

        next_actions = [
            {
                "action_type": "DEAL.EXTRACT_EMAIL_ARTIFACTS",
                "capability_id": "deal.extract_email_artifacts.v1",
                "title": "Extract entities + doc types from appended email artifacts",
                "inputs": {"deal_id": deal_id, "deal_path": str(deal_path), "artifact_paths": artifact_paths},
                "requires_approval": False,
                "idempotency_key": f"extract_email_artifacts:{message_id}",
            },
            {
                "action_type": "DEAL.ENRICH_MATERIALS",
                "capability_id": "deal.enrich_materials.v1",
                "title": "Enrich appended email materials (local)",
                "inputs": {"deal_id": deal_id, "deal_path": str(deal_path), "artifact_paths": artifact_paths, "bundle_path": str(bundle_dir)},
                "requires_approval": False,
                "idempotency_key": f"enrich_materials:{message_id}",
            },
            {
                "action_type": "DEAL.DEDUPE_AND_PLACE_MATERIALS",
                "capability_id": "deal.dedupe_and_place_materials.v1",
                "title": "Place attachments into deal folders (derived views)",
                "inputs": {"deal_id": deal_id, "deal_path": str(deal_path), "bundle_path": str(bundle_dir)},
                "requires_approval": False,
                "idempotency_key": f"dedupe_and_place_materials:{message_id}",
            },
            {
                "action_type": "RAG.REINDEX_DEAL",
                "capability_id": "rag.reindex_deal.v1",
                "title": "Reindex appended email artifacts into RAG",
                "inputs": {"deal_id": deal_id, "deal_path": str(deal_path), "artifact_paths": artifact_paths, "bundle_path": str(bundle_dir)},
                "requires_approval": False,
                "idempotency_key": f"rag_reindex_deal:{message_id}",
            },
        ]

        outputs: Dict[str, Any] = {
            "deal_id": deal_id,
            "deal_path": str(deal_path),
            "message_id": message_id,
            "thread_id": thread_id,
            "bundle_path": str(bundle_dir),
            "deduplicated": bool(deduped),
            "attachments": {"copied": copied, "skipped": skipped, "dir": str(attachments_dir), "copied_paths": copied_paths},
            "aggregate_links_path": str(agg_path),
            "next_actions": next_actions,
        }

        artifacts = [
            ArtifactMetadata(filename=email_md_path.name, mime_type="text/markdown", path=str(email_md_path), created_at=now_utc_iso()),
            ArtifactMetadata(filename=manifest_path.name, mime_type="application/json", path=str(manifest_path), created_at=now_utc_iso()),
            ArtifactMetadata(filename=links_path.name, mime_type="application/json", path=str(links_path), created_at=now_utc_iso()),
            ArtifactMetadata(
                filename=pending_auth_links_path.name,
                mime_type="application/json",
                path=str(pending_auth_links_path),
                created_at=now_utc_iso(),
            ),
        ]

        return ExecutionResult(outputs=outputs, artifacts=artifacts)
