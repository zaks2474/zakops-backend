from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from actions.engine.models import ActionError, ActionPayload, ArtifactMetadata, now_utc_iso
from actions.executors.base import ActionExecutionError, ActionExecutor, ExecutionContext, ExecutionResult


def _dataroom_root() -> Path:
    return Path(os.getenv("DATAROOM_ROOT", "/home/zaks/DataRoom")).resolve()


def _deal_registry_path(dataroom_root: Path) -> Path:
    return dataroom_root / ".deal-registry" / "deal_registry.json"


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")


def _sanitize_filename(name: str) -> str:
    base = (name or "").strip().split("/")[-1].split("\\")[-1]
    base = _SAFE_NAME_RE.sub("_", base).strip()
    return (base[:200] or "file").strip("._ ")


def _title_slug(text: str, *, max_len: int = 80) -> str:
    raw = (text or "").strip()
    raw = re.sub(r"^(re|fw|fwd)\s*:\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"[^A-Za-z0-9]+", " ", raw).strip()
    if not raw:
        return "New-Deal"
    words = [w for w in raw.split() if w]
    titled = "-".join([w[:1].upper() + w[1:].lower() if len(w) > 1 else w.upper() for w in words])
    return titled[:max_len].rstrip("-") or "New-Deal"


def _short_deal_suffix(deal_id: str) -> str:
    parts = [p for p in str(deal_id or "").strip().split("-") if p]
    if len(parts) >= 3:
        return "-".join(parts[-2:])
    return str(deal_id or "").strip() or "unknown"


def _ensure_deal_folder_template(deal_dir: Path) -> None:
    # Minimal, consistent deal skeleton (matches existing DataRoom conventions).
    subdirs = [
        "01-NDA",
        "02-CIM",
        "03-Financials",
        "04-Operations",
        "05-Legal",
        "06-Analysis",
        "07-Correspondence",
        "08-LOI-Offer",
        "09-Closing",
        "99-ACTIONS",
    ]
    for name in subdirs:
        (deal_dir / name).mkdir(parents=True, exist_ok=True)


def _resolve_quarantine_dir(inputs: Dict[str, Any], *, dataroom_root: Path, message_id: str) -> Optional[Path]:
    qdir = str(inputs.get("quarantine_dir") or "").strip()
    if qdir:
        return Path(qdir).expanduser().resolve()
    qroot = str(inputs.get("quarantine_root") or "").strip()
    if not qroot:
        qroot = str(dataroom_root / "00-PIPELINE" / "_INBOX_QUARANTINE")
    return (Path(qroot).expanduser().resolve() / message_id)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_file(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return False
    shutil.copy2(src, dst)
    return True


class CreateDealFromEmailExecutor(ActionExecutor):
    """
    DEAL.CREATE_FROM_EMAIL

    Convert an approved inbound email (with optional quarantine artifacts) into a real deal workspace.

    Idempotency strategy:
    - Primary: DealRegistry email_to_deal mapping (message_id -> deal_id)
    - Secondary: marker file in quarantine_dir/deal_created.json (for crash recovery)
    """

    action_type = "DEAL.CREATE_FROM_EMAIL"

    def validate(self, payload: ActionPayload) -> tuple[bool, Optional[str]]:
        inputs = payload.inputs or {}
        mid = str(inputs.get("gmail_message_id") or inputs.get("message_id") or "").strip()
        subject = str(inputs.get("subject") or "").strip()
        if not mid:
            return False, "Missing required field: gmail_message_id"
        if not subject:
            return False, "Missing required field: subject"
        return True, None

    def execute(self, payload: ActionPayload, ctx: ExecutionContext) -> ExecutionResult:
        inputs = payload.inputs or {}
        dataroom_root = _dataroom_root()

        message_id = str(inputs.get("gmail_message_id") or inputs.get("message_id") or "").strip()
        thread_id = str(inputs.get("gmail_thread_id") or inputs.get("thread_id") or "").strip() or None
        subject = str(inputs.get("subject") or "").strip()
        from_email = str(inputs.get("from_email") or "").strip().lower() or None
        from_header = str(inputs.get("from_header") or inputs.get("from") or "").strip() or None
        received_at = str(inputs.get("received_at") or inputs.get("date") or "").strip() or None
        snippet = str(inputs.get("snippet") or "").strip() or None

        if not message_id or not subject:
            raise ActionExecutionError(
                ActionError(
                    code="validation_failed",
                    message="gmail_message_id and subject are required",
                    category="validation",
                    retryable=False,
                )
            )

        quarantine_dir = _resolve_quarantine_dir(inputs, dataroom_root=dataroom_root, message_id=message_id)
        marker_path = (quarantine_dir / "deal_created.json") if quarantine_dir else None

        # Load Deal Registry
        registry_path = _deal_registry_path(dataroom_root)
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import sys

            if "/home/zaks/scripts" not in sys.path:
                sys.path.insert(0, "/home/zaks/scripts")
            from deal_registry import BrokerInfo, DealRegistry  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ActionExecutionError(
                ActionError(
                    code="deal_registry_import_failed",
                    message=f"Failed to import DealRegistry: {e}",
                    category="dependency",
                    retryable=False,
                )
            )

        registry = DealRegistry(str(registry_path))

        # Determine existing mapping (registry first, marker second).
        deal_id = registry.get_email_deal_mapping(message_id)
        deal_folder_path: Optional[str] = None
        deal_name: Optional[str] = None

        if not deal_id and marker_path and marker_path.exists():
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                deal_id = str(marker.get("deal_id") or "").strip() or None
                deal_folder_path = str(marker.get("deal_path") or "").strip() or None
                deal_name = str(marker.get("deal_name") or "").strip() or None
            except Exception:
                # Marker is best-effort only.
                deal_id = None

        created_new = False

        if deal_id:
            existing = registry.get_deal(deal_id)
            if not existing:
                raise ActionExecutionError(
                    ActionError(
                        code="deal_not_found_for_mapping",
                        message=f"Email is mapped to deal_id={deal_id} but deal not found in registry",
                        category="validation",
                        retryable=False,
                        details={"deal_id": deal_id, "message_id": message_id},
                    )
                )
            deal_folder_path = existing.folder_path
            deal_name = existing.canonical_name or existing.display_name or deal_id
        else:
            inferred_name = str(inputs.get("inferred_company_name") or "").strip()
            canonical_name = inferred_name or subject
            deal_name = canonical_name.strip()[:120] or "New Deal"

            base_dir = dataroom_root / "00-PIPELINE" / "Inbound"
            base_dir.mkdir(parents=True, exist_ok=True)

            deal_id = registry.generate_deal_id()
            slug = _title_slug(inferred_name or subject)
            folder_base = f"{slug}--{_short_deal_suffix(deal_id)}"
            deal_dir = base_dir / folder_base

            if deal_dir.exists():
                # Extremely unlikely; fall back to counter.
                for i in range(2, 200):
                    candidate = base_dir / f"{folder_base}-{i}"
                    if not candidate.exists():
                        deal_dir = candidate
                        break

            deal_dir.mkdir(parents=True, exist_ok=True)
            _ensure_deal_folder_template(deal_dir)
            broker = None
            if from_email:
                broker = BrokerInfo(name=str(inputs.get("inferred_broker_name") or "").strip(), email=from_email)

            deal = registry.create_deal(
                deal_id=deal_id,
                canonical_name=deal_name,
                folder_path=str(deal_dir),
                broker=broker,
                source="deal_create_from_email",
            )
            if thread_id and thread_id not in (deal.email_thread_ids or []):
                deal.email_thread_ids.append(thread_id)
                deal.add_audit("email_thread_added", "deal_create_from_email", f"Added thread_id={thread_id}")

            registry.add_email_deal_mapping(message_id, deal_id)
            if thread_id:
                registry.add_thread_deal_mapping(thread_id, deal_id)
            deal.add_audit(
                "email_mapped",
                "deal_create_from_email",
                f"Mapped email message_id={message_id}",
            )
            registry.save()

            deal_folder_path = str(deal_dir)
            created_new = True

        if not deal_folder_path:
            raise ActionExecutionError(
                ActionError(
                    code="deal_folder_path_missing",
                    message="Deal folder_path missing after create/load",
                    category="unknown",
                    retryable=False,
                )
            )

        deal_dir_path = Path(deal_folder_path).expanduser().resolve()
        _ensure_deal_folder_template(deal_dir_path)

        # Copy quarantine artifacts (best-effort; never destructive).
        written: List[str] = []

        corr_dir = deal_dir_path / "07-Correspondence"
        inbox_dir = corr_dir / "INBOX"
        att_dir = corr_dir / "ATTACHMENTS"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        att_dir.mkdir(parents=True, exist_ok=True)

        safe_mid = _sanitize_filename(message_id)
        email_md_path = inbox_dir / f"{safe_mid}.md"
        email_json_path = inbox_dir / f"{safe_mid}.json"

        if not email_md_path.exists():
            links = inputs.get("links") or []
            link_lines: List[str] = []
            if isinstance(links, list):
                for item in links:
                    if isinstance(item, dict):
                        url = str(item.get("url") or "").strip()
                        if url:
                            lt = str(item.get("type") or "").strip()
                            link_lines.append(f"- {lt + ': ' if lt else ''}{url}")
                    elif isinstance(item, str) and item.strip():
                        link_lines.append(f"- {item.strip()}")

            md = [
                "# Inbound Email",
                "",
                f"- **gmail_message_id:** `{message_id}`",
                f"- **gmail_thread_id:** `{thread_id or ''}`",
                f"- **subject:** {subject}",
                f"- **from_email:** {from_email or ''}",
                f"- **from_header:** {from_header or ''}",
                f"- **received_at:** {received_at or ''}",
                "",
            ]
            if snippet:
                md.extend(["## Snippet", "", snippet.strip(), ""])
            if link_lines:
                md.extend(["## Links", ""] + link_lines + [""])
            md.append("_Generated by ZakOps (deal_create_from_email)._\n")
            email_md_path.write_text("\n".join(md), encoding="utf-8")
            written.append(str(email_md_path))

        if not email_json_path.exists():
            payload_json = {
                "gmail_message_id": message_id,
                "gmail_thread_id": thread_id,
                "subject": subject,
                "from_email": from_email,
                "from_header": from_header,
                "received_at": received_at,
                "snippet": snippet,
                "links": inputs.get("links") or [],
                "attachments": inputs.get("attachments") or [],
            }
            _write_json(email_json_path, payload_json)
            written.append(str(email_json_path))

        # Copy files from quarantine dir
        copied_attachments: List[str] = []
        if quarantine_dir and quarantine_dir.exists():
            for child in sorted(quarantine_dir.iterdir()):
                if not child.is_file():
                    continue
                if child.name in {"deal_created.json"}:
                    continue
                dst = att_dir / _sanitize_filename(child.name)
                if _copy_file(child, dst):
                    copied_attachments.append(str(dst))
                    written.append(str(dst))

        # Copy any explicit attachment quarantine_path entries
        for item in (inputs.get("attachments") or []) if isinstance(inputs.get("attachments"), list) else []:
            if not isinstance(item, dict):
                continue
            qpath = str(item.get("quarantine_path") or "").strip()
            if not qpath:
                continue
            src = Path(qpath).expanduser().resolve()
            dst = att_dir / _sanitize_filename(str(item.get("filename") or src.name))
            if _copy_file(src, dst):
                copied_attachments.append(str(dst))
                written.append(str(dst))

        # Write action artifacts under deal/99-ACTIONS/{action_id}/
        action_dir = (deal_dir_path / "99-ACTIONS" / payload.action_id).resolve()
        action_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "action_id": payload.action_id,
            "action_type": payload.type,
            "created_at": now_utc_iso(),
            "created_new_deal": created_new,
            "deal_id": deal_id,
            "deal_name": deal_name,
            "deal_path": str(deal_dir_path),
            "email": {
                "gmail_message_id": message_id,
                "gmail_thread_id": thread_id,
                "subject": subject,
                "from_email": from_email,
                "from_header": from_header,
                "received_at": received_at,
            },
            "artifacts_written": written,
        }

        manifest_path = action_dir / "deal_create_from_email.manifest.json"
        _write_json(manifest_path, manifest)

        summary_path = action_dir / "deal_create_from_email.summary.md"
        if not summary_path.exists():
            summary = [
                "# Deal Created From Email",
                "",
                f"- **deal_id:** `{deal_id}`",
                f"- **deal_path:** `{deal_dir_path}`",
                f"- **created_new:** `{created_new}`",
                f"- **gmail_message_id:** `{message_id}`",
                "",
                "## Copied Artifacts",
                "",
            ]
            for p in written:
                summary.append(f"- `{p}`")
            summary.append("")
            summary_path.write_text("\n".join(summary), encoding="utf-8")

        artifacts: List[ArtifactMetadata] = [
            ArtifactMetadata(
                filename=manifest_path.name,
                mime_type="application/json",
                path=str(manifest_path),
                created_at=now_utc_iso(),
            ),
            ArtifactMetadata(
                filename=summary_path.name,
                mime_type="text/markdown",
                path=str(summary_path),
                created_at=now_utc_iso(),
            ),
        ]

        # Best-effort: write crash-recovery marker in quarantine dir
        if marker_path and quarantine_dir and quarantine_dir.exists():
            try:
                marker = {
                    "deal_id": deal_id,
                    "deal_name": deal_name,
                    "deal_path": str(deal_dir_path),
                    "message_id": message_id,
                    "thread_id": thread_id,
                    "created_at": now_utc_iso(),
                }
                _write_json(marker_path, marker)
            except Exception:
                pass

        # Create deal_profile.json from triage summary (best-effort)
        deal_profile_path = deal_dir_path / "deal_profile.json"
        if not deal_profile_path.exists():
            try:
                profile = {
                    "deal_id": deal_id,
                    "deal_name": deal_name,
                    "created_at": now_utc_iso(),
                    "source": "deal_create_from_email",
                    "company_info": {
                        "name": None,
                        "sector": None,
                        "location": None,
                        "website": None,
                    },
                    "financials": {
                        "asking_price": None,
                        "ebitda": None,
                        "revenue": None,
                        "sde": None,
                        "multiple": None,
                        "currency": None,
                    },
                    "deal_status": {
                        "nda_status": "none",
                        "cim_received": False,
                        "stage": "inbound",
                    },
                    "broker": {
                        "name": str(inputs.get("inferred_broker_name") or "").strip() or None,
                        "email": from_email,
                        "company": None,
                        "phone": None,
                    },
                    # Rich summary from triage
                    "triage_summary": {
                        "bullets": [],
                        "recommendation": None,
                        "why": None,
                        "confidence": None,
                        "ma_intent": None,
                    },
                    "evidence": [],
                }
                # Try to read triage summary and extract fields
                triage_summary_path = quarantine_dir / "triage_summary.json" if quarantine_dir else None
                if triage_summary_path and triage_summary_path.exists():
                    try:
                        triage = json.loads(triage_summary_path.read_text(encoding="utf-8"))
                        # Extract company info
                        target_co = triage.get("target_company") or {}
                        if target_co.get("name"):
                            profile["company_info"]["name"] = target_co["name"]
                        if target_co.get("industry"):
                            profile["company_info"]["sector"] = target_co["industry"]
                        if target_co.get("location"):
                            profile["company_info"]["location"] = target_co["location"]
                        if target_co.get("website"):
                            profile["company_info"]["website"] = target_co["website"]

                        # Extract financials from structured fields
                        deal_signals = triage.get("deal_signals") or {}
                        valuation = deal_signals.get("valuation_terms") or {}
                        if valuation.get("ask_price"):
                            profile["financials"]["asking_price"] = valuation["ask_price"]
                        if valuation.get("ebitda"):
                            profile["financials"]["ebitda"] = valuation["ebitda"]
                        if valuation.get("revenue"):
                            profile["financials"]["revenue"] = valuation["revenue"]
                        if valuation.get("sde"):
                            profile["financials"]["sde"] = valuation["sde"]
                        if valuation.get("multiple"):
                            profile["financials"]["multiple"] = valuation["multiple"]
                        if valuation.get("currency"):
                            profile["financials"]["currency"] = valuation["currency"]

                        # Extract actor info (broker)
                        actors = triage.get("actors") or {}
                        if actors.get("sender_org_guess"):
                            profile["broker"]["company"] = actors["sender_org_guess"]
                        if actors.get("sender_role_guess"):
                            profile["broker"]["role"] = actors["sender_role_guess"]

                        # Extract rich summary
                        summary = triage.get("summary") or {}
                        profile["triage_summary"]["bullets"] = summary.get("bullets") or []
                        profile["triage_summary"]["recommendation"] = summary.get("operator_recommendation")
                        profile["triage_summary"]["why"] = summary.get("why")
                        profile["triage_summary"]["confidence"] = triage.get("confidence")
                        profile["triage_summary"]["ma_intent"] = triage.get("ma_intent")

                        # Extract evidence
                        evidence = triage.get("evidence") or []
                        profile["evidence"] = [
                            {
                                "quote": e.get("quote"),
                                "reason": e.get("reason"),
                                "source": e.get("source"),
                                "weight": e.get("weight"),
                            }
                            for e in evidence
                            if isinstance(e, dict)
                        ]

                        # Try to extract financials from evidence quotes (heuristic)
                        for e in evidence:
                            quote = str(e.get("quote") or "").lower()
                            if "asking price" in quote and not profile["financials"]["asking_price"]:
                                # Try to extract dollar amount
                                import re
                                match = re.search(r"asking price[:\s]*\$?([\d,]+(?:\.\d+)?)\s*(m|million|k|thousand)?", quote, re.IGNORECASE)
                                if match:
                                    amount = match.group(1).replace(",", "")
                                    multiplier = match.group(2)
                                    try:
                                        val = float(amount)
                                        if multiplier and multiplier.lower() in ("m", "million"):
                                            val *= 1_000_000
                                        elif multiplier and multiplier.lower() in ("k", "thousand"):
                                            val *= 1_000
                                        profile["financials"]["asking_price"] = f"${val:,.0f}"
                                    except ValueError:
                                        pass
                            if "ebitda" in quote and not profile["financials"]["ebitda"]:
                                match = re.search(r"(\d+(?:\.\d+)?)\s*(m|million)?\s*ebitda", quote, re.IGNORECASE)
                                if match:
                                    amount = match.group(1)
                                    multiplier = match.group(2)
                                    try:
                                        val = float(amount)
                                        if multiplier and multiplier.lower() in ("m", "million"):
                                            val *= 1_000_000
                                        profile["financials"]["ebitda"] = f"${val:,.0f}"
                                    except ValueError:
                                        pass
                    except Exception:
                        pass
                _write_json(deal_profile_path, profile)
                written.append(str(deal_profile_path))
            except Exception:
                pass

        next_actions = [
            {
                "action_type": "DEAL.EXTRACT_EMAIL_ARTIFACTS",
                "capability_id": "deal.extract_email_artifacts.v1",
                "title": "Extract entities + doc types from inbound email artifacts",
                "inputs": {
                    "deal_id": deal_id,
                    "deal_path": str(deal_dir_path),
                    "artifact_paths": [email_md_path.as_posix()] + copied_attachments,
                },
                "requires_approval": False,
            }
        ]

        outputs: Dict[str, Any] = {
            "deal_id": deal_id,
            "deal_name": deal_name,
            "deal_path": str(deal_dir_path),
            "artifacts_written": written,
            "email_message_id": message_id,
            "next_actions": next_actions,
        }

        return ExecutionResult(outputs=outputs, artifacts=artifacts)


def hashlib_sha256(text: str) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update((text or "").encode("utf-8", errors="replace"))
    return h.hexdigest()
