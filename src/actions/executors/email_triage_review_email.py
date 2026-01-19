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
from actions.memory.triage_feedback import append_feedback, build_feedback_entry
from integrations.n8n_webhook import emit_quarantine_approved


# ============================================================================
# Financial Extraction Helpers
# ============================================================================

_MONEY_RE = re.compile(
    r"\$\s?([\d,]+(?:\.\d{1,2})?)\s*([KkMm])?(?:\s*(?:USD|usd))?",
    re.IGNORECASE,
)

def _parse_money(text: str) -> Optional[float]:
    """Parse a money string like '$145,000', '$350K', '$1.5M' into a float."""
    m = _MONEY_RE.search(text or "")
    if not m:
        return None
    num_str = m.group(1).replace(",", "")
    try:
        value = float(num_str)
    except ValueError:
        return None
    suffix = (m.group(2) or "").upper()
    if suffix == "K":
        value *= 1_000
    elif suffix == "M":
        value *= 1_000_000
    return value


def _extract_financials_from_text(text: str) -> Dict[str, Any]:
    """
    Extract financial data from email/triage text.

    Returns dict with keys: asking_price, ebitda, revenue, sde
    """
    result: Dict[str, Any] = {
        "asking_price": None,
        "ebitda": None,
        "revenue": None,
        "sde": None,
    }
    if not text:
        return result

    text_lower = text.lower()

    # Patterns for extracting financial values
    patterns = {
        "asking_price": [
            r"asking\s*(?:price)?[:\s]*(\$[\d,\.]+\s*[KkMm]?)",
            r"(\$[\d,\.]+\s*[KkMm]?)\s*asking",
            r"listed\s*(?:at|for)[:\s]*(\$[\d,\.]+\s*[KkMm]?)",
            r"price[:\s]*(\$[\d,\.]+\s*[KkMm]?)",
        ],
        "ebitda": [
            r"ebitda[:\s]*(\$[\d,\.]+\s*[KkMm]?)",
            r"(\$[\d,\.]+\s*[KkMm]?)\s*ebitda",
            r"earnings[:\s]*(\$[\d,\.]+\s*[KkMm]?)",
            r"(\$[\d,\.]+\s*[KkMm]?)\s*(?:net\s*)?earnings",
            r"cash\s*flow[:\s]*(\$[\d,\.]+\s*[KkMm]?)",
            r"(\$[\d,\.]+\s*[KkMm]?)\s*cash\s*flow",
        ],
        "revenue": [
            r"revenue[:\s]*(\$[\d,\.]+\s*[KkMm]?)",
            r"(\$[\d,\.]+\s*[KkMm]?)\s*revenue",
            r"gross\s*(?:sales|revenue)[:\s]*(\$[\d,\.]+\s*[KkMm]?)",
            r"arr[:\s]*(\$[\d,\.]+\s*[KkMm]?)",
            r"(\$[\d,\.]+\s*[KkMm]?)\s*arr",
        ],
        "sde": [
            r"sde[:\s]*(\$[\d,\.]+\s*[KkMm]?)",
            r"(\$[\d,\.]+\s*[KkMm]?)\s*sde",
            r"seller['\u2019]?s?\s*(?:discretionary\s*)?(?:earnings|income)[:\s]*(\$[\d,\.]+\s*[KkMm]?)",
        ],
    }

    for field, field_patterns in patterns.items():
        for pattern in field_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                money_str = match.group(1)
                value = _parse_money(money_str)
                if value and value > 0:
                    result[field] = value
                    break

    return result


def _extract_sector_from_text(text: str) -> Optional[str]:
    """Extract business sector/industry from text."""
    if not text:
        return None

    sectors = {
        "IT Services": ["it services", "msp", "managed services", "technology services", "tech services"],
        "SaaS": ["saas", "software as a service", "software-as-a-service", "cloud software"],
        "E-commerce": ["e-commerce", "ecommerce", "amazon fba", "shopify", "online retail"],
        "Manufacturing": ["manufacturing", "manufacturer", "production"],
        "Healthcare": ["healthcare", "medical", "health services", "clinic"],
        "Construction": ["construction", "roofing", "plumbing", "hvac", "electrical"],
        "Professional Services": ["consulting", "professional services", "advisory"],
        "Education": ["education", "training", "learning", "courses", "coaching"],
        "Media": ["media", "content", "publishing", "digital media"],
    }

    text_lower = text.lower()
    for sector, keywords in sectors.items():
        for kw in keywords:
            if kw in text_lower:
                return sector
    return None


def _extract_location_from_text(text: str) -> Optional[str]:
    """Extract location from text (US states, cities)."""
    if not text:
        return None

    # Common US state patterns
    state_patterns = [
        r"\b([A-Z]{2})\b",  # State abbreviations like TX, CA
        r"(?:located\s+(?:in|near)\s+)([A-Za-z\s]+(?:,\s*[A-Z]{2})?)",
        r"(?:based\s+(?:in|out\s+of)\s+)([A-Za-z\s]+(?:,\s*[A-Z]{2})?)",
    ]

    for pattern in state_patterns:
        match = re.search(pattern, text)
        if match:
            location = match.group(1).strip()
            if len(location) >= 2 and len(location) <= 50:
                return location
    return None


def _dataroom_root() -> Path:
    return Path(os.getenv("DATAROOM_ROOT", "/home/zaks/DataRoom")).resolve()


def _inbound_root() -> Path:
    return _dataroom_root() / "00-PIPELINE" / "Inbound"


def _quarantine_root() -> Path:
    return _dataroom_root() / "00-PIPELINE" / "_INBOX_QUARANTINE"


DEAL_SUBFOLDERS = [
    "01-NDA",
    "02-CIM",
    "03-Financials",
    "04-Operations",
    "05-Legal",
    "06-Analysis",
    "07-Correspondence",
    "08-LOI-Offer",
    "09-Closing",
]


_EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")


def _extract_email_address(value: str) -> str:
    m = _EMAIL_RE.search(value or "")
    return m.group(1).lower() if m else ""


def _extract_domain(email_addr: str) -> str:
    addr = (email_addr or "").strip()
    if "@" not in addr:
        return ""
    return addr.split("@", 1)[1].strip().lower()


def _clean_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", (text or "").strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned[:120] or "Deal"


def _infer_deal_name(payload: ActionPayload) -> str:
    subject = str((payload.inputs or {}).get("subject") or "").strip()
    company = str((payload.inputs or {}).get("company") or "").strip()
    base = company or subject or payload.title or "Deal"
    base = re.sub(r"^(re:|fw:|fwd:)\\s*", "", base, flags=re.I).strip()
    return base


_CODE_NAME_RE = re.compile(r"^(?:project|operation|deal|target)\\s+\\w+", flags=re.I)
_DEAL_NOISE_RE = re.compile(
    r"\\b(?:cim|teaser|nda|confidential|data\\s*room|dataroom|vdr|new\\s+listing|for\\s+sale|available)\\b",
    flags=re.I,
)


def _looks_like_code_name(name: str) -> bool:
    candidate = re.sub(r"^(re:|fw:|fwd:)\\s*", "", (name or "").strip(), flags=re.I).strip()
    if not candidate:
        return False
    return bool(_CODE_NAME_RE.match(candidate))


def _short_deal_suffix(deal_id: str) -> str:
    """
    Stable short suffix for folder names.

    DEAL-2026-092 -> 2026-092
    """
    parts = [p for p in str(deal_id or "").strip().split("-") if p]
    if len(parts) >= 3:
        return "-".join(parts[-2:])
    return str(deal_id or "").strip() or "unknown"


def _score_name(name: str, *, base: float, is_code_name: bool) -> float:
    n = (name or "").strip()
    if not n:
        return -1.0
    score = float(base)
    if is_code_name:
        score -= 0.35
    if len(n) < 4:
        score -= 0.4
    if len(n) > 90:
        score -= 0.2
    if _DEAL_NOISE_RE.search(n):
        score -= 0.2
    return score


def _resolve_deal_display_name(*, inputs: Dict[str, Any], subject: str, body: str, sender: str) -> Tuple[str, str]:
    """
    Resolve a business-name-first deal display name.

    Returns: (display_name, reason_code)
    """
    triage_company = str(inputs.get("company") or "").strip()
    subject_candidate = re.sub(r"^(re:|fw:|fwd:)\\s*", "", (subject or "").strip(), flags=re.I).strip()
    if " - " in subject_candidate:
        left = subject_candidate.split(" - ", 1)[0].strip()
        if 3 <= len(left) <= 90:
            subject_candidate = left

    resolver_name = ""
    resolver_score = -1.0
    try:
        from email_ingestion.enrichment.name_resolver import NameResolver

        res = NameResolver().resolve_company_name(subject=subject, body=body, sender_name=sender)
        if res and str(res.display_name or "").strip():
            resolver_name = str(res.display_name).strip()
            resolver_score = _score_name(resolver_name, base=float(res.confidence or 0.6), is_code_name=bool(getattr(res, "is_code_name", False)))
    except Exception:
        resolver_name = ""
        resolver_score = -1.0

    triage_score = _score_name(triage_company, base=0.8, is_code_name=_looks_like_code_name(triage_company))
    subject_score = _score_name(subject_candidate, base=0.35, is_code_name=_looks_like_code_name(subject_candidate))

    best_name = triage_company
    best_reason = "triage_company"
    best_score = triage_score

    if resolver_score > best_score:
        best_name = resolver_name
        best_reason = "name_resolver"
        best_score = resolver_score
    if subject_score > best_score:
        best_name = subject_candidate
        best_reason = "subject_fallback"

    best_name = (best_name or "").strip()[:120] or "Deal"
    return best_name, best_reason


def _parse_email_date(raw: str) -> Optional[datetime]:
    text = (raw or "").strip()
    if not text:
        return None
    # Example: "26 Dec 2025 21:01:30 +0000"
    for fmt in ("%d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    # Best-effort ISO fallback (strip Z)
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _timestamp_prefix(dt: Optional[datetime]) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _create_deal_workspace(*, deal_path: Path, deal_name: str, email_from: str, email_subject: str, email_received: str, email_body: str) -> None:
    deal_path.mkdir(parents=True, exist_ok=True)
    for sub in DEAL_SUBFOLDERS:
        (deal_path / sub).mkdir(exist_ok=True)

    today = datetime.now().strftime("%B %d, %Y")
    readme = f"""# {deal_name}

**Deal Status:** New - Inbound
**Date Added:** {today}
**Source:** Email from {email_from or "unknown"}

---

## QUICK FACTS

- **Subject:** {email_subject}
- **Received:** {email_received}
- **Status:** Awaiting Review

---

## INITIAL EMAIL SUMMARY

{(email_body or "").strip()[:800]}{'...' if (email_body and len(email_body) > 800) else ''}

---

## NEXT STEPS

- [ ] Review email and attachments
- [ ] Sign NDA if required
- [ ] Request CIM/financials
- [ ] Schedule introductory call
- [ ] Complete initial screening

---

## FOLDER STRUCTURE

- **01-NDA/** - NDA documents and signatures
- **02-CIM/** - Confidential Information Memorandum
- **03-Financials/** - P&Ls, tax returns, financial analysis
- **04-Operations/** - SOPs, client data, tech stack info
- **05-Legal/** - Contracts, leases, IP documentation
- **06-Analysis/** - Your evaluation, scorecard, models
- **07-Correspondence/** - Email threads, call notes
- **08-LOI-Offer/** - Letter of Intent, negotiation docs
- **09-Closing/** - Purchase agreement, closing documents

---

**Document Control:**
- Created: {today}
- Last Updated: {today}
- Status: Active - New Inbound
"""
    (deal_path / "README.md").write_text(readme, encoding="utf-8")


def _copy_quarantine_payload(*, quarantine_dir: Path, dest_dir: Path) -> Tuple[int, int]:
    """
    Copy quarantine contents into dest_dir.

    Returns: (copied_files, skipped_files)
    """
    copied = 0
    skipped = 0
    if not quarantine_dir.exists() or not quarantine_dir.is_dir():
        return (0, 0)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(quarantine_dir.iterdir()):
        if src.is_dir():
            # Do not recurse; keep deterministic and minimal.
            skipped += 1
            continue
        if src.name in {"email.json", "email_body.txt"}:
            # These are re-materialized below as part of the deal artifacts.
            skipped += 1
            continue
        try:
            shutil.copy2(src, dest_dir / src.name)
            copied += 1
        except Exception:
            skipped += 1
    return (copied, skipped)


class EmailTriageReviewEmailExecutor(ActionExecutor):
    """
    Approval-gated executor for EMAIL_TRIAGE.REVIEW_EMAIL.

    On approval/execution:
    - Create a new deal workspace (Inbound) if the email is not already linked to a deal
    - Register the deal in the existing DealRegistry (no new registry system)
    - Persist email artifacts under the deal folder
    """

    action_type = "EMAIL_TRIAGE.REVIEW_EMAIL"

    def validate(self, payload: ActionPayload) -> tuple[bool, Optional[str]]:
        msg_id = str((payload.inputs or {}).get("message_id") or "").strip()
        subject = str((payload.inputs or {}).get("subject") or "").strip()
        if not msg_id:
            return False, "Missing required inputs.message_id"
        if not subject:
            return False, "Missing required inputs.subject"
        return True, None

    def execute(self, payload: ActionPayload, ctx: ExecutionContext) -> ExecutionResult:
        inputs = payload.inputs or {}

        message_id = str(inputs.get("message_id") or "").strip()
        thread_id = str(inputs.get("thread_id") or "").strip()
        email_from = str(inputs.get("from") or "").strip()
        email_to = str(inputs.get("to") or "").strip()
        email_subject = str(inputs.get("subject") or "").strip()
        email_date_raw = str(inputs.get("date") or "").strip()

        if not message_id:
            raise ActionExecutionError(
                ActionError(
                    code="invalid_inputs",
                    message="inputs.message_id is required",
                    category="validation",
                    retryable=False,
                    details={},
                )
            )

        registry = getattr(ctx, "registry", None)

        quarantine_dir_raw = str(inputs.get("quarantine_dir") or "").strip()
        quarantine_root = _quarantine_root()
        quarantine_dir = Path(quarantine_dir_raw).resolve() if quarantine_dir_raw else (quarantine_root / message_id).resolve()

        email_body = ""
        try:
            body_path = quarantine_dir / "email_body.txt"
            if body_path.exists() and body_path.is_file():
                email_body = body_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            email_body = ""

        existing_deal_id = None
        existing_deal_folder = None

        # Prefer explicit deal_id (payload or inputs) if it resolves in the registry.
        explicit = str(inputs.get("link_deal_id") or inputs.get("deal_id") or payload.deal_id or "").strip()
        if registry and explicit and explicit.upper() != "GLOBAL":
            deal = registry.get_deal(explicit)
            if deal and deal.folder_path:
                existing_deal_id = explicit
                existing_deal_folder = str(deal.folder_path)

        # Otherwise, use email->deal mapping if present.
        if registry and not existing_deal_id:
            mapped = registry.get_email_deal_mapping(message_id)
            if mapped:
                deal = registry.get_deal(mapped)
                if deal and deal.folder_path:
                    existing_deal_id = mapped
                    existing_deal_folder = str(deal.folder_path)

        # Tier 3: thread_id -> deal mapping (deterministic).
        if registry and not existing_deal_id and thread_id:
            mapped = registry.get_thread_deal_mapping(thread_id)
            if mapped:
                deal = registry.get_deal(mapped)
                if deal and deal.folder_path:
                    existing_deal_id = mapped
                    existing_deal_folder = str(deal.folder_path)

        # Tier 4: heuristic DealMatcher fallback (best-effort, still local-only).
        if registry and not existing_deal_id:
            try:
                from deal_registry import DealMatcher, EmailContent

                matcher = DealMatcher(registry)
                res = matcher.match(
                    EmailContent(
                        subject=email_subject,
                        body=email_body or "",
                        sender=email_from,
                        message_id=message_id or None,
                        thread_id=thread_id or None,
                        received_date=email_date_raw or None,
                    )
                )
                if res.matched and res.deal_id:
                    deal = registry.get_deal(res.deal_id)
                    if deal and deal.folder_path:
                        existing_deal_id = res.deal_id
                        existing_deal_folder = str(deal.folder_path)
            except Exception:
                pass

        created_new_deal = False
        deal_id = existing_deal_id
        deal_folder = existing_deal_folder

        # Create deal if needed.
        if not deal_id:
            if not registry:
                raise ActionExecutionError(
                    ActionError(
                        code="registry_unavailable",
                        message="DealRegistry not available in execution context; cannot create deal record",
                        category="dependency",
                        retryable=False,
                        details={},
                    )
                )

            display_name, name_reason = _resolve_deal_display_name(
                inputs=inputs,
                subject=email_subject,
                body=email_body or "",
                sender=email_from,
            )

            new_deal_id = registry.generate_deal_id()
            suffix = _short_deal_suffix(new_deal_id)

            inbound_root = _inbound_root()
            inbound_root.mkdir(parents=True, exist_ok=True)
            slug = _clean_component(display_name)
            folder_base = f"{slug}--{suffix}"
            deal_path = (inbound_root / folder_base).resolve()
            counter = 2
            while deal_path.exists():
                deal_path = (inbound_root / f"{folder_base}-{counter}").resolve()
                counter += 1

            _create_deal_workspace(
                deal_path=deal_path,
                deal_name=display_name,
                email_from=email_from,
                email_subject=email_subject,
                email_received=email_date_raw,
                email_body=email_body,
            )

            broker = None
            try:
                from deal_registry import BrokerInfo as DealBrokerInfo

                broker_email = _extract_email_address(email_from) or str(inputs.get("sender_email") or "").strip()
                broker_name = ""
                if "<" in email_from and ">" in email_from:
                    broker_name = email_from.split("<", 1)[0].strip().strip('"').strip()
                broker = DealBrokerInfo(name=broker_name or "", email=broker_email or "")
                broker.domain = _extract_domain(broker_email)
            except Exception:
                broker = None

            deal_obj = registry.create_deal(
                deal_id=new_deal_id,
                canonical_name=display_name,
                folder_path=str(deal_path),
                broker=broker,
                source="email_triage",
            )
            try:
                deal_obj.display_name = display_name
                deal_obj.company_info.company_name = display_name
                deal_obj.add_audit("name_resolved", "email_triage", f"Resolved display_name={display_name} via {name_reason}")
                deal_obj.add_alias(display_name, "company_name", confidence=0.9, source="email_triage")
                triage_name = str(inputs.get("company") or "").strip()
                if triage_name and triage_name != display_name and _looks_like_code_name(triage_name):
                    deal_obj.add_alias(triage_name, "name_variation", confidence=0.6, source="email_triage")
            except Exception:
                pass

            # Extract and populate financial/sector/location data from email
            combined_text = f"{email_subject}\n{email_body}"
            financials = _extract_financials_from_text(combined_text)
            sector = _extract_sector_from_text(combined_text)
            location = _extract_location_from_text(combined_text)

            try:
                if hasattr(deal_obj, "metadata") and deal_obj.metadata:
                    if financials.get("asking_price"):
                        deal_obj.metadata["asking_price"] = financials["asking_price"]
                    if financials.get("ebitda"):
                        deal_obj.metadata["ebitda"] = financials["ebitda"]
                    if financials.get("revenue"):
                        deal_obj.metadata["revenue"] = financials["revenue"]
                if hasattr(deal_obj, "company_info") and deal_obj.company_info:
                    if sector:
                        deal_obj.company_info.industry = sector
                    if location:
                        deal_obj.company_info.location = location
            except Exception:
                pass

            # Create deal_profile.json with enrichment data
            try:
                profile_data = {
                    "deal_id": new_deal_id,
                    "deal_name": display_name,
                    "created_at": now_utc_iso(),
                    "source": "email_triage_review",
                    "company_info": {
                        "name": display_name,
                        "sector": sector,
                        "location": location,
                        "website": None,
                    },
                    "financials": {
                        "asking_price": financials.get("asking_price"),
                        "ebitda": financials.get("ebitda"),
                        "revenue": financials.get("revenue"),
                        "sde": financials.get("sde"),
                        "multiple": None,
                        "currency": "USD" if any(financials.values()) else None,
                    },
                    "deal_status": {
                        "nda_status": "none",
                        "cim_received": False,
                        "stage": "inbound",
                    },
                    "broker": {
                        "name": broker_name if broker else None,
                        "email": broker_email if broker else None,
                        "company": None,
                        "phone": None,
                        "role": "SELLER_REP",
                    },
                    "triage_summary": {
                        "bullets": [],
                        "summary_text": (email_body or "").strip()[:500] if email_body else None,
                        "recommendation": "APPROVE",
                        "confidence": float(inputs.get("confidence") or 0.7),
                    },
                }
                profile_path = deal_path / "deal_profile.json"
                profile_path.write_text(json.dumps(profile_data, indent=2), encoding="utf-8")
            except Exception:
                pass
            if thread_id and thread_id not in deal_obj.email_thread_ids:
                deal_obj.email_thread_ids.append(thread_id)
            registry.add_email_deal_mapping(message_id, new_deal_id)
            if thread_id:
                registry.add_thread_deal_mapping(thread_id, new_deal_id)
            registry.save()

            # Verify deal was actually persisted (atomicity check).
            # Reload registry and confirm the deal exists - if not, fail the action so it stays in queue.
            try:
                from deal_registry import DealRegistry
                registry_path = os.getenv("DEAL_REGISTRY_PATH", "/home/zaks/DataRoom/.deal-registry/deal_registry.json")
                verify_registry = DealRegistry(registry_path)
                verify_deal = verify_registry.get_deal(new_deal_id)
                if not verify_deal or not verify_deal.folder_path:
                    raise ActionExecutionError(
                        ActionError(
                            code="registry_verification_failed",
                            message=f"Deal {new_deal_id} not found in registry after save - deal creation may have failed",
                            category="persistence",
                            retryable=True,
                            details={"deal_id": new_deal_id, "folder_path": str(deal_path)},
                        )
                    )
            except ActionExecutionError:
                raise
            except Exception as e:
                # Non-blocking if verification cannot complete - just log
                import logging
                logging.getLogger(__name__).warning("Registry verification skipped: %s", e)

            deal_id = new_deal_id
            deal_folder = str(deal_path)
            created_new_deal = True
        else:
            # Ensure the email is mapped to this deal (idempotency + routing) and capture thread_id.
            try:
                if registry:
                    registry.add_email_deal_mapping(message_id, deal_id)
                    if thread_id:
                        registry.add_thread_deal_mapping(thread_id, deal_id)
                    deal_obj = registry.get_deal(deal_id)
                    if deal_obj and thread_id and thread_id not in (deal_obj.email_thread_ids or []):
                        deal_obj.email_thread_ids.append(thread_id)
                    registry.save()
            except Exception:
                pass

        if not deal_folder:
            raise ActionExecutionError(
                ActionError(
                    code="deal_folder_missing",
                    message="Deal folder_path missing after deal resolution",
                    category="validation",
                    retryable=False,
                    details={},
                )
            )

        deal_folder_path = Path(deal_folder).expanduser()
        deal_path = (_dataroom_root() / deal_folder_path).resolve() if not deal_folder_path.is_absolute() else deal_folder_path.resolve()

        # Queue post-approval materials ingestion (append-only correspondence bundle).
        quarantine_dir_raw = str(inputs.get("quarantine_dir") or "").strip()
        quarantine_root = _quarantine_root()
        quarantine_dir = Path(quarantine_dir_raw).resolve() if quarantine_dir_raw else (quarantine_root / message_id).resolve()

        next_actions = [
            {
                "action_type": "DEAL.APPEND_EMAIL_MATERIALS",
                "capability_id": "deal.append_email_materials.v1",
                "title": "Ingest approved email into deal correspondence bundle",
                "inputs": {
                    "deal_id": deal_id,
                    "deal_path": str(deal_path),
                    "message_id": message_id,
                    "thread_id": thread_id,
                    "from": email_from,
                    "to": email_to,
                    "date": email_date_raw,
                    "subject": email_subject,
                    "classification": inputs.get("classification"),
                    "urgency": inputs.get("urgency"),
                    "confidence": inputs.get("confidence"),
                    "links": inputs.get("links") or [],
                    "attachments": inputs.get("attachments") or [],
                    "quarantine_dir": str(quarantine_dir),
                },
                "requires_approval": False,
                "idempotency_key": f"append_email_materials:{deal_id}:{message_id}",
            },
        ]

        # Post-approval: sender history backfill (local-only). This attaches older materials from the
        # same sender into the deal, and/or generates additional quarantine review items.
        sender_email = str(inputs.get("sender_email") or _extract_email_address(email_from) or "").strip().lower()
        if sender_email and "@" in sender_email:
            try:
                lookback_days = int(os.getenv("DEAL_BACKFILL_SENDER_LOOKBACK_DAYS", "365") or "365")
            except Exception:
                lookback_days = 365
            try:
                max_messages = int(os.getenv("DEAL_BACKFILL_SENDER_MAX_MESSAGES", "50") or "50")
            except Exception:
                max_messages = 50
            mode = str(os.getenv("DEAL_BACKFILL_SENDER_MODE", "classify_and_quarantine") or "classify_and_quarantine").strip()
            if mode not in {"same_deal_only", "classify_and_quarantine"}:
                mode = "classify_and_quarantine"
            try:
                min_confidence_same = float(os.getenv("DEAL_BACKFILL_MIN_CONFIDENCE_SAME", "0.9") or "0.9")
            except Exception:
                min_confidence_same = 0.9
            try:
                max_thread_messages = int(os.getenv("DEAL_BACKFILL_MAX_THREAD_MESSAGES", "25") or "25")
            except Exception:
                max_thread_messages = 25

            next_actions.append(
                {
                    "action_type": "DEAL.BACKFILL_SENDER_HISTORY",
                    "capability_id": "deal.backfill_sender_history.v1",
                    "title": "Backfill sender history for approved deal",
                    "inputs": {
                        "deal_id": deal_id,
                        "approved_message_id": message_id,
                        "sender_email": sender_email,
                        "lookback_days": int(lookback_days),
                        "max_messages": int(max_messages),
                        "mode": mode,
                        "min_confidence_same": float(min_confidence_same),
                        "max_thread_messages": int(max_thread_messages),
                    },
                    "requires_approval": False,
                    "idempotency_key": f"backfill_sender_history:{deal_id}:{sender_email}:{int(lookback_days)}",
                }
            )

        outputs: Dict[str, Any] = {
            "deal_id": deal_id,
            "deal_folder": str(deal_path),
            "created_new_deal": created_new_deal,
            "quarantine_dir": str(quarantine_dir),
            "next_actions": next_actions,
        }

        artifacts: List[ArtifactMetadata] = []

        # Operator feedback dataset (minimal; no raw bodies). Best-effort.
        try:
            actor = payload.created_by
            for ev in reversed(payload.audit_trail or []):
                if getattr(ev, "event", "") == "approved":
                    actor = getattr(ev, "actor", actor) or actor
                    break

            entry = build_feedback_entry(
                decision="approve",
                message_id=message_id,
                thread_id=thread_id or None,
                sender=str(inputs.get("from") or ""),
                subject=email_subject,
                classification=str(inputs.get("classification") or "") or None,
                confidence=(float(inputs.get("confidence")) if inputs.get("confidence") is not None else None),
                actor=actor,
                action_id=payload.action_id,
                action_type=payload.type,
                deal_id=deal_id,
                extra={
                    "created_new_deal": bool(created_new_deal),
                    "deal_folder": str(deal_path),
                },
            )
            append_feedback(entry)
        except Exception:
            pass

        # Optional n8n integration (off by default unless ZAKOPS_N8N_WEBHOOK_URL is set).
        try:
            emit_quarantine_approved(
                message_id=message_id,
                thread_id=thread_id or None,
                deal_id=str(deal_id),
                deal_folder=str(deal_path),
            )
        except Exception:
            pass

        return ExecutionResult(outputs=outputs, artifacts=artifacts)
