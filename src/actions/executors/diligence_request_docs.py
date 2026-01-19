"""
DILIGENCE.REQUEST_DOCS Executor - Gemini-Powered Document Request Workflow

Workflow steps:
1. gather_context - Build ContextPack for deal
2. draft_email - Use Gemini Flash to draft broker email (strict JSON {subject, body})
3. store_artifact - Save draft as artifact

Output: Email draft artifact + checklist artifact
Follow-up: Suggests COMMUNICATION.SEND_EMAIL (requires approval) but does NOT auto-send

IMPORTANT: No auto-sending. All sends require explicit approval.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from typing import Any, Dict, Optional

from actions.engine.models import ActionError, ActionPayload, ArtifactMetadata, now_utc_iso
from actions.executors._artifacts import resolve_action_artifact_dir
from actions.executors.base import ActionExecutionError, ActionExecutor, ExecutionContext, ExecutionResult

logger = logging.getLogger(__name__)

# Use Gemini Flash for drafting (fast, cheap)
GEMINI_MODEL = os.getenv("GEMINI_MODEL_FLASH", "gemini-2.5-flash")
USE_LLM_FOR_DRAFT = os.getenv("KINETIC_USE_LLM_DRAFT", "true").lower() == "true"


def _get_deterministic_checklist(doc_type: str) -> list[str]:
    """Get deterministic checklist based on doc_type (fallback if LLM fails)."""
    dt = doc_type.lower()
    if "financial" in dt or "ltm" in dt or "p&l" in dt:
        return [
            "- LTM P&L and balance sheet",
            "- Last 3 years financial statements",
            "- Revenue by product/service line",
            "- Customer concentration (top 10)",
        ]
    elif "legal" in dt or "contracts" in dt:
        return [
            "- Material customer contracts",
            "- Vendor/lease agreements",
            "- Corporate formation docs (if applicable)",
        ]
    else:
        return [
            "- CIM / teaser (if available)",
            "- LTM financials",
            "- List of add-backs / adjustments",
            "- Org chart / headcount",
        ]


def _build_draft_prompt(context_str: str, doc_type: str, description: str, broker_name: str) -> str:
    """Build the prompt for Gemini to draft the email."""
    return f"""You are drafting a professional document request email to a business broker regarding an M&A transaction.

CONTEXT:
{context_str}

REQUEST:
- Document type needed: {doc_type}
- Specific request: {description}
- Broker name: {broker_name}

INSTRUCTIONS:
1. Draft a professional, concise email requesting the specified documents
2. Reference the specific deal if context is provided
3. Include a clear bulleted list of documents needed
4. Be polite but direct - this is standard M&A due diligence
5. Do NOT include any confidential information or numbers
6. Sign as "Zak" (the operator)

OUTPUT FORMAT (strict JSON):
Return ONLY a JSON object with exactly these fields:
{{
  "subject": "Brief email subject line",
  "body": "Full email body text with proper formatting"
}}

Do not include any other text before or after the JSON."""


async def _draft_email_with_gemini(
    context_str: str,
    doc_type: str,
    description: str,
    broker_name: str,
) -> Optional[Dict[str, str]]:
    """Use Gemini Flash to draft the email. Returns {subject, body} or None on failure."""
    try:
        # Import the provider module
        import sys
        sys.path.insert(0, "/home/zaks/scripts")
        from chat_llm_provider import get_provider

        provider = get_provider("gemini-flash")

        # Check availability
        if not provider.available:
            logger.warning("Gemini API key not available, falling back to deterministic")
            return None

        # Build prompt
        prompt = _build_draft_prompt(context_str, doc_type, description, broker_name)

        # Call Gemini
        response = await provider.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Low temperature for consistent output
            max_tokens=2000,
        )

        # Parse JSON response
        content = response.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            # Find the first newline after the fence
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline + 1:]
            # Strip trailing fence
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        try:
            result = json.loads(content)
            if "subject" in result and "body" in result:
                logger.info(f"Gemini draft successful: {response.latency_ms}ms")
                return result
            else:
                logger.warning(f"Gemini response missing fields: {result.keys()}")
                return None
        except json.JSONDecodeError as e:
            logger.warning(f"Gemini response not valid JSON: {e}")
            # Try to extract JSON from response
            match = re.search(r'\{[^{}]*"subject"[^{}]*"body"[^{}]*\}', content, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                    return result
                except Exception:
                    pass
            return None

    except Exception as e:
        logger.warning(f"Gemini draft failed: {e}")
        return None


def _run_coro_blocking(coro):
    """
    Run a coroutine from sync code.

    - If no loop is running: uses asyncio.run()
    - If a loop is already running (e.g., unit tests / API server): runs in a dedicated thread
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Dict[str, Any] = {}
    error: Dict[str, BaseException] = {}

    def _thread_main() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover
            error["exc"] = exc

    t = threading.Thread(target=_thread_main, name="zakops-request-docs-async", daemon=True)
    t.start()
    t.join()
    if error:
        raise error["exc"]
    return result.get("value")


def _create_fallback_draft(doc_type: str, description: str, broker_name: str, deal_name: str) -> Dict[str, str]:
    """Create a deterministic fallback email draft."""
    checklist = _get_deterministic_checklist(doc_type)
    checklist_str = "\n".join(checklist)

    subject = f"Document Request - {deal_name or 'Deal'} - {doc_type.title()}"

    body = f"""Hi {broker_name or 'there'},

Thank you for the opportunity to review {deal_name or 'this opportunity'}. As we progress with our due diligence, we would appreciate receiving the following documents:

{checklist_str}

{description}

Please let me know if you have any questions or if any of these items are not currently available.

Best regards,
Zak

---
_This is a draft - please review before sending._"""

    return {"subject": subject, "body": body}


class RequestDocsExecutor(ActionExecutor):
    """
    Gemini-powered diligence document request workflow.

    Workflow:
    1. Gather context from deal registry, events, case files
    2. Use Gemini Flash to draft a professional broker email
    3. Save artifacts: email draft (.md) + checklist (.md)

    IMPORTANT: Draft-only. Does NOT send emails. Sending requires a separate
    COMMUNICATION.SEND_EMAIL action with explicit approval.
    """

    action_type = "DILIGENCE.REQUEST_DOCS"

    def validate(self, payload: ActionPayload) -> tuple[bool, Optional[str]]:
        inputs = payload.inputs or {}
        doc_type = str(inputs.get("doc_type", "")).strip()
        description = str(inputs.get("description", "")).strip()
        if not doc_type:
            return False, "Missing required field: doc_type"
        if not description:
            return False, "Missing required field: description"
        return True, None

    def execute(self, payload: ActionPayload, ctx: ExecutionContext) -> ExecutionResult:
        inputs = payload.inputs or {}
        doc_type = str(inputs.get("doc_type", "")).strip()
        description = str(inputs.get("description", "")).strip()

        if not doc_type or not description:
            raise ActionExecutionError(
                ActionError(
                    code="validation_failed",
                    message="doc_type and description are required",
                    category="validation",
                    retryable=False,
                )
            )

        # Step 1: Gather context
        context_str = ""
        broker_name = "Broker"
        deal_name = payload.deal_id

        if payload.deal_id:
            try:
                from actions.context.context_pack import build_context_pack

                pack = build_context_pack(
                    payload.deal_id,
                    action_type=self.action_type,
                    include_rag=True,
                    max_events=5,
                    max_rag_chunks=3,
                )
                context_str = pack.to_prompt_context()
                broker_name = pack.broker.name if pack.broker and pack.broker.name else "Broker"
                deal_name = pack.display_name or pack.canonical_name or payload.deal_id
                logger.info(f"Context pack built: {len(context_str)} chars, broker={broker_name}")
            except Exception as e:
                logger.warning(f"Failed to build context pack: {e}")
                # Continue with minimal context

        # Step 2: Draft email with Gemini (or fallback)
        email_draft = None
        used_llm = False

        if USE_LLM_FOR_DRAFT:
            try:
                email_draft = _run_coro_blocking(_draft_email_with_gemini(context_str, doc_type, description, broker_name))
                if email_draft:
                    used_llm = True
            except Exception as e:
                logger.warning(f"Gemini draft failed, falling back to deterministic: {e}")

        # Fallback to deterministic if LLM failed or disabled
        if not email_draft:
            email_draft = _create_fallback_draft(doc_type, description, broker_name, deal_name)
            logger.info("Using deterministic fallback draft")

        # Step 3: Create artifacts
        out_dir = resolve_action_artifact_dir(ctx)
        artifacts = []

        # Artifact 1: Email draft
        draft_path = out_dir / "email_draft.md"
        draft_content = f"""# Document Request Email Draft

**To:** {broker_name}
**Subject:** {email_draft['subject']}

---

{email_draft['body']}

---

_Generated by ZakOps Action Engine. Review before sending._
_Drafted using: {'Gemini Flash' if used_llm else 'Deterministic Template'}_
"""
        draft_path.write_text(draft_content, encoding="utf-8")
        artifacts.append(
            ArtifactMetadata(
                filename=draft_path.name,
                mime_type="text/markdown",
                path=str(draft_path),
                created_at=now_utc_iso(),
            )
        )

        # Artifact 2: Checklist
        checklist = _get_deterministic_checklist(doc_type)
        checklist_path = out_dir / "document_checklist.md"
        checklist_content = f"""# Document Request Checklist ({doc_type})

## Request Summary
{description}

## Documents Requested
{chr(10).join(checklist)}

## Status
- [ ] Request sent to broker
- [ ] Documents received
- [ ] Documents reviewed

_Draft-only. Review before sharing externally._
"""
        checklist_path.write_text(checklist_content, encoding="utf-8")
        artifacts.append(
            ArtifactMetadata(
                filename=checklist_path.name,
                mime_type="text/markdown",
                path=str(checklist_path),
                created_at=now_utc_iso(),
            )
        )

        # Outputs
        outputs: Dict[str, Any] = {
            "doc_type": doc_type,
            "description": description,
            "broker_name": broker_name,
            "deal_name": deal_name,
            "email_subject": email_draft["subject"],
            "email_body_preview": email_draft["body"][:200] + "..." if len(email_draft["body"]) > 200 else email_draft["body"],
            "used_llm": used_llm,
            "draft_artifact": str(draft_path),
        }

        # Suggest follow-up action (but do NOT create it - operator must approve send)
        follow_up_suggestion = {
            "type": "COMMUNICATION.SEND_EMAIL",
            "title": f"Send document request to {broker_name}",
            "inputs": {
                "to": f"{broker_name} (broker email)",
                "subject": email_draft["subject"],
                "body_artifact_path": str(draft_path),
                "deal_id": payload.deal_id,
            },
            "requires_approval": True,
            "note": "This action requires explicit operator approval before sending.",
        }
        outputs["follow_up_suggestion"] = follow_up_suggestion

        return ExecutionResult(outputs=outputs, artifacts=artifacts)
