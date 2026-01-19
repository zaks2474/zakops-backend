from __future__ import annotations

from typing import Dict, List, Optional

from .base import ActionExecutor
from .tool_invoke import ToolInvokeExecutor


_EXECUTORS: Dict[str, ActionExecutor] = {}
_BUILTINS_LOADED = False
_TOOL_EXECUTOR: Optional[ToolInvokeExecutor] = None


def register_executor(executor: ActionExecutor) -> None:
    action_type = (executor.action_type or "").strip()
    if not action_type:
        raise ValueError("executor_missing_action_type")
    if action_type in _EXECUTORS:
        raise ValueError(f"duplicate_executor_for_action_type:{action_type}")
    _EXECUTORS[action_type] = executor


def load_builtin_executors() -> None:
    """
    Import and register built-in executors.

    This is intentionally lazy to keep import side-effects controlled for tests.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    from .analysis_build_valuation_model import BuildValuationModelExecutor
    from .communication_draft_email import DraftEmailExecutor
    from .communication_send_email import SendEmailExecutor
    from .deal_append_email_materials import AppendEmailMaterialsExecutor
    from .deal_create_from_email import CreateDealFromEmailExecutor
    from .deal_dedupe_and_place_materials import DedupeAndPlaceMaterialsExecutor
    from .deal_enrich_materials import EnrichMaterialsExecutor
    from .deal_extract_email_artifacts import ExtractEmailArtifactsExecutor
    from .deal_backfill_sender_history import DealBackfillSenderHistoryExecutor
    from .diligence_request_docs import RequestDocsExecutor
    from .document_generate_loi import GenerateLoiExecutor
    from .email_triage_reject_email import EmailTriageRejectEmailExecutor
    from .email_triage_review_email import EmailTriageReviewEmailExecutor
    from .presentation_generate_pitch_deck import GeneratePitchDeckExecutor
    from .rag_reindex_deal import RagReindexDealExecutor

    register_executor(DraftEmailExecutor())
    register_executor(SendEmailExecutor())
    register_executor(RequestDocsExecutor())
    register_executor(GenerateLoiExecutor())
    register_executor(BuildValuationModelExecutor())
    register_executor(GeneratePitchDeckExecutor())
    register_executor(EmailTriageReviewEmailExecutor())
    register_executor(EmailTriageRejectEmailExecutor())
    register_executor(CreateDealFromEmailExecutor())
    register_executor(AppendEmailMaterialsExecutor())
    register_executor(ExtractEmailArtifactsExecutor())
    register_executor(EnrichMaterialsExecutor())
    register_executor(DedupeAndPlaceMaterialsExecutor())
    register_executor(RagReindexDealExecutor())
    register_executor(DealBackfillSenderHistoryExecutor())

    _BUILTINS_LOADED = True


def get_executor(action_type: str) -> Optional[ActionExecutor]:
    if not _BUILTINS_LOADED:
        load_builtin_executors()
    at = (action_type or "").strip()
    direct = _EXECUTORS.get(at)
    if direct is not None:
        return direct
    if at.upper().startswith("TOOL."):
        global _TOOL_EXECUTOR
        if _TOOL_EXECUTOR is None:
            _TOOL_EXECUTOR = ToolInvokeExecutor()
        return _TOOL_EXECUTOR
    return None


def list_executors() -> List[str]:
    if not _BUILTINS_LOADED:
        load_builtin_executors()
    return sorted(_EXECUTORS.keys())
