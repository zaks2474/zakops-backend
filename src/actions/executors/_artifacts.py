from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from actions.engine.models import ActionError
from actions.executors.base import ActionExecutionError, ExecutionContext


def _dataroom_root() -> Path:
    return Path(os.getenv("DATAROOM_ROOT", "/home/zaks/DataRoom"))


def resolve_action_artifact_dir(ctx: ExecutionContext) -> Path:
    """
    Return the directory where this action should write artifacts:
    {deal.folder_path}/99-ACTIONS/{action_id}/
    """
    deal = ctx.deal or {}
    folder_path = (deal.get("folder_path") or "").strip()
    if not folder_path:
        raise ActionExecutionError(
            ActionError(
                code="deal_folder_path_missing",
                message="Deal folder_path missing; cannot determine artifact destination",
                category="validation",
                retryable=False,
            )
        )

    action_id = ctx.action.action_id
    base = (_dataroom_root() / folder_path / "99-ACTIONS" / action_id).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base
