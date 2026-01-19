"""
Action Execution Engine

Executes approved actions with proper logging and error handling.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from uuid import UUID
from enum import Enum
import json
import logging

from ..database.adapter import get_database
from ..events import publish_action_event
from ..events.taxonomy import ActionEventType

logger = logging.getLogger(__name__)


class ActionStatus(str, Enum):
    """Action statuses."""
    PENDING_APPROVAL = "PENDING_APPROVAL"
    READY = "READY"
    QUEUED = "QUEUED"
    APPROVED = "approved"
    REJECTED = "REJECTED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ActionExecutor:
    """
    Executes approved actions.
    """

    async def execute(
        self,
        action_id: str,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute an approved action.

        Args:
            action_id: Action to execute
            trace_id: Trace ID for correlation

        Returns:
            Execution result
        """
        db = await get_database()

        # Get action
        action = await db.fetchrow(
            "SELECT * FROM zakops.actions WHERE action_id = $1",
            action_id
        )

        if not action:
            raise ValueError(f"Action not found: {action_id}")

        # Check if action is in an executable state
        status = action["status"]
        if status not in (ActionStatus.QUEUED.value, ActionStatus.APPROVED.value):
            raise ValueError(f"Action not in executable state: {status}")

        now = datetime.now(timezone.utc)
        deal_id = action.get("deal_id")

        # Mark as executing
        await db.execute(
            "UPDATE zakops.actions SET status = $2, updated_at = $3 WHERE action_id = $1",
            action_id, ActionStatus.EXECUTING.value, now
        )

        # Emit execution started event
        try:
            if deal_id:
                await publish_action_event(
                    action_id=UUID(action_id),
                    correlation_id=UUID(deal_id),
                    event_type=ActionEventType.EXECUTING.value,
                    event_data={
                        "action_id": action_id,
                        "deal_id": deal_id,
                        "action_type": action["action_type"],
                        "trace_id": trace_id
                    },
                    deal_id=UUID(deal_id)
                )
        except Exception as e:
            logger.warning(f"Failed to publish execution started event: {e}")

        try:
            # Execute based on action type
            result = await self._execute_action(action)

            # Mark as completed
            await db.execute(
                """
                UPDATE zakops.actions
                SET status = $2, outputs = $3, updated_at = $4
                WHERE action_id = $1
                """,
                action_id,
                ActionStatus.COMPLETED.value,
                json.dumps(result),
                datetime.now(timezone.utc)
            )

            # Emit completion event
            try:
                if deal_id:
                    await publish_action_event(
                        action_id=UUID(action_id),
                        correlation_id=UUID(deal_id),
                        event_type=ActionEventType.COMPLETED.value,
                        event_data={
                            "action_id": action_id,
                            "deal_id": deal_id,
                            "action_type": action["action_type"],
                            "success": True,
                            "trace_id": trace_id
                        },
                        deal_id=UUID(deal_id)
                    )
            except Exception as e:
                logger.warning(f"Failed to publish completion event: {e}")

            logger.info(f"Action {action_id} executed successfully")
            return result

        except Exception as e:
            # Mark as failed
            error_msg = str(e)[:500]
            await db.execute(
                """
                UPDATE zakops.actions
                SET status = $2, outputs = $3, updated_at = $4
                WHERE action_id = $1
                """,
                action_id,
                ActionStatus.FAILED.value,
                json.dumps({"error": error_msg}),
                datetime.now(timezone.utc)
            )

            # Emit failure event
            try:
                if deal_id:
                    await publish_action_event(
                        action_id=UUID(action_id),
                        correlation_id=UUID(deal_id),
                        event_type=ActionEventType.FAILED.value,
                        event_data={
                            "action_id": action_id,
                            "deal_id": deal_id,
                            "action_type": action["action_type"],
                            "error": error_msg,
                            "trace_id": trace_id
                        },
                        deal_id=UUID(deal_id)
                    )
            except Exception as event_error:
                logger.warning(f"Failed to publish failure event: {event_error}")

            logger.error(f"Action {action_id} failed: {error_msg}")
            raise

    async def _execute_action(self, action: dict) -> Dict[str, Any]:
        """Execute action based on type."""
        action_type = action["action_type"]
        inputs = action.get("inputs", {})
        if isinstance(inputs, str):
            try:
                inputs = json.loads(inputs)
            except:
                inputs = {}

        if action_type == "create_task":
            return await self._execute_create_task(action, inputs)

        elif action_type == "send_email":
            return await self._execute_send_email(action, inputs)

        elif action_type == "stage_change":
            return await self._execute_stage_change(action, inputs)

        elif action_type == "analyze_document":
            return await self._execute_analyze_document(action, inputs)

        elif action_type == "fetch_deal_info":
            return await self._execute_fetch_deal_info(action, inputs)

        else:
            # Default: mark as completed without specific execution
            return {
                "status": "completed",
                "message": f"Action type '{action_type}' completed",
                "inputs": inputs
            }

    async def _execute_create_task(self, action: dict, inputs: dict) -> Dict[str, Any]:
        """Execute create_task action."""
        # In a real system, this might create a task in an external system
        return {
            "status": "completed",
            "task_created": True,
            "title": inputs.get("title"),
            "deal_id": action.get("deal_id")
        }

    async def _execute_send_email(self, action: dict, inputs: dict) -> Dict[str, Any]:
        """Execute send_email action."""
        # In a real system, this would send via email provider
        logger.info(f"[MOCK] Sending email to {inputs.get('to')}: {inputs.get('subject')}")

        return {
            "status": "completed",
            "email_sent": True,
            "to": inputs.get("to"),
            "subject": inputs.get("subject")
        }

    async def _execute_stage_change(self, action: dict, inputs: dict) -> Dict[str, Any]:
        """Execute stage_change action."""
        from ..deals.workflow import get_workflow_engine

        deal_id = action.get("deal_id")
        if not deal_id:
            raise ValueError("stage_change action requires deal_id")

        engine = await get_workflow_engine()

        transition = await engine.transition_stage(
            deal_id=deal_id,
            new_stage=inputs.get("new_stage"),
            transitioned_by="action_executor",
            reason=inputs.get("reason", "Automated stage change from action")
        )

        return {
            "status": "completed",
            "from_stage": transition.from_stage,
            "to_stage": transition.to_stage
        }

    async def _execute_analyze_document(self, action: dict, inputs: dict) -> Dict[str, Any]:
        """Execute analyze_document action."""
        # In a real system, this would trigger document analysis
        return {
            "status": "completed",
            "analysis_triggered": True,
            "document_id": inputs.get("document_id")
        }

    async def _execute_fetch_deal_info(self, action: dict, inputs: dict) -> Dict[str, Any]:
        """Execute fetch_deal_info action."""
        db = await get_database()

        deal_id = inputs.get("deal_id") or action.get("deal_id")
        if not deal_id:
            raise ValueError("fetch_deal_info requires deal_id")

        deal = await db.fetchrow(
            "SELECT * FROM zakops.deals WHERE deal_id = $1",
            deal_id
        )

        if not deal:
            return {
                "status": "completed",
                "found": False,
                "deal_id": deal_id
            }

        return {
            "status": "completed",
            "found": True,
            "deal_id": deal_id,
            "deal_name": deal.get("canonical_name"),
            "stage": deal.get("stage"),
            "status": deal.get("status")
        }

    async def get_pending_actions(self, deal_id: Optional[str] = None, limit: int = 50) -> list:
        """Get pending actions that need execution."""
        db = await get_database()

        if deal_id:
            actions = await db.fetch(
                """
                SELECT * FROM zakops.actions
                WHERE status = $1 AND deal_id = $2
                ORDER BY created_at ASC
                LIMIT $3
                """,
                ActionStatus.QUEUED.value, deal_id, limit
            )
        else:
            actions = await db.fetch(
                """
                SELECT * FROM zakops.actions
                WHERE status = $1
                ORDER BY created_at ASC
                LIMIT $2
                """,
                ActionStatus.QUEUED.value, limit
            )

        return list(actions)


# Singleton instance
_executor: Optional[ActionExecutor] = None


async def get_action_executor() -> ActionExecutor:
    """Get action executor instance."""
    global _executor
    if _executor is None:
        _executor = ActionExecutor()
    return _executor
