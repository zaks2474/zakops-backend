"""
Tool Registry

Manages available tools for agent execution.
"""

from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from uuid import UUID
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition of an available tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable
    risk_level: str = "low"  # low, medium, high, critical
    requires_approval: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API/agent consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "risk_level": self.risk_level,
            "requires_approval": self.requires_approval
        }


class ToolRegistry:
    """Registry of available tools for agent execution."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default/built-in tools."""

        # Analysis tools (low risk)
        self.register(ToolDefinition(
            name="analyze_document",
            description="Analyze a document and extract key information",
            parameters={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "ID of document to analyze"},
                    "analysis_type": {"type": "string", "enum": ["summary", "key_terms", "entities", "full"]}
                },
                "required": ["document_id"]
            },
            handler=self._analyze_document,
            risk_level="low",
            requires_approval=False
        ))

        self.register(ToolDefinition(
            name="fetch_deal_info",
            description="Fetch information about a deal",
            parameters={
                "type": "object",
                "properties": {
                    "deal_id": {"type": "string", "description": "ID of deal"},
                    "include": {"type": "array", "items": {"type": "string"}, "description": "Fields to include"}
                },
                "required": ["deal_id"]
            },
            handler=self._fetch_deal_info,
            risk_level="low",
            requires_approval=False
        ))

        self.register(ToolDefinition(
            name="list_documents",
            description="List documents associated with a deal",
            parameters={
                "type": "object",
                "properties": {
                    "deal_id": {"type": "string", "description": "ID of deal"},
                    "category": {"type": "string", "description": "Filter by category"}
                },
                "required": ["deal_id"]
            },
            handler=self._list_documents,
            risk_level="low",
            requires_approval=False
        ))

        # Action tools (medium risk - create actions)
        self.register(ToolDefinition(
            name="create_task",
            description="Create a task/action for human review",
            parameters={
                "type": "object",
                "properties": {
                    "deal_id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]}
                },
                "required": ["deal_id", "title"]
            },
            handler=self._create_task,
            risk_level="medium",
            requires_approval=False  # Action itself will go through HITL
        ))

        self.register(ToolDefinition(
            name="suggest_stage_change",
            description="Suggest changing deal stage",
            parameters={
                "type": "object",
                "properties": {
                    "deal_id": {"type": "string"},
                    "new_stage": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["deal_id", "new_stage", "reason"]
            },
            handler=self._suggest_stage_change,
            risk_level="high",
            requires_approval=True
        ))

        # Communication tools (high risk)
        self.register(ToolDefinition(
            name="draft_email",
            description="Draft an email for human review",
            parameters={
                "type": "object",
                "properties": {
                    "deal_id": {"type": "string"},
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                },
                "required": ["deal_id", "to", "subject", "body"]
            },
            handler=self._draft_email,
            risk_level="high",
            requires_approval=True
        ))

    def register(self, tool: ToolDefinition):
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} (risk: {tool.risk_level})")

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas for agent consumption."""
        return [tool.to_dict() for tool in self._tools.values()]

    async def execute(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Any:
        """Execute a tool and return result."""
        tool = self.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Execute handler
        if asyncio.iscoroutinefunction(tool.handler):
            return await tool.handler(tool_input, context or {})
        else:
            return tool.handler(tool_input, context or {})

    # =========================================================================
    # Tool Handlers (Mock implementations - replace with real logic)
    # =========================================================================

    async def _analyze_document(self, input: Dict, context: Dict) -> Dict[str, Any]:
        """Mock document analysis."""
        return {
            "document_id": input.get("document_id"),
            "analysis_type": input.get("analysis_type", "summary"),
            "result": {
                "summary": "Document analysis placeholder",
                "key_points": ["Point 1", "Point 2", "Point 3"]
            }
        }

    async def _fetch_deal_info(self, input: Dict, context: Dict) -> Dict[str, Any]:
        """Fetch deal information."""
        from ..database.adapter import get_database

        try:
            db = await get_database()
            deal_id = input["deal_id"]

            # Try to parse as UUID, otherwise use as string
            deal = await db.fetchrow(
                "SELECT * FROM zakops.deals WHERE deal_id = $1",
                deal_id
            )

            if deal:
                return dict(deal)
            return {"error": "Deal not found", "deal_id": deal_id}
        except Exception as e:
            logger.warning(f"Failed to fetch deal info: {e}")
            return {"error": str(e), "deal_id": input.get("deal_id")}

    async def _list_documents(self, input: Dict, context: Dict) -> Dict[str, Any]:
        """List deal documents."""
        # Mock implementation - artifacts table may not exist yet
        return {
            "deal_id": input["deal_id"],
            "documents": [],
            "note": "Document listing not yet implemented"
        }

    async def _create_task(self, input: Dict, context: Dict) -> Dict[str, Any]:
        """Create a task action."""
        # This will be handled by action creation
        return {
            "action_type": "create_task",
            "deal_id": input["deal_id"],
            "title": input["title"],
            "description": input.get("description", ""),
            "priority": input.get("priority", "medium"),
            "status": "pending_creation"
        }

    async def _suggest_stage_change(self, input: Dict, context: Dict) -> Dict[str, Any]:
        """Suggest a stage change."""
        return {
            "action_type": "stage_change",
            "deal_id": input["deal_id"],
            "new_stage": input["new_stage"],
            "reason": input["reason"],
            "status": "pending_approval"
        }

    async def _draft_email(self, input: Dict, context: Dict) -> Dict[str, Any]:
        """Draft an email."""
        return {
            "action_type": "send_email",
            "deal_id": input["deal_id"],
            "to": input["to"],
            "subject": input["subject"],
            "body": input["body"],
            "status": "pending_approval"
        }


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
