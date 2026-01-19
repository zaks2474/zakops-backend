"""
ZakOps Agent Bridge Configuration

Centralized configuration management for the bridge service.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class BridgeConfig:
    """Configuration for the ZakOps Agent Bridge."""

    # Server settings
    HOST: str = os.getenv("ZAKOPS_BRIDGE_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("ZAKOPS_BRIDGE_PORT", "9100"))
    API_KEY: str = os.getenv("ZAKOPS_BRIDGE_API_KEY", "")

    # Backend service URLs
    DEAL_API_URL: str = os.getenv("ZAKOPS_DEAL_API_URL", "http://localhost:8090")
    RAG_API_URL: str = os.getenv("ZAKOPS_RAG_API_URL", "http://localhost:8052")

    # Filesystem paths
    DATAROOM_ROOT: Path = Path(os.getenv("DATAROOM_ROOT", "/home/zaks/DataRoom"))

    @property
    def pipeline_path(self) -> Path:
        return self.DATAROOM_ROOT / "00-PIPELINE" / "Inbound"

    @property
    def quarantine_path(self) -> Path:
        return self.DATAROOM_ROOT / "00-PIPELINE" / "_INBOX_QUARANTINE"

    @property
    def registry_path(self) -> Path:
        return self.DATAROOM_ROOT / ".deal-registry"

    @property
    def log_path(self) -> Path:
        return self.registry_path / "logs" / "agent_bridge.jsonl"

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if not self.API_KEY:
            issues.append("WARNING: No API key configured (ZAKOPS_BRIDGE_API_KEY)")

        if not self.DATAROOM_ROOT.exists():
            issues.append(f"ERROR: DataRoom not found at {self.DATAROOM_ROOT}")

        return issues


# Global config instance
config = BridgeConfig()
