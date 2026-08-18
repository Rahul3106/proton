"""
Settings: application configuration settings management per Section 18 requirements.
Handles environment variable configuration for LLM provider, API keys, output paths, and logging defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass
class Settings:
    """Configurable system settings for Proton-Agent."""

    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    temperature: float = 0.2
    output_dir: Path = Path("outputs")
    plots_dir: Path = Path("outputs/plots")
    reports_dir: Path = Path("outputs/reports")
    models_dir: Path = Path("outputs/models")

    def __post_init__(self) -> None:
        if not self.gemini_api_key:
            self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Retrieves process-wide application settings."""
    return Settings(
        llm_provider=os.environ.get("PROTON_LLM_PROVIDER", "gemini"),
        gemini_model=os.environ.get("PROTON_GEMINI_MODEL", "gemini-1.5-flash"),
    )
