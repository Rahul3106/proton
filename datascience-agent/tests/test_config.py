"""
Tests for configuration settings management.
"""
from proton_agent.config import Settings, get_settings


def test_settings_initialization():
    settings = get_settings()
    assert settings.output_dir.exists()
    assert settings.plots_dir.exists()
    assert settings.reports_dir.exists()
    assert settings.models_dir.exists()
