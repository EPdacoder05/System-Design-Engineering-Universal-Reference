# ============================================================================
# Pytest Fixtures Template
# ============================================================================
# Copy this to your project's tests/conftest.py
# Customize fixtures for your project's specific needs
# ============================================================================

"""Shared pytest fixtures for the test suite."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest


# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory for test isolation."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove sensitive environment variables for test isolation."""
    sensitive_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
        "TF_TOKEN",
    ]
    for var in sensitive_vars:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set up mock environment variables for testing."""
    env_vars = {
        "AWS_REGION": "us-east-1",
        "AWS_PROFILE": "default",
        "GITHUB_ORG": "your-org",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


# ---------------------------------------------------------------------------
# File fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config_file(tmp_workspace: Path) -> Path:
    """Create a sample configuration file for testing."""
    config = tmp_workspace / "config.json"
    config.write_text('{"key": "value", "nested": {"enabled": true}}')
    return config


@pytest.fixture
def sample_terraform_file(tmp_workspace: Path) -> Path:
    """Create a sample Terraform configuration file."""
    tf_file = tmp_workspace / "main.tf"
    tf_file.write_text('''
terraform {
  backend "s3" {
    bucket = "your-org-tfstate-bucket"
    key    = "project/terraform.tfstate"
    region = "us-east-1"
  }
}
'''.strip())
    return tf_file


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests requiring external services")
