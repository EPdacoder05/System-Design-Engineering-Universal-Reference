"""Unit tests for tools.opsmemory.agent.redactor."""

import pytest

from tools.opsmemory.agent.redactor import redact_text


def test_aws_access_key_is_redacted():
    text = "Use this key: AKIAIOSFODNN7EXAMPLE12"
    redacted, count = redact_text(text)
    assert "[REDACTED]" in redacted
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    assert count >= 1


def test_github_token_is_redacted():
    token = "ghp_" + "A" * 36
    text = f"Set GITHUB_TOKEN={token}"
    redacted, count = redact_text(text)
    assert "[REDACTED]" in redacted
    assert token not in redacted
    assert count >= 1


def test_text_without_secrets_unchanged():
    text = "This is a perfectly safe log message with no credentials."
    redacted, count = redact_text(text)
    assert redacted == text
    assert count == 0


def test_multiple_secrets_all_redacted():
    aws_key = "AKIAIOSFODNN7EXAMPLE12"
    gh_token = "ghs_" + "B" * 36
    text = f"aws_key={aws_key} github={gh_token}"
    redacted, count = redact_text(text)
    assert aws_key not in redacted
    assert gh_token not in redacted
    assert count >= 2


def test_api_key_pattern_is_redacted():
    text = "api_key: somevalue123"
    redacted, count = redact_text(text)
    assert "[REDACTED]" in redacted
    assert "somevalue123" not in redacted
    assert count >= 1


def test_token_assignment_is_redacted():
    text = "token=mysupersecrettoken"
    redacted, count = redact_text(text)
    assert "[REDACTED]" in redacted
    assert count >= 1


def test_private_key_block_is_redacted():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
    redacted, count = redact_text(text)
    assert "[REDACTED]" in redacted
    assert count >= 1
