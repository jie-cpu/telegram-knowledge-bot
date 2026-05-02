"""Shared fixtures for meta-tests."""

import pytest


@pytest.fixture
def orchestrator_config():
    """Default orchestrator config used across tests."""
    return {
        "model": "gpt-4o-mini",
        "max_iterations": 50,
        "timeout_minutes": 120,
        "task_decomposition_depth": 3,
        "escalation_threshold": 5,
        "checkpoint_interval_seconds": 300,
        "heartbeat_interval_seconds": 60,
        "max_missed_heartbeats": 3,
    }
