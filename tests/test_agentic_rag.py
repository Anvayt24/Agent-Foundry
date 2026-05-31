"""Smoke tests for the agentic RAG wiring.

These tests avoid live LLM/network calls. They verify that the RAG tool is
wrapped correctly and that the agent factory produces a usable executor.
Tests that require a configured ``GEMINI_API_KEY`` are skipped when it is absent.
"""
import os
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

requires_api_key = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; skipping tests that build LLM-backed agents.",
)


def test_rag_tool_is_callable():
    """The RAG tool wrapper should be importable and callable."""
    from RAG.rag_tool import rag_tool

    assert callable(rag_tool)


@requires_api_key
def test_worker_agent_builds():
    """The worker factory should return an executor exposing invoke()."""
    from agents.worker import create_worker

    executor = create_worker()
    assert hasattr(executor, "invoke")


@requires_api_key
def test_planner_agent_builds():
    from agents.planner import create_planner

    executor = create_planner()
    assert hasattr(executor, "invoke")
