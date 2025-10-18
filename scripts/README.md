# Scripts

This directory contains the executable entry points for AgentFoundry.

## Available Modes
- **Centralized orchestrator (`orchestrator.py`)**
  - Planner, Worker, and Verifier run under a single coordinator.
  - Best for deterministic, single-shot tasks or automated pipelines.
- **Agent-to-Agent network (`a2a_network.py`)**
  - Planner, Worker, and Verifier communicate directly via the message bus.
  - Supports interactive sessions and per-request `session_id` isolation.

Both modes share the same agent factories, toolset, and RAG/MCP integrations. Use whichever suits your automation or experimentation needs.

## Running the Scripts
```bash
# Centralized orchestrator
python scripts/orchestrator.py

# Agent-to-Agent interactive loop
python scripts/a2a_network.py
```
