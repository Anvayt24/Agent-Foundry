"""Path bootstrap so scripts run from any working directory.

Adds the project's ``src/`` directory to ``sys.path`` so the package modules
(``agents``, ``core``, ``config``, ``RAG``, ``MCP``, ``memory``) resolve whether
or not the package has been installed with ``pip install -e .``.
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
