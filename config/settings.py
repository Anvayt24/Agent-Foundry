from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# Simple constants-based settings
ENVIRONMENT: str = os.getenv("APP_ENV", "development")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Paths
RAG_DB_PATH: Path = Path(os.getenv("RAG_DB_PATH", BASE_DIR / "rag_db"))
MEM0_CHROMA_PATH: Path = Path(os.getenv("MEM0_CHROMA_PATH", BASE_DIR / ".mem0_chroma"))

# LLM (Gemini)
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Optional MCP server
MCP_TRANSPORT: str = os.getenv("MCP_TRANSPORT", "stdio")
MCP_SERVER_HOST: str = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT: int = int(os.getenv("MCP_SERVER_PORT", "8000"))

def ensure_google_key() -> None:
    """Ensure GOOGLE_API_KEY is set for langchain-google-genai usage."""
    if GEMINI_API_KEY and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
