from fastmcp import FastMCP
from pathlib import Path

app = FastMCP("AgentFoundry-MCP")

@app.tool
def file_search(root: str = ".", pattern: str = "*.md") -> str:
    #Return a newline separated list of matching file paths
    root_path = Path(root)
    matches = list(root_path.rglob(pattern))
    return "\n".join(str(m) for m in matches)

@app.tool
def read_file(path: str, max_chars: int = 5000) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():      # Read the contents of a text file 
        return f"[Error] File not found: {path}"
    try:
        content = p.read_text(encoding="utf-8")
        return content[:max_chars] + ("..." if len(content) > max_chars else "")
    except Exception as e:
        return f"[Error] Could not read {path}: {e}"

@app.tool
def save_file(path: str, content: str) -> str:
    p = Path(path)           #Save content to a file
    try:
        p.write_text(content, encoding="utf-8")
        return f"[Success] Saved file at {p.resolve()}"
    except Exception as e:
        return f"[Error] Could not save file: {e}"

if __name__ == "__main__":
    # Default to stdio transport so clients (like our LangChain adapter) can spawn this server.
    app.run(transport="stdio")

