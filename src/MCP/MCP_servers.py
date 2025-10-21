from fastmcp import FastMCP
from pathlib import Path

app = FastMCP("AgentFoundry-MCP")

@app.tool
def file_search(root: str = ".", pattern: str = "*.md") -> str:
    """
    Searches for files matching a glob pattern within a root directory.
    Use this to find files when you know the directory and a pattern like '*.py' or 'worker.py'.
    Parameters:
      - root: The directory to start the search from (e.g., '.').
      - pattern: The file pattern to search for (e.g., 'worker.py', '*.txt').
    """
    root_path = Path(root)
    matches = list(root_path.rglob(pattern))
    return "\n".join(str(m) for m in matches)

@app.tool
def read_file(path: str, max_chars: int = 5000) -> str:
    """
    Reads the contents of a single text file at a given path.
    Use this to inspect the content of a file you have already found.
    Parameters:
      - path: The full path to the file (e.g., 'src/agents/worker.py').
    """
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
    """
    Saves text content to a file at a specified path. Creates the file if it doesn't exist.
    Use this to write new scripts, save plans, or record results.
    Parameters:
      - path: The full path for the file to be saved (e.g., 'output/results.txt').
      - content: The text content to write into the file.
    """
    p = Path(path)           #Save content to a file
    try:
        p.write_text(content, encoding="utf-8")
        return f"[Success] Saved file at {p.resolve()}"
    except Exception as e:
        return f"[Error] Could not save file: {e}"

if __name__ == "__main__":
    # Default to stdio transport so clients (like our LangChain adapter) can spawn this server.
    app.run(transport="stdio")

