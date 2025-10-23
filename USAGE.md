# Usage

Concise examples of how the agent runs in the terminal, how to add a new MCP tool, and how to tune behavior. Setup is already covered in `README.md`.

---

## Run Modes (Terminal Transcripts)

- **Centralized Orchestrator** (`scripts/orchestrator.py`)

```bash
python scripts/orchestrator.py
```

Example:
```text
User: Summarize the content of sample.txt and list 3 key points.
processing by planner worker and verifier.
Final Answer:
- <concise point 1>
- <concise point 2>
- <concise point 3>
```

- **Agent-to-Agent (A2A) network** (`scripts/a2a_network.py`)

```bash
python scripts/a2a_network.py
```

Example:
```text
User (A2A)> Find occurrences of the word "Agent" in sample.txt and explain the context.

planner and worker running and executing the tasks by calling rag_tool, read_file,etc tools.
verifier verifying the results.

Final Answer:
- Found N occurrences of "Agent".
- Context summary: <brief explanation of where/how it appears>

User (A2A)> exit
```


## Add a New MCP Tool

Add a tool in `src/MCP/MCP_servers.py`. Tools are auto-discovered and loaded by the Worker via `load_mcp_tools()`.

```python
# src/MCP/MCP_servers.py
from fastmcp import FastMCP
from pathlib import Path

app = FastMCP("AgentFoundry-MCP")

@app.tool
def your_tool_name(param1: type, param2: type) -> return_type:
    """Description of what your tool does."""
    # Implement your tool logic here
    # Example:
    # result = perform_operation(param1, param2)
    # return result

if __name__ == "__main__":
    app.run(transport="stdio")
```

- After saving, run your agent again. The tool will be available by its exact name to the Worker.
- The Worker auto-loads MCP tools in `src/agents/worker.py` via `load_mcp_tools()`.

---

## Tuning Examples (Code)

- **A2A loop timing** — `scripts/a2a_network.py`

```python
# Increase limits to allow more agent turns
max_idle_rounds = 15
end_time = time.time() + 20.0
```

- **ReAct iteration budget** — `src/core/central.py`

```python
# AgentExecutor(..., max_iterations=3) -> increase for more tool/use reasoning
return AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
)
```


- **Verifier style** — `src/agents/verifier.py`
  - Adjust `VERIFIER_SYSTEM_PROMPT` rules or add tools in `create_verifier()` for stricter checks or different summarization.

- **Planner behavior** — `src/agents/planner.py`
  - Modify `PLANNER_SYSTEM_PROMPT` or the `plan_task` function to change subtask granularity.
