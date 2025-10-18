# 🚀 AgentFoundry  
*Forge, orchestrate, and extend intelligent multi-agent systems with RAG, MCP, and A2A communication.*  

---
## 🧠 What is AgentFoundry?

AgentFoundry is a developer-first framework for building and experimenting with collaborative multi-agent systems. It provides a sandbox and a framework, giving developers both flexibility and power to:

- Build domain-specific intelligent agents,
- Connect them through **shared memory** and **A2A protocols**,
- Integrate external capabilities using **MCP servers** and
- Extend the system without rewriting the core.

AgentFoundry is designed for researchers, ML engineers, and AI developers who want to move beyond single-agent pipelines and explore true agentic cooperation.

---

## 🚀 Why AgentFoundry?

AgentFoundry is designed for developers who want full control over their agentic systems. It provides a modular, transparent, and hackable framework that allows you to:

- **Prototype your own AI orchestration logic**
- **Plug in custom tools or APIs dynamically**
- **Build domain-specific AI agents** (e.g., medical assistants, research copilots, or autonomous workflow bots)
- **Experiment with A2A (Agent-to-Agent) collaboration and distributed reasoning**

Each component in AgentFoundry is replaceable, extensible, and traceable. You have full control over the planning logic, message routing, RAG retrieval, and MCP-based tool integration. This means you can easily customize and experiment with different architectures and use cases.

## 🎯 Key Features

- **Planner Agent**: Breaks down complex tasks into subtasks.
- **Worker Agents**: Execute the subtasks.
- **Verifier Agent**: Validates outputs for correctness.
- **RAG Integration**: Dynamic retrieval via agents.
- **MCP Server Integration**: Enables external tools.
- **Memory Layer**: Central memory module managing shared state and data access across agents.
- **Extensibility**: Easily add more tools (APIs, DBs, system commands) via MCP.
- **Centralized or Decentralized control**: Choose between a central orchestrator or decentralized A2A communication between agents.


---

### Agent Flow (general):  
1. **User Input**: Direct command via interactive CLI
2. **Planner**: Analyzes request and creates task plan
3. **Worker**: Executes tasks using available tools:
   - **MCP Tools**: `file_search`, `read_file`, `save_file`(mock tools present in this repo)
   - **RAG Tool**: Knowledge base retrieval when needed
4. **Verifier**: Validates and refines outputs
5. **Real-time Response**: Live feedback and results to user

---

## 🛠️ Tech Stack  
- **Python**  
- **LangChain** – agent orchestration  
- **Google Gemini LLM** – reasoning & generation  
- **Chroma / FAISS** – vector store for RAG  
- **FastMCP** – lightweight MCP server  
- **MCP Toolkit for LangChain** – dynamic tool integration  
- **Mem0** - memory layer for agents

---

## 📂 Project Structure

```text
agentfoundry/
├── scripts/
│   ├── a2a_network.py
│   └── orchestrator.py
├── src/
│   ├── agents/
│   │   ├── planner.py
│   │   ├── worker.py
│   │   └── verifier.py
│   ├── core/
│   │   ├── central.py
│   │   └── messaging.py
│   ├── RAG/
│   │   ├── load_docs.py
│   │   ├── retriever.py
│   │   ├── rag_tool.py
│   │   └── vector_store.py
│   ├── MCP/
│   │   ├── MCP_servers.py
│   │   └── mcp_tools_adapter.py
│   ├── memory/
│   │   └── memory_manager.py
│   └── config/
│       ├── __init__.py
│       └── settings.py
├── rag_db/
├── pyproject.toml
├── README.md
├── .env
├── .gitignore
└── requirements.txt
```


## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/AgentFoundry.git
cd AgentFoundry
```

### 2. Install (editable)

```bash
pip install -e .
```

### 3. Run MCP Server (only when running over a different port)

```bash
python -m MCP.MCP_servers
```

### 4. Run the Orchestrator

```bash
python scripts/orchestrator.py
```
**OR**

### 5. Run the A2A Network

```bash
python scripts/a2a_network.py
```

### Environment Setup
- **.env**: define `GOOGLE_API_KEY` or `GEMINI_API_KEY`, plus any other provider keys you enable.
- **Paths**: override `RAG_DB_PATH` (defaults to `rag_db/`) and `MEM0_CHROMA_PATH` (defaults to `.mem0_chroma/`) if you want custom storage locations.
- **MCP transport**: switch from the default `stdio` by setting `MCP_TRANSPORT`, `MCP_SERVER_HOST`, and `MCP_SERVER_PORT`.

## 🤝 Contribution

AgentFoundry is developer-facing — contributions are welcome

Feel free to open issues or submit pull requests!
