# 🚀 AgentFoundry  
*A multi-agent Agentic RAG framework with MCP server integration*  

---

## 📌 Overview  
**AgentFoundry** is a developer-facing framework for building **multi-agent systems** that combine:  
- **Agentic RAG (Retrieval-Augmented Generation with agents)** where agents   **plan, decide, and execute retrieval calls** instead of just stuffing context.  
- **MCP (Model Context Protocol) tools** for interacting with external environments 

The goal is to provide a **minimal but extensible** framework for experimenting with **Agentic RAG + MCP integration** — showing how agents can plan, retrieve, act, and validate.  

---

## 🎯 Key Features
- ✅ **Planner Agent** – breaks down complex tasks into subtasks.  
- ✅ **Worker Agents** – execute subtasks using **RAG** and **MCP tools**.  
- ✅ **Verifier Agent** – validates outputs for correctness.  
- ✅ **Agentic RAG Integration** – dynamic retrieval via agents (Planner & Worker use RAG tool only when needed).
- ✅ **MCP Server Integration** – enables external tools 
- ✅ **Normal Orchestration** – central orchestrator coordinating Planner → Worker → Verifier.  
- ✅ **Extensible** – easily add more tools (APIs, DBs, system commands) via MCP.  
- ✅ **Agent-to-Agent (A2A) Communication** – direct agent communication via shared MessageBus with:
  - **Asynchronous message passing** between agents
  - **Cooperative task execution** with idle detection
  - **Tool-aware agents** that can use MCP and RAG tools dynamically
  - **Real-time interaction** with live user input
  - **Error handling** with timeouts and idle detection
  - **Message Bus** for efficient inter-agent communication

---

## 🏗️ Current Status  
✔️ **Phase 1:** RAG pipeline working (retriever + Gemini LLM).  
✔️ **Phase 2:** Wrapped RAG into a LangChain Tool (`rag_answer_tool`).  
✔️ **Phase 3:** Multi-agent flow (Planner, Worker, Verifier) built with LangChain.  
✔️ **Phase 4:** MCP server added with tools and integrated with Worker.  
✔️ **Phase 5:** A2A communication mode implemented with MessageBus.  

🔜 **Next planned improvements**:  
- Add more MCP tools.
- Streamlit/CLI interface for developer interaction.  
- Optional scaling with Ray Serve.  

---

### Agent Flow :  

#### Traditional Orchestrator Mode:
1. **Planner**: Breaks into subtasks → (search, read, summarize). Decides that **retrieval from knowledge base may be needed** for context.  
2. **Worker**:  
   - Calls MCP `file_search("*.md")`  
   - Calls MCP `read_file("README.md")`  
   - Dynamically decides to call the **`rag_answer_tool`** to fetch additional context from the vector store.  
   - Combines retrieved knowledge + file content and summarizes using LLM.  
3. **Verifier**: Checks that the retrieved knowledge and summary are consistent and accurate.  
4. **Final Output**: A validated, knowledge-grounded summary of README.md.

#### A2A Network Mode:
1. **User Input**: Direct command via interactive CLI
2. **Planner Agent**: Analyzes request and creates task plan
3. **Message Bus**: Distributes subtasks to Worker agent
4. **Worker Agent**: Executes tasks using available tools:
   - **MCP Tools**: `file_search`, `read_file`, `save_file`
   - **RAG Tool**: Knowledge base retrieval when needed
5. **Verifier Agent**: Validates and refines outputs
6. **Real-time Response**: Live feedback and results to user

**A2A Advantages:**
- **Interactive Development**: Test agent capabilities in real-time
- **Tool Discovery**: Agents dynamically select appropriate tools
- **Error Recovery**: Graceful handling of tool failures and network issues
- **Extensible**: Easy to add new agents and tools to the network 

---

## 🛠️ Tech Stack  
- **Python**  
- **LangChain** – agent orchestration  
- **Google Gemini LLM** – reasoning & generation  
- **Chroma / FAISS** – vector store for RAG  
- **FastMCP** – lightweight MCP server  
- **MCP Toolkit for LangChain** – dynamic tool integration  

---

## 📂 Project Structure

```text
agentfoundry/
├── MCP/
│   ├── MCP_servers.py
│   └── mcp_tools_adapter.py
├── RAG/
│   ├── agentic_rag.py
│   ├── load_docs.py
│   ├── retriever.py
│   ├── rag_tool.py
│   └── vector_store.py
├── agents/
│   ├── planner.py
│   ├── worker.py
│   └── verifier.py
├── rag_db/
├── central.py
├── orchestrator.py
├── messaging.py
├── a2a_network.py
├── requirements.txt
├── pyproject.toml
├── README.md
├── .env
├── .gitignore
└── sample.txt
```


## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/AgentFoundry.git
cd AgentFoundry
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run MCP Server (Tools available on port 8000/stdio)

```bash
python MCP/MCP_servers.py
```

### 4. Run the Orchestrator

```bash
python orchestrator.py
```

### 5. Run the A2A Network

```bash
python a2a_network.py
```

## 🤝 Contribution

AgentFoundry is developer-facing — contributions are welcome for:

- Adding new MCP tools
- A2A communication 
- Enhancing scalability & observability

Feel free to open issues or submit pull requests!
