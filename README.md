# 🚀 AgentFoundry  
*A general-purpose multi-agent Agentic RAG framework with MCP server integration*  

---

## 📌 Overview  
**AgentFoundry** is a developer-facing framework for building **multi-agent systems** that combine:  
- **Agentic RAG (Retrieval-Augmented Generation with agents)** where agents   **plan, decide, and execute retrieval calls** instead of just stuffing context.  
- **MCP (Model Context Protocol) tools** for interacting with external environments 
- **Agent orchestration** with Planner, Worker, and Verifier agents.  

The goal is to provide a **minimal but extensible** framework for experimenting with **Agentic RAG + MCP integration** — showing how agents can plan, retrieve, act, and validate.  

---

## 🎯 Key Features  
- ✅ **Planner Agent** – breaks down complex tasks into subtasks.  
- ✅ **Worker Agents** – execute subtasks using **RAG** and **MCP tools**.  
- ✅ **Verifier Agent** – validates outputs for correctness.  
- ✅ **Agentic RAG Integration** – dynamic retrieval via agents (Planner & Worker use RAG tool only when needed).
- ✅ **MCP Server Integration** – enables external tools like:  
  - `file_search` → find files in a repo  
  - `read_file` → read file contents  
  - `save_file` → write generated output back to disk  
- ✅ **Normal Orchestration** – central orchestrator coordinating Planner → Worker → Verifier.  
- ✅ **Extensible** – easily add more tools (APIs, DBs, system commands) via MCP.  

---

## 🏗️ Current Status  
✔️ **Phase 1:** RAG pipeline working (retriever + Gemini LLM).  
✔️ **Phase 2:** Wrapped RAG into a LangChain Tool (`rag_answer_tool`).  
✔️ **Phase 3:** Multi-agent flow (Planner, Worker, Verifier) built with LangChain.  
✔️ **Phase 4:** MCP server added with **file ops tools** and integrated with Worker.  

🔜 **Next planned improvements**:  
- Add more MCP tools.
-  A2A (Agent-to-Agent) communication mode in v2  
- Streamlit/CLI interface for developer interaction.  
- Optional scaling with Ray Serve.  

---

### Agent Flow :  
1. **Planner**: Breaks into subtasks → (search, read, summarize). Decides that **retrieval from knowledge base may be needed** for context.  
2. **Worker**:  
   - Calls MCP `file_search("*.md")`  
   - Calls MCP `read_file("README.md")`  
   - Dynamically decides to call the **`rag_answer_tool`** to fetch additional context from the vector store.  
   - Combines retrieved knowledge + file content and summarizes using LLM.  
3. **Verifier**: Checks that the retrieved knowledge and summary are consistent and accurate.  
4. **Final Output**: A validated, knowledge-grounded summary of README.md. 

---

## 🛠️ Tech Stack  
- **Python**  
- **LangChain** – agent orchestration  
- **Google Gemini LLM** – reasoning & generation  
- **Chroma / FAISS** – vector store for RAG  
- **FastMCP** – lightweight MCP server  
- **MCP Toolkit for LangChain** – dynamic tool integration  

---

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

### 3. Run MCP Server (Tools available on port 8000)

```bash
python mcp_server.py
```

### 4. Run the Orchestrator

```bash
python orchestrator.py
```

---

## 🤝 Contribution

AgentFoundry is developer-facing — contributions are welcome for:

- Adding new MCP tools
- A2A communication 
- Enhancing scalability & observability

Feel free to open issues or submit pull requests!



