from mem0 import Memory
import os
from dotenv import load_dotenv

load_dotenv()


class MemoryManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryManager, cls).__new__(cls)
            cls._instance.initialize_memory()
        return cls._instance

    def initialize_memory(self):
        """Initialise Mem0 OSS memory with Gemini as the LLM."""
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found. Add it to your .env file so Mem0 can call Gemini.")

        config = {
            "llm": {
                "provider": "gemini",
                "config": {
                    "model": "gemini-2.5-flash",
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                },
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "mem0_agentfoundry_384",
                    "path": os.path.join(os.getcwd(), ".mem0_chroma"),
                },
            },
        }

        # Keep a flag to control inference depending on LLM availability
        self.infer_enabled = True
        self.memory = Memory.from_config(config)

    def add_memory(self, content, user_id: str = "agent_system", metadata: dict = None):
        """Add a memory entry via Mem0."""
        if metadata is None:
            metadata = {}

        try:
            if isinstance(content, str):
                content = [{"role": "user", "content": content}]
            return self.memory.add(content, user_id=user_id, metadata=metadata, infer=self.infer_enabled)
        except Exception as exc:
            print(f"Mem0 add_memory error: {exc}")
            return None

    def search_memories(self, query: str, user_id: str = "agent_system", limit: int = 5):
        """Search stored memories for the user."""
        if not query or not query.strip():
            return []
        try:
            results = self.memory.search(query=query, user_id=user_id)
        except Exception as exc:
            print(f"Mem0 search_memories error: {exc}")
            return []

        if not results:
            return []

        if isinstance(results, dict) and "results" in results:
            items = results["results"]
            memories = [item["memory"] for item in items if isinstance(item, dict) and "memory" in item]
            return memories[:limit]

        return results

memory_manager = MemoryManager()
