from mem0 import Memory

class MemoryManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryManager, cls).__new__(cls)
            cls._instance.initialize_memory()
        return cls._instance
    
    def initialize_memory(self):
        """Initialization of the memory"""
        self.memory = Memory()
    
    def add_memory(self, content: str, user_id: str = "agent_system", metadata: dict = None):
        """Add a memory"""
        if metadata is None:
            metadata = {}
        self.memory.add(content, user_id=user_id, metadata=metadata)
    
    def search_memories(self, query: str, user_id: str = "agent_system", limit: int = 5):
        """Search for relevant memories."""
        results = self.memory.search(query=query, user_id=user_id, limit=limit)
        return [item['memory'] for item in results['results']] if results else []

memory_manager = MemoryManager()
