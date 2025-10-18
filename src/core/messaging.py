from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict
from queue import Queue
import threading
import time


class MessageType(Enum):
    TASK_REQUEST = "TASK_REQUEST"
    TASK_RESPONSE = "TASK_RESPONSE"
    TASK_RESULT = "TASK_RESULT"
    ERROR = "ERROR"


@dataclass
class Message:
    sender: str
    recipient: str
    message_type: MessageType
    payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MessageBus:
    """Core A2A communication bus using thread-safe queues."""

    def __init__(self):
        self.queues: Dict[str, Queue] = {}
        self.lock = threading.Lock()

    def register_agent(self, agent_name: str):
        with self.lock:
            if agent_name not in self.queues:
                self.queues[agent_name] = Queue()

    def send(self, message: Message):
        with self.lock:
            if message.recipient not in self.queues:
                raise ValueError(f"Recipient {message.recipient} not registered")
            self.queues[message.recipient].put(message)

    def receive(self, agent_name: str, timeout: float = None) -> Message | None:
        if agent_name not in self.queues:
            raise ValueError(f"Agent {agent_name} not registered")
        try:
            return self.queues[agent_name].get(timeout=timeout)
        except Exception:
            return None
