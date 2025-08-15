# messaging.py
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class Message:
    sender: str
    recipient: str
    payload: Any

class MessengerBase:
    def send(self, recipient_func: Callable, message: Message):
        raise NotImplementedError
    
    def send_direct(self, recipient_func: Callable, message: Message):
        """Send a message directly to a recipient function."""
        return recipient_func(message.payload)
