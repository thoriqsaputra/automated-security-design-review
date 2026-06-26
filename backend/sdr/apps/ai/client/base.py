from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Generator, Union
from dataclasses import dataclass
import base64
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

class AIProvider(Enum):
    NVIDIA = "nvidia"
    OPENROUTER = "openrouter"
    ROUTELLM = "routellm"
    LOCAL = "local"

@dataclass
class AIModel:
    name: str
    provider: AIProvider
    max_tokens: int
    temperature: float = 0.1
    supports_json: bool = True
    cost_per_1k_tokens: Optional[float] = None
    is_embedding_model: bool = False

@dataclass
class AIResponse:
    content: str
    model: str
    provider: AIProvider
    usage: Optional[Dict[str, Any]] = None
    raw_usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    error_code: Optional[str] = None
    finish_reason: Optional[str] = None

class AIServiceInterface(ABC):
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, Any]], model: str, **kwargs) -> Union[AIResponse, Generator[str, None, None]]:
        pass
    
    @abstractmethod
    def get_embedding(self, text: str, model: str, dimensions: int = 1024) -> List[float]:
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str], model: str, dimensions: int = 1024) -> List[List[float]]:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[AIModel]:
        pass

def convert_to_langchain_messages(messages: List[Dict[str, Any]], image_bytes: Optional[bytes] = None, image_format: str = "png") -> List[BaseMessage]:
    lc_messages = []
    
    # Process image if provided (attach to the last user message)
    if image_bytes:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        for i in range(len(messages)-1, -1, -1):
            if messages[i].get("role") == "user":
                content = messages[i].get("content", "")
                messages[i]["content"] = [
                    {"type": "text", "text": str(content)},
                    {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image_b64}"}}
                ]
                break

    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    return lc_messages

def convert_to_openai_messages(messages: List[Dict[str, Any]], image_bytes: Optional[bytes] = None, image_format: str = "png") -> List[Dict[str, Any]]:
    # Process image if provided (attach to the last user message)
    if image_bytes:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        for i in range(len(messages)-1, -1, -1):
            if messages[i].get("role") == "user":
                content = messages[i].get("content", "")
                messages[i]["content"] = [
                    {"type": "text", "text": str(content)},
                    {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image_b64}"}}
                ]
                break
    return messages
