"""Pydantic schemas for OpenAI-compatible CAMIR serving gateway."""

import time
import uuid
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="camir-auto", description="Model identifier or 'camir-auto' for routing")
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.0
    stream: bool = False
    policy: Optional[str] = Field(default="P5", description="Routing policy: 'P2', 'P3', 'P4', 'P5', 'always-small'")


class CompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo
    camir_metadata: Optional[Dict[str, Any]] = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "camir"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]


class HealthResponse(BaseModel):
    status: str
    active_policy: str
    hardware_profile: str
    hardware_info: str
    thermal_state: str
    mock_mode: bool
