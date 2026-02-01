"""
OpenClaw - LLM Provider Interface
=================================
Abstract base class and implementations for Claude, Gemini, and Groq.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator
from enum import Enum
import json
import os
import asyncio
import re
import time
import random

import httpx
from rich.console import Console

console = Console()


class ProviderType(Enum):
    CLAUDE = "claude"
    GEMINI = "gemini"
    GROQ = "groq"


@dataclass
class Message:
    """Represents a conversation message."""
    role: str  # "user" | "assistant" | "system"
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)


@dataclass
class ToolDefinition:
    """Defines a tool that can be called by the LLM."""
    name: str
    description: str
    parameters: dict[str, Any]
    
    def to_claude_format(self) -> dict:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }
    
    def to_openai_format(self) -> dict:
        """Convert to OpenAI/Groq tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    def to_gemini_format(self) -> dict:
        """Convert to Google Gemini tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


@dataclass
class LLMResponse:
    """Standardized response from LLM providers."""
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict = field(default_factory=dict)
    raw_response: Any = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = config.get("model")
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.7)
    
    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system_prompt: str | None = None
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system_prompt: str | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream a response from the LLM."""
        pass


class GroqProvider(BaseLLMProvider):
    """Groq provider implementation (free tier available)."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable required")
        self.base_url = "https://api.groq.com/openai/v1"
        
        # Default to llama if no model specified
        if not self.model:
            self.model = "llama-3.3-70b-versatile"
    
    def _clean_schema(self, schema: dict) -> dict:
        """Remove unsupported fields from schema for Groq compatibility."""
        if not isinstance(schema, dict):
            return schema
        
        # Fields that might cause issues
        unsupported_fields = {"default", "examples", "title"}
        
        cleaned = {}
        for key, value in schema.items():
            if key in unsupported_fields:
                continue
            elif key == "properties" and isinstance(value, dict):
                cleaned[key] = {
                    k: self._clean_schema(v) for k, v in value.items()
                }
            elif isinstance(value, dict):
                cleaned[key] = self._clean_schema(value)
            else:
                cleaned[key] = value
        
        return cleaned
    
    def _format_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Format tools for Groq API."""
        formatted = []
        for tool in tools:
            clean_params = self._clean_schema(tool.parameters)
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": clean_params
                }
            })
        return formatted
    
    def _format_messages(self, messages: list[Message], system_prompt: str | None = None) -> list[dict]:
        """Format messages for Groq API (OpenAI compatible)."""
        formatted = []
        
        # Add system prompt first
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            if msg.role == "system":
                continue  # Already handled
            
            # Handle tool results
            if msg.tool_results:
                for result in msg.tool_results:
                    formatted.append({
                        "role": "tool",
                        "tool_call_id": result["tool_use_id"],
                        "content": str(result["content"])
                    })
                continue
            
            message_dict = {"role": msg.role, "content": msg.content or ""}
            
            # Add tool calls if present
            if msg.tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["input"]) if isinstance(tc["input"], dict) else tc["input"]
                        }
                    }
                    for tc in msg.tool_calls
                ]
            
            formatted.append(message_dict)
        
        return formatted
    
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system_prompt: str | None = None
    ) -> LLMResponse:
        """Generate a response using Groq."""
        
        formatted_messages = self._format_messages(messages, system_prompt)
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        if tools:
            payload["tools"] = self._format_tools(tools)
            payload["tool_choice"] = "auto"
        
        max_retries = 5
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json=payload
                    )
                    
                    if response.status_code == 429:
                        error_detail = response.text
                        # Try to parse wait time from error message
                        wait_time = base_delay * (2 ** attempt)  # Default exponential backoff
                        
                        try:
                            error_json = response.json()
                            msg = error_json.get("error", {}).get("message", "")
                            # Look for "Please try again in Xs"
                            match = re.search(r"try again in (\d+\.?\d*)s", msg)
                            if match:
                                wait_time = float(match.group(1)) + 1.0  # Add buffer
                        except:
                            pass
                            
                        # Add jitter
                        wait_time += random.uniform(0.1, 0.5)
                        
                        console.print(f"[yellow]Groq Rate Limit hit. Retrying in {wait_time:.2f}s (Attempt {attempt+1}/{max_retries})[/yellow]")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    # Better error handling
                    if response.status_code != 200:
                        error_detail = response.text
                        try:
                            error_json = response.json()
                            error_detail = error_json.get("error", {}).get("message", error_detail)
                        except:
                            pass
                        raise Exception(f"Groq API error ({response.status_code}): {error_detail}")
                    
                    data = response.json()
                
                # Parse response
                choice = data["choices"][0]
                message = choice["message"]
                
                content = message.get("content", "") or ""
                tool_calls = []
                
                if message.get("tool_calls"):
                    for tc in message["tool_calls"]:
                        try:
                            args = json.loads(tc["function"]["arguments"])
                        except json.JSONDecodeError:
                            args = {}
                        tool_calls.append({
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": args
                        })
                
                return LLMResponse(
                    content=content,
                    tool_calls=tool_calls,
                    stop_reason=choice.get("finish_reason", "stop"),
                    usage=data.get("usage", {}),
                    raw_response=data
                )
                
            except Exception as e:
                # If it's the last attempt, re-raise
                if attempt == max_retries - 1:
                    raise e
                # If not a rate limit error (handled above) but some other transient error, could also retry
                # For now, we mainly care about 429s which are handled in the status check
                if "Groq API error (429)" in str(e):
                     # Loop will handle this via continue if logic was restructured, 
                     # but here we handled 429 inside the response check.
                     # This catch is for other network exceptions.
                     pass
                raise e
    
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system_prompt: str | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream a response using Groq."""
        
        formatted_messages = self._format_messages(messages, system_prompt)
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            continue


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required")
        
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=api_key)
    
    def _format_messages(self, messages: list[Message]) -> list[dict]:
        """Format messages for Claude API."""
        formatted = []
        for msg in messages:
            if msg.role == "system":
                continue  # System handled separately
            
            content = []
            
            # Add tool results if present
            if msg.tool_results:
                for result in msg.tool_results:
                    content.append({
                        "type": "tool_result",
                        "tool_use_id": result["tool_use_id"],
                        "content": result["content"]
                    })
            
            # Add text content
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            
            # Add tool calls if present
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "input": tool_call["input"]
                    })
            
            formatted.append({
                "role": msg.role,
                "content": content if len(content) > 1 else (content[0] if content else msg.content)
            })
        
        return formatted
    
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system_prompt: str | None = None
    ) -> LLMResponse:
        """Generate a response using Claude."""
        
        formatted_messages = self._format_messages(messages)
        
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": formatted_messages,
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        if tools:
            kwargs["tools"] = [t.to_claude_format() for t in tools]
        
        response = await self.async_client.messages.create(**kwargs)
        
        # Parse response
        content = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            },
            raw_response=response
        )
    
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system_prompt: str | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream a response using Claude."""
        
        formatted_messages = self._format_messages(messages)
        
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": formatted_messages,
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        if tools:
            kwargs["tools"] = [t.to_claude_format() for t in tools]
        
        async with self.async_client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider implementation."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable required")
        
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        self.model_instance = genai.GenerativeModel(self.model)
    
    def _format_messages(self, messages: list[Message]) -> list[dict]:
        """Format messages for Gemini API."""
        formatted = []
        for msg in messages:
            if msg.role == "system":
                continue
            
            role = "user" if msg.role == "user" else "model"
            
            parts = []
            
            # Add tool results
            if msg.tool_results:
                for result in msg.tool_results:
                    parts.append({
                        "function_response": {
                            "name": result["name"],
                            "response": {"result": result["content"]}
                        }
                    })
            
            # Add text
            if msg.content:
                parts.append({"text": msg.content})
            
            # Add tool calls
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    parts.append({
                        "function_call": {
                            "name": tool_call["name"],
                            "args": tool_call["input"]
                        }
                    })
            
            formatted.append({"role": role, "parts": parts})
        
        return formatted
    
    def _format_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Format tools for Gemini API."""
        functions = []
        for tool in tools:
            # Clean parameters - remove 'default' fields that Gemini doesn't support
            clean_params = self._clean_schema(tool.parameters)
            functions.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": clean_params
            })
        return [{"function_declarations": functions}]
    
    def _clean_schema(self, schema: dict) -> dict:
        """Remove unsupported fields from schema for Gemini compatibility."""
        if not isinstance(schema, dict):
            return schema
        
        # Fields that Gemini doesn't support
        unsupported_fields = {"default", "examples", "title"}
        
        cleaned = {}
        for key, value in schema.items():
            if key in unsupported_fields:
                continue
            elif key == "properties" and isinstance(value, dict):
                cleaned[key] = {
                    k: self._clean_schema(v) for k, v in value.items()
                }
            elif isinstance(value, dict):
                cleaned[key] = self._clean_schema(value)
            else:
                cleaned[key] = value
        
        return cleaned
    
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system_prompt: str | None = None
    ) -> LLMResponse:
        """Generate a response using Gemini."""
        
        # Configure model with system instruction
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        
        model = self.genai.GenerativeModel(
            self.model,
            generation_config=generation_config,
            system_instruction=system_prompt if system_prompt else None,
            tools=self._format_tools(tools) if tools else None
        )
        
        formatted_messages = self._format_messages(messages)
        
        # Start chat and send messages
        chat = model.start_chat(history=formatted_messages[:-1] if len(formatted_messages) > 1 else [])
        
        last_message = formatted_messages[-1] if formatted_messages else {"parts": [{"text": ""}]}
        response = await chat.send_message_async(last_message["parts"])
        
        # Parse response
        content = ""
        tool_calls = []
        
        for part in response.parts:
            if hasattr(part, "text"):
                content += part.text
            elif hasattr(part, "function_call"):
                fc = part.function_call
                tool_calls.append({
                    "id": f"gemini_{fc.name}_{len(tool_calls)}",
                    "name": fc.name,
                    "input": dict(fc.args)
                })
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason="end_turn" if not tool_calls else "tool_use",
            usage={},  # Gemini doesn't provide detailed usage
            raw_response=response
        )
    
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system_prompt: str | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream a response using Gemini."""
        
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        
        model = self.genai.GenerativeModel(
            self.model,
            generation_config=generation_config,
            system_instruction=system_prompt if system_prompt else None
        )
        
        formatted_messages = self._format_messages(messages)
        chat = model.start_chat(history=formatted_messages[:-1] if len(formatted_messages) > 1 else [])
        
        last_message = formatted_messages[-1] if formatted_messages else {"parts": [{"text": ""}]}
        response = await chat.send_message_async(last_message["parts"], stream=True)
        
        async for chunk in response:
            if chunk.text:
                yield chunk.text


def get_provider(provider_type: str, config: dict) -> BaseLLMProvider:
    """Factory function to get the appropriate provider."""
    providers = {
        "claude": ClaudeProvider,
        "gemini": GeminiProvider,
        "groq": GroqProvider,
    }
    
    if provider_type not in providers:
        raise ValueError(f"Unknown provider: {provider_type}. Available: {list(providers.keys())}")
    
    return providers[provider_type](config)
