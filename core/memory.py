"""
OpenClaw - Memory System
========================
Persistent memory for conversation history and learned information.
"""

import json
import asyncio
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any
from abc import ABC, abstractmethod

import aiofiles
from rich.console import Console

console = Console()


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    timestamp: str
    type: str  # "message" | "fact" | "task" | "reflection"
    content: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(**data)


@dataclass
class ConversationTurn:
    """A single conversation turn."""
    role: str
    content: str
    timestamp: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTurn":
        return cls(**data)


class BaseMemoryStore(ABC):
    """Abstract base class for memory storage backends."""
    
    @abstractmethod
    async def save(self, key: str, data: Any) -> None:
        """Save data to memory."""
        pass
    
    @abstractmethod
    async def load(self, key: str) -> Any | None:
        """Load data from memory."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete data from memory."""
        pass
    
    @abstractmethod
    async def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys with optional prefix filter."""
        pass


class JSONMemoryStore(BaseMemoryStore):
    """JSON file-based memory storage."""
    
    def __init__(self, persist_path: str):
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {}
        self._loaded = False
    
    async def _ensure_loaded(self) -> None:
        """Ensure data is loaded from disk."""
        if self._loaded:
            return
        
        if self.persist_path.exists():
            async with aiofiles.open(self.persist_path, "r") as f:
                content = await f.read()
                self._data = json.loads(content) if content else {}
        
        self._loaded = True
    
    async def _persist(self) -> None:
        """Persist data to disk."""
        async with aiofiles.open(self.persist_path, "w") as f:
            await f.write(json.dumps(self._data, indent=2, default=str))
    
    async def save(self, key: str, data: Any) -> None:
        await self._ensure_loaded()
        self._data[key] = data
        await self._persist()
    
    async def load(self, key: str) -> Any | None:
        await self._ensure_loaded()
        return self._data.get(key)
    
    async def delete(self, key: str) -> None:
        await self._ensure_loaded()
        if key in self._data:
            del self._data[key]
            await self._persist()
    
    async def list_keys(self, prefix: str = "") -> list[str]:
        await self._ensure_loaded()
        if prefix:
            return [k for k in self._data.keys() if k.startswith(prefix)]
        return list(self._data.keys())


class Memory:
    """
    Main memory class for the agent.
    
    Manages:
    - Conversation history
    - Learned facts
    - Task history
    - Self-reflections
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.max_context_messages = config.get("max_context_messages", 20)
        
        # Initialize storage backend
        store_type = config.get("type", "json")
        if store_type == "json":
            persist_path = config.get("persist_path", "./data/memory.json")
            self.store = JSONMemoryStore(persist_path)
        else:
            raise ValueError(f"Unknown memory store type: {store_type}")
        
        # In-memory conversation buffer
        self._conversation: list[ConversationTurn] = []
        self._session_id: str | None = None
    
    async def start_session(self, session_id: str | None = None) -> str:
        """Start a new conversation session."""
        if session_id:
            self._session_id = session_id
            # Try to load existing session
            existing = await self.store.load(f"session:{session_id}")
            if existing:
                self._conversation = [
                    ConversationTurn.from_dict(t) for t in existing.get("turns", [])
                ]
                console.print(f"[blue]📚 Loaded session: {session_id} ({len(self._conversation)} turns)[/blue]")
                return session_id
        else:
            self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self._conversation = []
        console.print(f"[blue]🆕 New session: {self._session_id}[/blue]")
        return self._session_id
    
    async def add_turn(
        self,
        role: str,
        content: str,
        tool_calls: list[dict] | None = None,
        tool_results: list[dict] | None = None
    ) -> None:
        """Add a conversation turn."""
        if not self.enabled:
            return
        
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            tool_calls=tool_calls or [],
            tool_results=tool_results or []
        )
        
        self._conversation.append(turn)
        
        # Persist session
        if self._session_id:
            await self.store.save(
                f"session:{self._session_id}",
                {
                    "session_id": self._session_id,
                    "turns": [t.to_dict() for t in self._conversation],
                    "updated": datetime.now().isoformat()
                }
            )
    
    def get_context_messages(self) -> list[dict]:
        """Get recent conversation history for context."""
        if not self.enabled:
            return []
        
        # Get last N messages
        recent = self._conversation[-self.max_context_messages:]
        
        return [
            {
                "role": t.role,
                "content": t.content,
                "tool_calls": t.tool_calls,
                "tool_results": t.tool_results
            }
            for t in recent
        ]
    
    async def add_fact(self, fact: str, source: str | None = None, confidence: float = 1.0) -> None:
        """Store a learned fact."""
        if not self.enabled:
            return
        
        facts = await self.store.load("facts") or []
        
        fact_entry = {
            "id": f"fact_{len(facts)}",
            "fact": fact,
            "source": source,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        facts.append(fact_entry)
        await self.store.save("facts", facts)
        console.print(f"[green]💡 Learned: {fact[:50]}...[/green]")
    
    async def get_facts(self, query: str | None = None) -> list[dict]:
        """Retrieve stored facts, optionally filtered by query."""
        facts = await self.store.load("facts") or []
        
        if query:
            query_lower = query.lower()
            facts = [f for f in facts if query_lower in f["fact"].lower()]
        
        return facts
    
    async def add_reflection(self, reflection: str, context: str | None = None) -> None:
        """Store a self-reflection."""
        if not self.enabled:
            return
        
        reflections = await self.store.load("reflections") or []
        
        reflection_entry = {
            "id": f"reflection_{len(reflections)}",
            "reflection": reflection,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        reflections.append(reflection_entry)
        await self.store.save("reflections", reflections)
        console.print(f"[cyan]🤔 Reflected: {reflection[:50]}...[/cyan]")
    
    async def get_summary(self) -> dict:
        """Get a summary of memory contents."""
        facts = await self.store.load("facts") or []
        reflections = await self.store.load("reflections") or []
        sessions = await self.store.list_keys("session:")
        
        return {
            "current_session": self._session_id,
            "current_turns": len(self._conversation),
            "total_sessions": len(sessions),
            "total_facts": len(facts),
            "total_reflections": len(reflections)
        }
    
    async def clear(self, keep_facts: bool = True) -> None:
        """Clear memory."""
        self._conversation = []
        
        if not keep_facts:
            await self.store.delete("facts")
            await self.store.delete("reflections")
        
        # Clear sessions
        sessions = await self.store.list_keys("session:")
        for session_key in sessions:
            await self.store.delete(session_key)
        
        console.print("[yellow]🧹 Memory cleared[/yellow]")
