"""
OpenClaw - Autonomous AI Agent
==============================

A multi-provider AI agent with web search and file management capabilities.

Usage:
    from openclaw import OpenClawAgent, AgentConfig
    
    agent = OpenClawAgent()
    response = await agent.run("Search for latest AI news")
"""

from .core import (
    OpenClawAgent,
    AgentConfig,
    create_agent,
    Memory,
    ReasoningEngine,
)

from .providers import (
    get_provider,
    ClaudeProvider,
    GeminiProvider,
    Message,
    ToolDefinition,
)

from .tools import (
    WebSearchTool,
    FileManagerTool,
)

__version__ = "1.0.0"
__author__ = "OpenClaw Team"

__all__ = [
    # Core
    "OpenClawAgent",
    "AgentConfig",
    "create_agent",
    "Memory",
    "ReasoningEngine",
    # Providers
    "get_provider",
    "ClaudeProvider",
    "GeminiProvider",
    "Message",
    "ToolDefinition",
    # Tools
    "WebSearchTool",
    "FileManagerTool",
]
