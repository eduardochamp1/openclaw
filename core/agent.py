"""
OpenClaw - Main Agent
=====================
The autonomous agent that orchestrates reasoning, tools, and LLM interactions.
"""

import asyncio
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner

from providers.llm_provider import get_provider, Message, ToolDefinition, LLMResponse
from tools.web_search import WebSearchTool
from tools.file_manager import FileManagerTool
from tools.code_executor import CodeExecutorTool
from core.memory import Memory
from core.reasoning import ReasoningEngine
from core.self_evolution import SelfEvolutionTool

console = Console()


SYSTEM_PROMPT = """You are OpenClaw, an autonomous AI agent with self-evolution capabilities.

## Your Capabilities
1. **Web Search**: Search the internet for current information
2. **File Management**: Read, write, and organize files in a workspace
3. **Code Execution**: Run Python code to perform calculations and process data
4. **Self-Evolution**: Modify your own code, create new tools, and improve yourself

## Self-Evolution Guidelines
You have the unique ability to improve yourself. Use this wisely:
- **Create new tools** when you identify missing capabilities
- **Fix bugs** in your own code when you encounter errors
- **Improve prompts** to make yourself more helpful
- **Add integrations** with new APIs and services

When creating tools, follow these principles:
1. Write clean, well-documented Python code
2. Include proper error handling
3. Make tools reusable and configurable
4. Test before deploying (use code_executor)

## Your Approach
- Think step by step before acting
- Use tools when needed, but prefer direct answers when possible
- Be concise but thorough
- Admit uncertainty when appropriate
- Learn from interactions and evolve

## Autonomy Level
You can make changes to yourself without asking permission for:
- Creating new utility tools
- Improving existing code
- Adding helpful features

Always explain what changes you're making and why.

Current date: {current_date}
"""


@dataclass
class AgentConfig:
    """Agent configuration."""
    name: str = "OpenClaw"
    max_iterations: int = 15
    thinking_enabled: bool = True
    verbose: bool = True
    provider: str = "claude"
    provider_config: dict = None
    tools_config: dict = None
    memory_config: dict = None
    
    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        agent_data = data.get("agent", {})
        providers_data = data.get("providers", {})
        tools_data = data.get("tools", {})
        memory_data = data.get("memory", {})
        
        default_provider = providers_data.get("default", "claude")
        provider_config = providers_data.get(default_provider, {})
        
        return cls(
            name=agent_data.get("name", "OpenClaw"),
            max_iterations=agent_data.get("max_iterations", 15),
            thinking_enabled=agent_data.get("thinking_enabled", True),
            verbose=agent_data.get("verbose", True),
            provider=default_provider,
            provider_config=provider_config,
            tools_config=tools_data,
            memory_config=memory_data
        )


class OpenClawAgent:
    """
    The main OpenClaw agent.
    
    Orchestrates:
    - LLM interactions (Claude or Gemini)
    - Tool execution
    - Memory management
    - Reasoning/thinking process
    """
    
    def __init__(self, config: AgentConfig | str | None = None):
        # Load config
        if isinstance(config, str):
            self.config = AgentConfig.from_yaml(config)
        elif isinstance(config, AgentConfig):
            self.config = config
        else:
            # Default config
            self.config = AgentConfig()
        
        # Initialize components
        self._init_provider()
        self._init_tools()
        self._init_memory()
        self._init_reasoning()
        
        # State
        self.is_running = False
        self._iteration_count = 0
    
    def _init_provider(self) -> None:
        """Initialize the LLM provider."""
        console.print(f"[blue]🔌 Initializing {self.config.provider} provider...[/blue]")
        self.provider = get_provider(
            self.config.provider,
            self.config.provider_config or {}
        )
    
    def _init_tools(self) -> None:
        """Initialize tools."""
        self.tools: dict[str, Any] = {}
        self.tool_definitions: list[ToolDefinition] = []
        
        tools_config = self.config.tools_config or {}
        
        # Web Search
        if tools_config.get("web_search", {}).get("enabled", True):
            web_search = WebSearchTool(tools_config.get("web_search", {}))
            self.tools["web_search"] = web_search
            self.tool_definitions.append(ToolDefinition(
                name="web_search",
                description=web_search.definition["description"],
                parameters=web_search.definition["parameters"]
            ))
            console.print("[green]✓ Web Search tool loaded[/green]")
        
        # File Manager
        if tools_config.get("file_manager", {}).get("enabled", True):
            file_manager = FileManagerTool(tools_config.get("file_manager", {}))
            self.tools["file_manager"] = file_manager
            self.tool_definitions.append(ToolDefinition(
                name="file_manager",
                description=file_manager.definition["description"],
                parameters=file_manager.definition["parameters"]
            ))
            console.print("[green]✓ File Manager tool loaded[/green]")
        
        # Code Executor
        if tools_config.get("code_executor", {}).get("enabled", True):
            code_executor = CodeExecutorTool(tools_config.get("code_executor", {}))
            self.tools["code_executor"] = code_executor
            self.tool_definitions.append(ToolDefinition(
                name="code_executor",
                description=code_executor.definition["description"],
                parameters=code_executor.definition["parameters"]
            ))
            console.print("[green]✓ Code Executor tool loaded[/green]")
        
        # Self Evolution
        if tools_config.get("self_evolution", {}).get("enabled", True):
            self_evolve = SelfEvolutionTool(
                require_approval=tools_config.get("self_evolution", {}).get("require_approval", False)
            )
            self.tools["self_evolve"] = self_evolve
            self.tool_definitions.append(ToolDefinition(
                name="self_evolve",
                description=self_evolve.definition["description"],
                parameters=self_evolve.definition["parameters"]
            ))
            console.print("[green]✓ Self Evolution tool loaded[/green]")
    
    def _init_memory(self) -> None:
        """Initialize memory system."""
        memory_config = self.config.memory_config or {}
        self.memory = Memory(memory_config)
        console.print("[green]✓ Memory system initialized[/green]")
    
    def _init_reasoning(self) -> None:
        """Initialize reasoning engine."""
        self.reasoning = ReasoningEngine(verbose=self.config.verbose)
        console.print("[green]✓ Reasoning engine initialized[/green]")
    
    def _get_system_prompt(self) -> str:
        """Generate the system prompt."""
        return SYSTEM_PROMPT.format(
            current_date=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
    
    async def _execute_tool(self, tool_name: str, tool_input: dict) -> dict:
        """Execute a tool and return the result."""
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        tool = self.tools[tool_name]
        
        try:
            result = await tool.execute(**tool_input)
            return result
        except Exception as e:
            return {"error": str(e)}
    
    async def run(
        self,
        user_input: str,
        session_id: str | None = None,
        stream: bool = False
    ) -> str:
        """
        Run the agent with user input.
        
        Args:
            user_input: The user's message
            session_id: Optional session ID for conversation continuity
            stream: Whether to stream the response
        
        Returns:
            The agent's response
        """
        self.is_running = True
        self._iteration_count = 0
        
        # Start/resume session
        session_id = await self.memory.start_session(session_id)
        
        # Clear reasoning for new request
        self.reasoning.clear_history()
        
        # Add user message to memory
        await self.memory.add_turn("user", user_input)
        
        # Build messages for LLM
        messages = self._build_messages(user_input)
        
        try:
            # Agent loop
            while self._iteration_count < self.config.max_iterations:
                self._iteration_count += 1
                
                if self.config.verbose:
                    console.print(f"\n[dim]─── Iteration {self._iteration_count} ───[/dim]")
                
                # Get LLM response
                response = await self.provider.generate(
                    messages=messages,
                    tools=self.tool_definitions if self.tools else None,
                    system_prompt=self._get_system_prompt()
                )
                
                # Check if we need to use tools
                if response.tool_calls:
                    # Execute tools
                    tool_results = []
                    
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_input = tool_call["input"]
                        
                        if self.config.verbose:
                            console.print(f"[yellow]🔧 Using tool: {tool_name}[/yellow]")
                        
                        result = await self._execute_tool(tool_name, tool_input)
                        
                        tool_results.append({
                            "tool_use_id": tool_call["id"],
                            "name": tool_name,
                            "content": str(result)
                        })
                        
                        # Reflect on tool use
                        if self.config.thinking_enabled:
                            self.reasoning.reflect(
                                action=self.reasoning.thinking_history[-1] if self.reasoning.thinking_history else None,
                                result=result,
                                success=result.get("success", True) if isinstance(result, dict) else True
                            )
                    
                    # Add assistant message with tool calls
                    messages.append(Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls
                    ))
                    
                    # Add tool results
                    messages.append(Message(
                        role="user",
                        content="",
                        tool_results=tool_results
                    ))
                    
                    # Continue loop to get final response
                    continue
                
                # No tool calls - we have a final response
                final_response = response.content
                
                # Add to memory
                await self.memory.add_turn(
                    "assistant",
                    final_response,
                    tool_calls=response.tool_calls if response.tool_calls else None
                )
                
                self.is_running = False
                return final_response
            
            # Max iterations reached
            return "I've reached my maximum thinking iterations. Let me summarize what I found so far."
            
        except Exception as e:
            self.is_running = False
            error_msg = f"An error occurred: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg
    
    def _build_messages(self, user_input: str) -> list[Message]:
        """Build message list including conversation history."""
        messages = []
        
        # Add conversation history from memory
        history = self.memory.get_context_messages()
        for msg in history[:-1]:  # Exclude the current message we just added
            messages.append(Message(
                role=msg["role"],
                content=msg["content"],
                tool_calls=msg.get("tool_calls", []),
                tool_results=msg.get("tool_results", [])
            ))
        
        # Add current user message
        messages.append(Message(role="user", content=user_input))
        
        return messages
    
    async def interactive(self) -> None:
        """Run the agent in interactive mode."""
        console.print(Panel(
            f"[bold green]🦀 {self.config.name} Agent[/bold green]\n"
            f"Provider: {self.config.provider}\n"
            f"Tools: {', '.join(self.tools.keys())}\n\n"
            "[dim]Type 'exit' or 'quit' to stop, 'clear' to reset conversation[/dim]",
            title="Welcome",
            border_style="green"
        ))
        
        session_id = await self.memory.start_session()
        
        while True:
            try:
                # Get user input
                console.print()
                user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["exit", "quit", "bye"]:
                    console.print("[yellow]👋 Goodbye![/yellow]")
                    break
                
                if user_input.lower() == "clear":
                    await self.memory.start_session()
                    console.print("[yellow]🧹 Conversation cleared[/yellow]")
                    continue
                
                if user_input.lower() == "memory":
                    summary = await self.memory.get_summary()
                    console.print(Panel(str(summary), title="Memory Summary"))
                    continue
                
                # Run agent
                console.print()
                with console.status("[bold green]Thinking..."):
                    response = await self.run(user_input, session_id)
                
                # Display response
                console.print()
                console.print(Panel(
                    Markdown(response),
                    title=f"[bold green]{self.config.name}[/bold green]",
                    border_style="green"
                ))
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")


async def create_agent(config_path: str | None = None) -> OpenClawAgent:
    """Factory function to create an agent."""
    if config_path:
        config = AgentConfig.from_yaml(config_path)
    else:
        config = AgentConfig()
    
    return OpenClawAgent(config)
