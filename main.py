#!/usr/bin/env python3
"""
OpenClaw - Autonomous AI Agent
==============================

Usage:
    python main.py                    # Interactive mode
    python main.py --config path.yaml # Custom config
    python main.py "your question"    # Single query mode

Environment Variables:
    ANTHROPIC_API_KEY  - For Claude provider
    GOOGLE_API_KEY     - For Gemini provider
    SERPER_API_KEY     - For Serper search (optional)
    TAVILY_API_KEY     - For Tavily search (optional)
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import typer
from rich.console import Console
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

console = Console()
app = typer.Typer(
    name="openclaw",
    help="OpenClaw - Autonomous AI Agent",
    add_completion=False
)


@app.command()
def main(
    query: str = typer.Argument(None, help="Single query to process"),
    config: str = typer.Option(
        None, "--config", "-c",
        help="Path to configuration YAML file"
    ),
    provider: str = typer.Option(
        None, "--provider", "-p",
        help="LLM provider to use (claude/gemini)"
    ),
    verbose: bool = typer.Option(
        True, "--verbose/--quiet", "-v/-q",
        help="Enable/disable verbose output"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i",
        help="Force interactive mode even with query"
    )
):
    """
    OpenClaw - Autonomous AI Agent
    
    Run without arguments for interactive mode, or pass a query for single execution.
    """
    # Import here to avoid circular imports
    from core.agent import OpenClawAgent, AgentConfig
    
    async def run():
        # Build config
        if config:
            agent_config = AgentConfig.from_yaml(config)
        else:
            # Use default config from project
            default_config = PROJECT_ROOT / "config" / "settings.yaml"
            if default_config.exists():
                agent_config = AgentConfig.from_yaml(str(default_config))
            else:
                agent_config = AgentConfig()
        
        # Override from CLI
        if provider:
            agent_config.provider = provider
        agent_config.verbose = verbose
        
        # Create agent
        try:
            agent = OpenClawAgent(agent_config)
        except ValueError as e:
            console.print(f"[red]Configuration error: {e}[/red]")
            console.print("\n[yellow]Make sure you have set the required API keys:[/yellow]")
            console.print("  export ANTHROPIC_API_KEY=your_key  # For Claude")
            console.print("  export GOOGLE_API_KEY=your_key     # For Gemini")
            raise typer.Exit(1)
        
        if query and not interactive:
            # Single query mode
            console.print(f"\n[cyan]Query:[/cyan] {query}\n")
            response = await agent.run(query)
            console.print(f"\n[green]Response:[/green]\n{response}")
        else:
            # Interactive mode
            await agent.interactive()
    
    # Run async
    asyncio.run(run())


@app.command()
def version():
    """Show version information."""
    console.print("[bold green]OpenClaw Agent[/bold green] v1.0.0")
    console.print("An autonomous AI agent with web search and file management")


@app.command()
def check():
    """Check environment and configuration."""
    console.print("[bold]Environment Check[/bold]\n")
    
    # Check API keys
    checks = [
        ("GROQ_API_KEY", "Groq provider (FREE)"),
        ("ANTHROPIC_API_KEY", "Claude provider (paid)"),
        ("GOOGLE_API_KEY", "Gemini provider"),
        ("SERPER_API_KEY", "Serper search (optional)"),
        ("TAVILY_API_KEY", "Tavily search (optional)"),
    ]
    
    for key, desc in checks:
        value = os.getenv(key)
        if value:
            # Mask the key for display
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "****"
            console.print(f"  [green]✓[/green] {key}: {masked} ({desc})")
        else:
            color = "yellow" if "optional" in desc.lower() or "paid" in desc.lower() else "red"
            symbol = "○" if "optional" in desc.lower() or "paid" in desc.lower() else "✗"
            console.print(f"  [{color}]{symbol}[/{color}] {key}: Not set ({desc})")
    
    # Check config file
    console.print("\n[bold]Configuration[/bold]\n")
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    if config_path.exists():
        console.print(f"  [green]✓[/green] Config file found: {config_path}")
    else:
        console.print(f"  [yellow]○[/yellow] No config file at {config_path}")
    
    # Check workspace
    workspace = PROJECT_ROOT / "workspace"
    if workspace.exists():
        console.print(f"  [green]✓[/green] Workspace exists: {workspace}")
    else:
        console.print(f"  [yellow]○[/yellow] Workspace will be created: {workspace}")
    
    # Check dependencies
    console.print("\n[bold]Dependencies[/bold]\n")
    deps = ["anthropic", "google.generativeai", "duckduckgo_search", "rich", "httpx"]
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
            console.print(f"  [green]✓[/green] {dep}")
        except ImportError:
            console.print(f"  [red]✗[/red] {dep} - run: pip install -r requirements.txt")


if __name__ == "__main__":
    # Se não houver argumentos ou o primeiro argumento não for um comando conhecido
    if len(sys.argv) == 1:
        # Modo interativo padrão
        sys.argv.append("main")
    elif sys.argv[1] not in ["main", "check", "version", "--help", "-h"]:
        # Se passou uma query direta, adiciona "main" antes
        sys.argv.insert(1, "main")
    
    app()
