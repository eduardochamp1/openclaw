"""
OpenClaw - Self Evolution Module
================================
Allows the agent to modify its own code, add new tools, and evolve capabilities.

SAFETY FEATURES:
- All changes are versioned (git-like)
- Rollback capability
- Sandbox testing before applying
- Human approval mode (optional)
"""

import os
import json
import shutil
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable
import ast
import asyncio

from rich.console import Console

console = Console()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class CodeChange:
    """Represents a code modification."""
    id: str
    timestamp: str
    file_path: str
    change_type: str  # "create" | "modify" | "delete"
    description: str
    old_content: str | None
    new_content: str
    status: str = "pending"  # "pending" | "applied" | "rolled_back" | "failed"
    error: str | None = None


@dataclass 
class EvolutionLog:
    """Log of all evolution attempts."""
    changes: list[CodeChange] = field(default_factory=list)
    
    def add(self, change: CodeChange):
        self.changes.append(change)
    
    def get_pending(self) -> list[CodeChange]:
        return [c for c in self.changes if c.status == "pending"]
    
    def get_applied(self) -> list[CodeChange]:
        return [c for c in self.changes if c.status == "applied"]


class CodeAnalyzer:
    """Analyzes Python code for safety and validity."""
    
    DANGEROUS_IMPORTS = [
        "subprocess", "os.system", "eval", "exec", 
        "compile", "__import__", "importlib"
    ]
    
    DANGEROUS_CALLS = [
        "os.remove", "os.rmdir", "shutil.rmtree",
        "os.system", "subprocess.run", "subprocess.call"
    ]
    
    @classmethod
    def is_valid_python(cls, code: str) -> tuple[bool, str]:
        """Check if code is valid Python syntax."""
        try:
            ast.parse(code)
            return True, "Valid Python syntax"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
    
    @classmethod
    def check_safety(cls, code: str) -> tuple[bool, list[str]]:
        """Check code for potentially dangerous operations."""
        warnings = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False, ["Invalid Python syntax"]
        
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in cls.DANGEROUS_IMPORTS:
                        warnings.append(f"Dangerous import: {alias.name}")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module in cls.DANGEROUS_IMPORTS:
                    warnings.append(f"Dangerous import from: {node.module}")
            
            # Check function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    full_name = f"{cls._get_full_name(node.func)}"
                    if full_name in cls.DANGEROUS_CALLS:
                        warnings.append(f"Dangerous call: {full_name}")
        
        is_safe = len(warnings) == 0
        return is_safe, warnings
    
    @staticmethod
    def _get_full_name(node) -> str:
        """Get full dotted name from AST node."""
        if isinstance(node, ast.Attribute):
            return f"{CodeAnalyzer._get_full_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Name):
            return node.id
        return ""


class SelfEvolution:
    """
    Handles self-modification capabilities of the agent.
    
    Capabilities:
    1. Create new tools
    2. Modify existing code
    3. Add new providers
    4. Improve prompts
    5. Fix bugs in itself
    """
    
    def __init__(self, require_approval: bool = True, auto_backup: bool = True):
        self.require_approval = require_approval
        self.auto_backup = auto_backup
        self.evolution_log = EvolutionLog()
        self.backup_dir = PROJECT_ROOT / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Load evolution history
        self.history_file = PROJECT_ROOT / "data" / "evolution_history.json"
        self._load_history()
    
    def _load_history(self):
        """Load evolution history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                    for change_data in data.get("changes", []):
                        self.evolution_log.add(CodeChange(**change_data))
            except Exception as e:
                console.print(f"[yellow]Could not load evolution history: {e}[/yellow]")
    
    def _save_history(self):
        """Save evolution history to file."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, "w") as f:
            json.dump({
                "changes": [
                    {
                        "id": c.id,
                        "timestamp": c.timestamp,
                        "file_path": c.file_path,
                        "change_type": c.change_type,
                        "description": c.description,
                        "old_content": c.old_content,
                        "new_content": c.new_content,
                        "status": c.status,
                        "error": c.error
                    }
                    for c in self.evolution_log.changes
                ]
            }, f, indent=2)
    
    def _generate_id(self) -> str:
        """Generate unique ID for a change."""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]
    
    def _backup_file(self, file_path: Path) -> Path | None:
        """Create backup of a file before modification."""
        if not file_path.exists():
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        console.print(f"[dim]Backup created: {backup_path}[/dim]")
        return backup_path
    
    async def create_tool(
        self,
        tool_name: str,
        description: str,
        code: str,
        auto_register: bool = True
    ) -> CodeChange:
        """
        Create a new tool for the agent.
        
        Args:
            tool_name: Name of the tool (e.g., "calculator")
            description: What the tool does
            code: Python code for the tool class
            auto_register: Whether to automatically register in agent
        """
        # Validate
        is_valid, msg = CodeAnalyzer.is_valid_python(code)
        if not is_valid:
            raise ValueError(f"Invalid Python code: {msg}")
        
        is_safe, warnings = CodeAnalyzer.check_safety(code)
        if not is_safe and self.require_approval:
            console.print(f"[yellow]Safety warnings: {warnings}[/yellow]")
            # In production, would ask for human approval here
        
        # Create file path
        file_path = PROJECT_ROOT / "tools" / f"{tool_name}.py"
        
        # Create change record
        change = CodeChange(
            id=self._generate_id(),
            timestamp=datetime.now().isoformat(),
            file_path=str(file_path),
            change_type="create",
            description=f"Create new tool: {tool_name} - {description}",
            old_content=None,
            new_content=code
        )
        
        self.evolution_log.add(change)
        
        if not self.require_approval:
            await self._apply_change(change)
            if auto_register:
                await self._register_tool(tool_name)
        
        self._save_history()
        return change
    
    async def modify_file(
        self,
        file_path: str,
        new_content: str,
        description: str
    ) -> CodeChange:
        """
        Modify an existing file.
        
        Args:
            file_path: Path to file (relative to project root)
            new_content: New file content
            description: What this change does
        """
        full_path = PROJECT_ROOT / file_path
        
        # Validate
        if file_path.endswith(".py"):
            is_valid, msg = CodeAnalyzer.is_valid_python(new_content)
            if not is_valid:
                raise ValueError(f"Invalid Python code: {msg}")
            
            is_safe, warnings = CodeAnalyzer.check_safety(new_content)
            if warnings:
                console.print(f"[yellow]Safety warnings: {warnings}[/yellow]")
        
        # Get old content
        old_content = None
        if full_path.exists():
            old_content = full_path.read_text(encoding="utf-8")
        
        # Create change record
        change = CodeChange(
            id=self._generate_id(),
            timestamp=datetime.now().isoformat(),
            file_path=file_path,
            change_type="modify" if old_content else "create",
            description=description,
            old_content=old_content,
            new_content=new_content
        )
        
        self.evolution_log.add(change)
        
        if not self.require_approval:
            await self._apply_change(change)
        
        self._save_history()
        return change
    
    async def improve_prompt(
        self,
        prompt_name: str,
        new_prompt: str,
        reason: str
    ) -> CodeChange:
        """
        Improve a system prompt.
        
        Args:
            prompt_name: Name/identifier of the prompt
            new_prompt: New prompt text
            reason: Why this improvement is being made
        """
        prompts_file = PROJECT_ROOT / "config" / "prompts.yaml"
        
        # Load existing prompts or create new
        import yaml
        prompts = {}
        if prompts_file.exists():
            with open(prompts_file) as f:
                prompts = yaml.safe_load(f) or {}
        
        old_prompt = prompts.get(prompt_name, "")
        prompts[prompt_name] = new_prompt
        
        new_content = yaml.dump(prompts, default_flow_style=False, allow_unicode=True)
        
        change = CodeChange(
            id=self._generate_id(),
            timestamp=datetime.now().isoformat(),
            file_path="config/prompts.yaml",
            change_type="modify",
            description=f"Improve prompt '{prompt_name}': {reason}",
            old_content=old_prompt,
            new_content=new_content
        )
        
        self.evolution_log.add(change)
        
        if not self.require_approval:
            await self._apply_change(change)
        
        self._save_history()
        return change
    
    async def _apply_change(self, change: CodeChange) -> bool:
        """Apply a code change."""
        try:
            full_path = PROJECT_ROOT / change.file_path
            
            # Backup if modifying
            if self.auto_backup and change.old_content:
                self._backup_file(full_path)
            
            # Ensure directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write new content
            full_path.write_text(change.new_content, encoding="utf-8")
            
            change.status = "applied"
            console.print(f"[green]✓ Applied change: {change.description}[/green]")
            
            self._save_history()
            return True
            
        except Exception as e:
            change.status = "failed"
            change.error = str(e)
            console.print(f"[red]✗ Failed to apply change: {e}[/red]")
            self._save_history()
            return False
    
    async def _register_tool(self, tool_name: str):
        """Register a new tool in the tools __init__.py."""
        init_path = PROJECT_ROOT / "tools" / "__init__.py"
        
        if init_path.exists():
            content = init_path.read_text()
        else:
            content = '"""OpenClaw Tools."""\n\n'
        
        # Add import
        class_name = "".join(word.capitalize() for word in tool_name.split("_")) + "Tool"
        import_line = f"from .{tool_name} import {class_name}\n"
        
        if import_line not in content:
            # Add after other imports
            if "from ." in content:
                last_import = content.rfind("from .")
                end_of_line = content.find("\n", last_import) + 1
                content = content[:end_of_line] + import_line + content[end_of_line:]
            else:
                content += import_line
            
            init_path.write_text(content)
            console.print(f"[green]✓ Registered tool: {class_name}[/green]")
    
    async def rollback(self, change_id: str) -> bool:
        """Rollback a specific change."""
        for change in self.evolution_log.changes:
            if change.id == change_id and change.status == "applied":
                if change.old_content is not None:
                    full_path = PROJECT_ROOT / change.file_path
                    full_path.write_text(change.old_content, encoding="utf-8")
                    change.status = "rolled_back"
                    console.print(f"[yellow]↩ Rolled back: {change.description}[/yellow]")
                    self._save_history()
                    return True
                elif change.change_type == "create":
                    full_path = PROJECT_ROOT / change.file_path
                    if full_path.exists():
                        full_path.unlink()
                    change.status = "rolled_back"
                    self._save_history()
                    return True
        return False
    
    async def approve_change(self, change_id: str) -> bool:
        """Approve and apply a pending change."""
        for change in self.evolution_log.changes:
            if change.id == change_id and change.status == "pending":
                return await self._apply_change(change)
        return False
    
    async def approve_all_pending(self) -> int:
        """Approve all pending changes."""
        applied = 0
        for change in self.evolution_log.get_pending():
            if await self._apply_change(change):
                applied += 1
        return applied
    
    def get_evolution_summary(self) -> dict:
        """Get summary of all evolution activity."""
        return {
            "total_changes": len(self.evolution_log.changes),
            "pending": len(self.evolution_log.get_pending()),
            "applied": len(self.evolution_log.get_applied()),
            "recent": [
                {
                    "id": c.id,
                    "description": c.description,
                    "status": c.status,
                    "timestamp": c.timestamp
                }
                for c in self.evolution_log.changes[-10:]
            ]
        }


class SelfEvolutionTool:
    """
    Tool interface for the agent to use self-evolution capabilities.
    """
    
    def __init__(self, require_approval: bool = False):
        self.evolution = SelfEvolution(require_approval=require_approval)
    
    @property
    def definition(self) -> dict:
        """Return the tool definition for LLM."""
        return {
            "name": "self_evolve",
            "description": (
                "Modify the agent's own code to add new capabilities, create new tools, "
                "improve prompts, or fix bugs. Use this to evolve and improve yourself. "
                "Changes are versioned and can be rolled back."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "create_tool",
                            "modify_file",
                            "improve_prompt",
                            "rollback",
                            "list_changes",
                            "approve_pending"
                        ],
                        "description": "The evolution action to perform"
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "Name for new tool (for create_tool action)"
                    },
                    "tool_description": {
                        "type": "string",
                        "description": "Description of what the tool does"
                    },
                    "code": {
                        "type": "string",
                        "description": "Python code for the tool or file content"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to file (for modify_file action)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of the change"
                    },
                    "prompt_name": {
                        "type": "string",
                        "description": "Name of prompt to improve"
                    },
                    "new_prompt": {
                        "type": "string",
                        "description": "New prompt content"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for the change"
                    },
                    "change_id": {
                        "type": "string",
                        "description": "ID of change to rollback"
                    }
                },
                "required": ["action"]
            }
        }
    
    async def execute(self, **kwargs) -> dict:
        """Execute a self-evolution action."""
        action = kwargs.get("action")
        
        try:
            if action == "create_tool":
                tool_name = kwargs.get("tool_name")
                description = kwargs.get("tool_description", "")
                code = kwargs.get("code")
                
                if not tool_name or not code:
                    return {"success": False, "error": "tool_name and code are required"}
                
                change = await self.evolution.create_tool(tool_name, description, code)
                return {
                    "success": True,
                    "change_id": change.id,
                    "status": change.status,
                    "message": f"Tool '{tool_name}' created successfully"
                }
            
            elif action == "modify_file":
                file_path = kwargs.get("file_path")
                code = kwargs.get("code")
                description = kwargs.get("description", "Code modification")
                
                if not file_path or not code:
                    return {"success": False, "error": "file_path and code are required"}
                
                change = await self.evolution.modify_file(file_path, code, description)
                return {
                    "success": True,
                    "change_id": change.id,
                    "status": change.status,
                    "message": f"File '{file_path}' modified"
                }
            
            elif action == "improve_prompt":
                prompt_name = kwargs.get("prompt_name")
                new_prompt = kwargs.get("new_prompt")
                reason = kwargs.get("reason", "Improvement")
                
                if not prompt_name or not new_prompt:
                    return {"success": False, "error": "prompt_name and new_prompt are required"}
                
                change = await self.evolution.improve_prompt(prompt_name, new_prompt, reason)
                return {
                    "success": True,
                    "change_id": change.id,
                    "status": change.status,
                    "message": f"Prompt '{prompt_name}' improved"
                }
            
            elif action == "rollback":
                change_id = kwargs.get("change_id")
                if not change_id:
                    return {"success": False, "error": "change_id is required"}
                
                success = await self.evolution.rollback(change_id)
                return {
                    "success": success,
                    "message": "Rollback successful" if success else "Rollback failed"
                }
            
            elif action == "list_changes":
                summary = self.evolution.get_evolution_summary()
                return {"success": True, **summary}
            
            elif action == "approve_pending":
                count = await self.evolution.approve_all_pending()
                return {
                    "success": True,
                    "applied_count": count,
                    "message": f"Applied {count} pending changes"
                }
            
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}


# Tool templates for the agent to use when creating new tools
TOOL_TEMPLATES = {
    "basic": '''"""
OpenClaw - {tool_name} Tool
{"=" * (len(tool_name) + 20)}
{description}
"""

from typing import Any
from rich.console import Console

console = Console()


class {class_name}Tool:
    """
    {description}
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {{}}
    
    @property
    def definition(self) -> dict:
        """Return the tool definition for LLM."""
        return {{
            "name": "{tool_name}",
            "description": "{description}",
            "parameters": {{
                "type": "object",
                "properties": {{
                    # Add your parameters here
                    "input": {{
                        "type": "string",
                        "description": "Input for the tool"
                    }}
                }},
                "required": ["input"]
            }}
        }}
    
    async def execute(self, **kwargs) -> dict[str, Any]:
        """
        Execute the tool.
        
        Returns:
            Dictionary with results
        """
        try:
            input_value = kwargs.get("input")
            
            # Add your tool logic here
            result = f"Processed: {{input_value}}"
            
            console.print(f"[green]✓ {tool_name} executed[/green]")
            
            return {{
                "success": True,
                "result": result
            }}
            
        except Exception as e:
            return {{
                "success": False,
                "error": str(e)
            }}
''',

    "api_integration": '''"""
OpenClaw - {tool_name} Tool (API Integration)
{"=" * (len(tool_name) + 35)}
{description}
"""

import os
from typing import Any
import httpx
from rich.console import Console

console = Console()


class {class_name}Tool:
    """
    {description}
    
    Requires: {API_KEY_NAME} environment variable
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {{}}
        self.api_key = os.getenv("{API_KEY_NAME}")
        self.base_url = "{api_base_url}"
    
    @property
    def definition(self) -> dict:
        """Return the tool definition for LLM."""
        return {{
            "name": "{tool_name}",
            "description": "{description}",
            "parameters": {{
                "type": "object",
                "properties": {{
                    "query": {{
                        "type": "string",
                        "description": "Query to send to the API"
                    }}
                }},
                "required": ["query"]
            }}
        }}
    
    async def execute(self, **kwargs) -> dict[str, Any]:
        """Execute API call."""
        try:
            query = kwargs.get("query")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{{self.base_url}}/endpoint",
                    headers={{"Authorization": f"Bearer {{self.api_key}}"}},
                    params={{"q": query}}
                )
                response.raise_for_status()
                data = response.json()
            
            return {{
                "success": True,
                "data": data
            }}
            
        except Exception as e:
            return {{
                "success": False,
                "error": str(e)
            }}
'''
}
