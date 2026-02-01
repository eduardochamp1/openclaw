"""
OpenClaw - File Manager Tool
============================
File operations: read, write, list, search, edit files.
"""

import os
import json
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from datetime import datetime

import aiofiles
from rich.console import Console

console = Console()


@dataclass
class FileInfo:
    """Information about a file."""
    name: str
    path: str
    size: int
    modified: str
    is_directory: bool
    extension: str | None = None


class FileManagerTool:
    """
    File management tool for the agent.
    
    Provides safe file operations within a workspace directory.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.workspace = Path(config.get("workspace", "./workspace")).resolve()
        self.allowed_extensions = set(config.get("allowed_extensions", [
            ".txt", ".md", ".json", ".yaml", ".yml", ".py", ".js", ".html", ".css"
        ]))
        self.max_file_size = config.get("max_file_size_mb", 10) * 1024 * 1024  # Convert to bytes
        
        # Ensure workspace exists
        self.workspace.mkdir(parents=True, exist_ok=True)
    
    def _validate_path(self, file_path: str) -> Path:
        """Validate and resolve a file path within workspace."""
        # Resolve the path
        if os.path.isabs(file_path):
            resolved = Path(file_path).resolve()
        else:
            resolved = (self.workspace / file_path).resolve()
        
        # Security: ensure path is within workspace
        try:
            resolved.relative_to(self.workspace)
        except ValueError:
            raise ValueError(f"Access denied: Path must be within workspace ({self.workspace})")
        
        return resolved
    
    def _validate_extension(self, path: Path) -> None:
        """Validate file extension is allowed."""
        if path.suffix and path.suffix.lower() not in self.allowed_extensions:
            raise ValueError(
                f"Extension '{path.suffix}' not allowed. "
                f"Allowed: {', '.join(self.allowed_extensions)}"
            )
    
    @property
    def definition(self) -> dict:
        """Return the tool definition for LLM."""
        return {
            "name": "file_manager",
            "description": (
                "Manage files in the workspace: read, write, list, search, and edit files. "
                f"Workspace: {self.workspace}. "
                f"Allowed extensions: {', '.join(self.allowed_extensions)}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "write", "append", "list", "search", "delete", "info", "mkdir"],
                        "description": "The file operation to perform"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path (relative to workspace or absolute within workspace)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (for write/append actions)"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (for search action, supports glob patterns)"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search/list recursively",
                        "default": False
                    }
                },
                "required": ["action"]
            }
        }
    
    async def execute(
        self,
        action: str,
        path: str | None = None,
        content: str | None = None,
        pattern: str | None = None,
        recursive: bool = False
    ) -> dict[str, Any]:
        """
        Execute a file operation.
        
        Actions:
            - read: Read file content
            - write: Write/overwrite file
            - append: Append to file
            - list: List directory contents
            - search: Search for files by pattern
            - delete: Delete a file
            - info: Get file information
            - mkdir: Create directory
        """
        try:
            if action == "list":
                return await self._list_directory(path or ".", recursive)
            
            elif action == "search":
                if not pattern:
                    return {"success": False, "error": "Pattern required for search"}
                return await self._search_files(pattern, path or ".", recursive)
            
            elif action == "mkdir":
                if not path:
                    return {"success": False, "error": "Path required for mkdir"}
                return await self._create_directory(path)
            
            elif action == "read":
                if not path:
                    return {"success": False, "error": "Path required for read"}
                return await self._read_file(path)
            
            elif action == "write":
                if not path:
                    return {"success": False, "error": "Path required for write"}
                if content is None:
                    return {"success": False, "error": "Content required for write"}
                return await self._write_file(path, content, mode="w")
            
            elif action == "append":
                if not path:
                    return {"success": False, "error": "Path required for append"}
                if content is None:
                    return {"success": False, "error": "Content required for append"}
                return await self._write_file(path, content, mode="a")
            
            elif action == "delete":
                if not path:
                    return {"success": False, "error": "Path required for delete"}
                return await self._delete_file(path)
            
            elif action == "info":
                if not path:
                    return {"success": False, "error": "Path required for info"}
                return await self._get_info(path)
            
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _read_file(self, path: str) -> dict[str, Any]:
        """Read file content."""
        resolved = self._validate_path(path)
        self._validate_extension(resolved)
        
        if not resolved.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        if not resolved.is_file():
            return {"success": False, "error": f"Not a file: {path}"}
        
        if resolved.stat().st_size > self.max_file_size:
            return {"success": False, "error": f"File too large (max {self.max_file_size // 1024 // 1024}MB)"}
        
        async with aiofiles.open(resolved, "r", encoding="utf-8") as f:
            content = await f.read()
        
        console.print(f"[green]📖 Read: {resolved.name}[/green]")
        
        return {
            "success": True,
            "path": str(resolved.relative_to(self.workspace)),
            "content": content,
            "size": len(content)
        }
    
    async def _write_file(self, path: str, content: str, mode: str = "w") -> dict[str, Any]:
        """Write content to file."""
        resolved = self._validate_path(path)
        self._validate_extension(resolved)
        
        # Ensure parent directory exists
        resolved.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(resolved, mode, encoding="utf-8") as f:
            await f.write(content)
        
        action = "📝 Wrote" if mode == "w" else "➕ Appended to"
        console.print(f"[green]{action}: {resolved.name}[/green]")
        
        return {
            "success": True,
            "path": str(resolved.relative_to(self.workspace)),
            "action": "write" if mode == "w" else "append",
            "bytes_written": len(content.encode("utf-8"))
        }
    
    async def _delete_file(self, path: str) -> dict[str, Any]:
        """Delete a file."""
        resolved = self._validate_path(path)
        
        if not resolved.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        if resolved.is_dir():
            return {"success": False, "error": "Use rmdir for directories (not implemented for safety)"}
        
        resolved.unlink()
        console.print(f"[red]🗑️ Deleted: {resolved.name}[/red]")
        
        return {
            "success": True,
            "path": str(resolved.relative_to(self.workspace)),
            "action": "deleted"
        }
    
    async def _list_directory(self, path: str, recursive: bool = False) -> dict[str, Any]:
        """List directory contents."""
        resolved = self._validate_path(path)
        
        if not resolved.exists():
            return {"success": False, "error": f"Directory not found: {path}"}
        
        if not resolved.is_dir():
            return {"success": False, "error": f"Not a directory: {path}"}
        
        files = []
        
        if recursive:
            for item in resolved.rglob("*"):
                if item.name.startswith("."):
                    continue
                files.append(self._get_file_info(item))
        else:
            for item in resolved.iterdir():
                if item.name.startswith("."):
                    continue
                files.append(self._get_file_info(item))
        
        # Sort: directories first, then by name
        files.sort(key=lambda x: (not x["is_directory"], x["name"].lower()))
        
        return {
            "success": True,
            "path": str(resolved.relative_to(self.workspace)),
            "count": len(files),
            "files": files
        }
    
    async def _search_files(self, pattern: str, path: str, recursive: bool = True) -> dict[str, Any]:
        """Search for files matching a pattern."""
        resolved = self._validate_path(path)
        
        if not resolved.exists():
            return {"success": False, "error": f"Directory not found: {path}"}
        
        files = []
        glob_method = resolved.rglob if recursive else resolved.glob
        
        for item in glob_method(pattern):
            if item.name.startswith("."):
                continue
            files.append(self._get_file_info(item))
        
        return {
            "success": True,
            "pattern": pattern,
            "path": str(resolved.relative_to(self.workspace)),
            "count": len(files),
            "files": files
        }
    
    async def _create_directory(self, path: str) -> dict[str, Any]:
        """Create a directory."""
        resolved = self._validate_path(path)
        
        resolved.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]📁 Created: {resolved.name}[/green]")
        
        return {
            "success": True,
            "path": str(resolved.relative_to(self.workspace)),
            "action": "created"
        }
    
    async def _get_info(self, path: str) -> dict[str, Any]:
        """Get file/directory information."""
        resolved = self._validate_path(path)
        
        if not resolved.exists():
            return {"success": False, "error": f"Path not found: {path}"}
        
        info = self._get_file_info(resolved)
        info["success"] = True
        
        return info
    
    def _get_file_info(self, path: Path) -> dict[str, Any]:
        """Get file information as dictionary."""
        stat = path.stat()
        return {
            "name": path.name,
            "path": str(path.relative_to(self.workspace)),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_directory": path.is_dir(),
            "extension": path.suffix if path.is_file() else None
        }
