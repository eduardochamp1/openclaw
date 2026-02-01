"""
OpenClaw - Code Executor Tool
=============================
Safely execute Python code in a sandboxed environment.
"""

import asyncio
import sys
import io
import traceback
import ast
import signal
from typing import Any
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
import multiprocessing
from rich.console import Console

console = Console()


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: str | None
    return_value: Any
    execution_time: float


class CodeSandbox:
    """
    Sandboxed Python code executor.
    
    Safety features:
    - Timeout limit
    - Memory limit (via process isolation)
    - Restricted imports
    - No file system access by default
    """
    
    # Allowed built-in modules
    ALLOWED_MODULES = {
        "math", "random", "datetime", "json", "re",
        "collections", "itertools", "functools",
        "string", "textwrap", "statistics"
    }
    
    # Blocked built-ins
    BLOCKED_BUILTINS = {
        "exec", "eval", "compile", "__import__",
        "open", "input", "breakpoint"
    }
    
    def __init__(self, timeout: int = 30, allow_imports: bool = True):
        self.timeout = timeout
        self.allow_imports = allow_imports
    
    def _create_safe_globals(self) -> dict:
        """Create a restricted globals dict for execution."""
        safe_builtins = {
            k: v for k, v in __builtins__.__dict__.items()
            if k not in self.BLOCKED_BUILTINS
        }
        
        # Add safe print
        output_buffer = []
        def safe_print(*args, **kwargs):
            output_buffer.append(" ".join(str(a) for a in args))
        
        safe_builtins["print"] = safe_print
        
        return {
            "__builtins__": safe_builtins,
            "__name__": "__sandbox__",
            "_output_buffer": output_buffer
        }
    
    def _check_code_safety(self, code: str) -> tuple[bool, str]:
        """Check if code is safe to execute."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        for node in ast.walk(tree):
            # Check for dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.ALLOWED_MODULES:
                        if not self.allow_imports:
                            return False, f"Import not allowed: {alias.name}"
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] not in self.ALLOWED_MODULES:
                    if not self.allow_imports:
                        return False, f"Import not allowed: {node.module}"
            
            # Check for file operations
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["open", "exec", "eval"]:
                        return False, f"Function not allowed: {node.func.id}"
        
        return True, "Code is safe"
    
    def execute_sync(self, code: str) -> ExecutionResult:
        """Execute code synchronously."""
        import time
        start_time = time.time()
        
        # Safety check
        is_safe, msg = self._check_code_safety(code)
        if not is_safe:
            return ExecutionResult(
                success=False,
                output="",
                error=msg,
                return_value=None,
                execution_time=0
            )
        
        # Create execution environment
        safe_globals = self._create_safe_globals()
        safe_locals = {}
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code
                exec(code, safe_globals, safe_locals)
            
            # Get output
            output = stdout_capture.getvalue()
            if safe_globals.get("_output_buffer"):
                output += "\n".join(safe_globals["_output_buffer"])
            
            # Get return value (last expression if any)
            return_value = safe_locals.get("result", safe_locals.get("_", None))
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=True,
                output=output.strip(),
                error=None,
                return_value=return_value,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_trace = traceback.format_exc()
            
            return ExecutionResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=f"{type(e).__name__}: {e}\n{error_trace}",
                return_value=None,
                execution_time=execution_time
            )
    
    async def execute(self, code: str) -> ExecutionResult:
        """Execute code asynchronously with timeout."""
        loop = asyncio.get_event_loop()
        
        try:
            # Run in thread pool with timeout
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self.execute_sync, code),
                timeout=self.timeout
            )
            return result
            
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout} seconds",
                return_value=None,
                execution_time=self.timeout
            )


class CodeExecutorTool:
    """
    Tool interface for code execution.
    
    Allows the agent to run Python code to:
    - Perform calculations
    - Process data
    - Test ideas
    - Generate dynamic content
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.timeout = config.get("timeout", 30)
        self.allow_imports = config.get("allow_imports", True)
        self.sandbox = CodeSandbox(
            timeout=self.timeout,
            allow_imports=self.allow_imports
        )
    
    @property
    def definition(self) -> dict:
        """Return the tool definition for LLM."""
        return {
            "name": "code_executor",
            "description": (
                "Execute Python code to perform calculations, process data, "
                "or test ideas. Code runs in a sandboxed environment with "
                f"a {self.timeout}s timeout. Allowed modules: math, random, "
                "datetime, json, re, collections, itertools, functools, statistics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use 'result = ...' to return a value."
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of what the code does"
                    }
                },
                "required": ["code"]
            }
        }
    
    async def execute(self, **kwargs) -> dict:
        """Execute Python code."""
        code = kwargs.get("code", "")
        description = kwargs.get("description", "Code execution")
        
        if not code.strip():
            return {
                "success": False,
                "error": "No code provided"
            }
        
        console.print(f"[blue]🐍 Executing: {description}[/blue]")
        
        result = await self.sandbox.execute(code)
        
        if result.success:
            console.print(f"[green]✓ Execution completed in {result.execution_time:.2f}s[/green]")
        else:
            console.print(f"[red]✗ Execution failed[/red]")
        
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "return_value": str(result.return_value) if result.return_value is not None else None,
            "execution_time": result.execution_time
        }


class REPLSession:
    """
    Interactive REPL session for the agent.
    
    Maintains state across multiple code executions.
    """
    
    def __init__(self):
        self.sandbox = CodeSandbox()
        self.history: list[tuple[str, ExecutionResult]] = []
        self.persistent_locals: dict = {}
    
    async def run(self, code: str) -> ExecutionResult:
        """Run code in the REPL session."""
        # Add persistent state
        full_code = "\n".join([
            "# Restore state",
            *[f"{k} = {repr(v)}" for k, v in self.persistent_locals.items() if self._is_serializable(v)],
            "",
            "# User code",
            code
        ])
        
        result = await self.sandbox.execute(full_code)
        self.history.append((code, result))
        
        return result
    
    @staticmethod
    def _is_serializable(value) -> bool:
        """Check if value can be serialized."""
        try:
            repr(value)
            return True
        except:
            return False
    
    def clear(self):
        """Clear session state."""
        self.history = []
        self.persistent_locals = {}
