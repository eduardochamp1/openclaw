"""
OpenClaw - Reasoning Engine
===========================
Cognitive loop: Perceive → Orient → Act → Reflect (POAR)
"""

from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


class ThinkingPhase(Enum):
    PERCEIVE = "perceive"      # Understand input
    ORIENT = "orient"          # Analyze context and options
    ACT = "act"               # Execute action
    REFLECT = "reflect"       # Evaluate results


@dataclass
class ThinkingStep:
    """A single step in the reasoning process."""
    phase: ThinkingPhase
    thought: str
    confidence: float  # 0.0 to 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """Result of the reasoning process."""
    steps: list[ThinkingStep]
    final_action: str | None
    tool_to_use: str | None
    tool_input: dict[str, Any] | None
    confidence: float
    should_respond: bool
    response: str | None = None


class ReasoningEngine:
    """
    Implements the POAR cognitive cycle.
    
    This engine provides structured thinking for the agent,
    helping it reason through problems systematically.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.thinking_history: list[ThinkingStep] = []
    
    def _log_thought(self, phase: ThinkingPhase, thought: str, confidence: float = 1.0) -> ThinkingStep:
        """Log a thinking step."""
        step = ThinkingStep(
            phase=phase,
            thought=thought,
            confidence=confidence
        )
        self.thinking_history.append(step)
        
        if self.verbose:
            phase_emoji = {
                ThinkingPhase.PERCEIVE: "👁️",
                ThinkingPhase.ORIENT: "🧭",
                ThinkingPhase.ACT: "⚡",
                ThinkingPhase.REFLECT: "🪞"
            }
            emoji = phase_emoji.get(phase, "💭")
            confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            
            console.print(Panel(
                f"{thought}\n\n[dim]Confidence: [{confidence_bar}] {confidence:.0%}[/dim]",
                title=f"{emoji} {phase.value.upper()}",
                border_style="blue" if confidence > 0.7 else "yellow"
            ))
        
        return step
    
    def perceive(self, user_input: str, context: dict[str, Any] | None = None) -> ThinkingStep:
        """
        PERCEIVE: Understand what the user is asking.
        
        Analyzes:
        - Intent: What does the user want?
        - Entities: What are the key elements?
        - Constraints: What limitations exist?
        """
        analysis = {
            "raw_input": user_input,
            "input_length": len(user_input),
            "has_question": "?" in user_input,
            "has_command": any(cmd in user_input.lower() for cmd in [
                "create", "write", "search", "find", "read", "delete", "list"
            ]),
            "context_available": context is not None
        }
        
        # Identify likely intent
        intent = self._classify_intent(user_input)
        analysis["intent"] = intent
        
        thought = (
            f"User request: \"{user_input[:100]}{'...' if len(user_input) > 100 else ''}\"\n"
            f"Detected intent: {intent}\n"
            f"Has question: {analysis['has_question']}\n"
            f"Has command: {analysis['has_command']}"
        )
        
        return self._log_thought(ThinkingPhase.PERCEIVE, thought, confidence=0.9)
    
    def orient(
        self,
        perception: ThinkingStep,
        available_tools: list[str],
        memory_context: list[dict] | None = None
    ) -> ThinkingStep:
        """
        ORIENT: Analyze situation and determine best approach.
        
        Considers:
        - Available tools and capabilities
        - Past context and memory
        - Multiple approaches and their tradeoffs
        """
        # Extract intent from perception
        intent = perception.metadata.get("intent", "unknown")
        
        # Determine tool needs
        tool_analysis = self._analyze_tool_needs(perception.thought, available_tools)
        
        # Build orientation
        thought_parts = [
            f"Available tools: {', '.join(available_tools)}",
            f"Recommended tool: {tool_analysis.get('recommended_tool', 'none')}",
            f"Reasoning: {tool_analysis.get('reasoning', 'N/A')}"
        ]
        
        if memory_context:
            thought_parts.append(f"Context from {len(memory_context)} previous messages available")
        
        thought = "\n".join(thought_parts)
        confidence = tool_analysis.get("confidence", 0.8)
        
        step = self._log_thought(ThinkingPhase.ORIENT, thought, confidence)
        step.metadata = tool_analysis
        return step
    
    def plan_action(
        self,
        orientation: ThinkingStep,
        user_input: str
    ) -> ThinkingStep:
        """
        ACT (Planning): Determine specific action to take.
        
        Returns the action plan before execution.
        """
        tool = orientation.metadata.get("recommended_tool")
        
        if tool:
            tool_input = self._generate_tool_input(tool, user_input)
            thought = (
                f"Action: Use tool '{tool}'\n"
                f"Parameters: {tool_input}"
            )
            confidence = 0.85
        else:
            thought = "Action: Generate direct response (no tool needed)"
            tool_input = None
            confidence = 0.9
        
        step = self._log_thought(ThinkingPhase.ACT, thought, confidence)
        step.metadata = {
            "tool": tool,
            "tool_input": tool_input
        }
        return step
    
    def reflect(
        self,
        action: ThinkingStep,
        result: Any,
        success: bool
    ) -> ThinkingStep:
        """
        REFLECT: Evaluate the outcome and learn.
        
        Analyzes:
        - Was the action successful?
        - What can be improved?
        - Should we try again?
        """
        if success:
            thought = (
                f"Action completed successfully.\n"
                f"Result type: {type(result).__name__}\n"
                f"Learning: Action was appropriate for this request."
            )
            confidence = 0.9
        else:
            thought = (
                f"Action failed or produced unexpected result.\n"
                f"Error/Result: {str(result)[:200]}\n"
                f"Learning: May need different approach."
            )
            confidence = 0.5
        
        step = self._log_thought(ThinkingPhase.REFLECT, thought, confidence)
        step.metadata = {
            "success": success,
            "should_retry": not success and confidence > 0.3
        }
        return step
    
    def _classify_intent(self, text: str) -> str:
        """Classify user intent from text."""
        text_lower = text.lower()
        
        intents = {
            "search": ["search", "find", "look up", "what is", "who is", "when", "where"],
            "create": ["create", "write", "make", "generate", "build"],
            "read": ["read", "show", "display", "open", "view", "cat"],
            "modify": ["edit", "change", "update", "modify", "fix"],
            "delete": ["delete", "remove", "clear"],
            "list": ["list", "show all", "ls", "dir"],
            "explain": ["explain", "how", "why", "what does"],
            "chat": ["hello", "hi", "hey", "thanks", "help"]
        }
        
        for intent, keywords in intents.items():
            if any(kw in text_lower for kw in keywords):
                return intent
        
        return "general"
    
    def _analyze_tool_needs(self, perception: str, available_tools: list[str]) -> dict:
        """Analyze which tool would best serve the request."""
        perception_lower = perception.lower()
        
        tool_mapping = {
            "web_search": ["search", "find", "look up", "current", "latest", "news", "who is", "what is"],
            "file_manager": ["file", "read", "write", "create", "save", "list", "directory", "folder"]
        }
        
        scores = {}
        for tool, keywords in tool_mapping.items():
            if tool in available_tools:
                score = sum(1 for kw in keywords if kw in perception_lower)
                if score > 0:
                    scores[tool] = score
        
        if scores:
            best_tool = max(scores.keys(), key=lambda t: scores[t])
            return {
                "recommended_tool": best_tool,
                "confidence": min(0.9, 0.5 + scores[best_tool] * 0.1),
                "reasoning": f"Keywords matched for {best_tool}",
                "alternatives": [t for t in scores.keys() if t != best_tool]
            }
        
        return {
            "recommended_tool": None,
            "confidence": 0.8,
            "reasoning": "No specific tool needed - can respond directly",
            "alternatives": []
        }
    
    def _generate_tool_input(self, tool: str, user_input: str) -> dict:
        """Generate appropriate input for a tool based on user request."""
        if tool == "web_search":
            # Extract search query
            # Simple extraction - in production, use LLM for this
            query = user_input
            for prefix in ["search for", "find", "look up", "what is", "who is"]:
                if prefix in user_input.lower():
                    idx = user_input.lower().index(prefix) + len(prefix)
                    query = user_input[idx:].strip()
                    break
            
            return {
                "query": query,
                "max_results": 5
            }
        
        elif tool == "file_manager":
            # Determine file action
            text_lower = user_input.lower()
            
            if "read" in text_lower or "show" in text_lower:
                action = "read"
            elif "write" in text_lower or "create" in text_lower or "save" in text_lower:
                action = "write"
            elif "list" in text_lower or "ls" in text_lower:
                action = "list"
            elif "delete" in text_lower or "remove" in text_lower:
                action = "delete"
            else:
                action = "list"
            
            return {
                "action": action,
                "path": "."  # Will be refined by LLM
            }
        
        return {}
    
    def get_full_reasoning(self) -> str:
        """Get formatted reasoning chain."""
        if not self.thinking_history:
            return "No reasoning recorded."
        
        lines = ["## Reasoning Chain\n"]
        
        for i, step in enumerate(self.thinking_history, 1):
            lines.append(f"### Step {i}: {step.phase.value.upper()}")
            lines.append(f"*Confidence: {step.confidence:.0%}*\n")
            lines.append(step.thought)
            lines.append("")
        
        return "\n".join(lines)
    
    def clear_history(self) -> None:
        """Clear reasoning history for new request."""
        self.thinking_history = []
