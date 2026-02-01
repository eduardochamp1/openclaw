"""
OpenClaw - Web Server
=====================
FastAPI server with WebSocket for real-time chat.
"""

import asyncio
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
print(f"Loading env from {env_path}")
print(f"GROQ_API_KEY present: {'GROQ_API_KEY' in os.environ}")

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.agent import OpenClawAgent, AgentConfig

# Initialize FastAPI
app = FastAPI(
    title="OpenClaw Agent",
    description="Autonomous AI Agent with Web Interface",
    version="1.0.0"
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions
sessions: dict[str, OpenClawAgent] = {}


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


def get_or_create_agent(session_id: str | None = None) -> tuple[OpenClawAgent, str]:
    """Get existing agent or create new one."""
    if session_id and session_id in sessions:
        return sessions[session_id], session_id
    
    # Create new session
    new_session_id = session_id or str(uuid.uuid4())[:8]
    
    # Load config
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    if config_path.exists():
        agent_config = AgentConfig.from_yaml(str(config_path))
    else:
        agent_config = AgentConfig()
    
    agent_config.verbose = False  # Disable console output for web
    
    agent = OpenClawAgent(agent_config)
    sessions[new_session_id] = agent
    
    return agent, new_session_id


# Serve static files
static_path = PROJECT_ROOT / "web" / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main chat interface."""
    html_path = PROJECT_ROOT / "web" / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>OpenClaw Agent</h1><p>Web interface not found. Run from project root.</p>")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """REST endpoint for chat."""
    try:
        agent, session_id = get_or_create_agent(message.session_id)
        response = await agent.run(message.message, session_id)
        return ChatResponse(response=response, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    
    try:
        agent, session_id = get_or_create_agent(session_id)
        await agent.memory.start_session(session_id)
        
        # Send welcome message
        await websocket.send_json({
            "type": "system",
            "content": f"🦀 OpenClaw conectado! Sessão: {session_id}",
            "session_id": session_id
        })
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            user_message = data.get("message", "")
            
            if not user_message:
                continue
            
            # Send typing indicator
            await websocket.send_json({
                "type": "typing",
                "content": "Pensando..."
            })
            
            try:
                # Get response from agent
                response = await agent.run(user_message, session_id)
                
                # Send response
                await websocket.send_json({
                    "type": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "content": f"Erro: {str(e)}"
                })
    
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


@app.get("/api/sessions")
async def list_sessions():
    """List active sessions."""
    return {
        "sessions": list(sessions.keys()),
        "count": len(sessions)
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "sessions_active": len(sessions)
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
