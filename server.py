#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import os
import uuid
from typing import Any, Callable, Dict, List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from agent.agent import Agent

# Load environment variables
load_dotenv()

# Configuration from env vars
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "claude-3-7-sonnet-latest")
RAW_AUTHORIZED_IMPORTS = os.getenv("AUTHORIZED_IMPORTS", "math,random,datetime,json,re")
AUTHORIZED_IMPORTS = [
    imp.strip() for imp in RAW_AUTHORIZED_IMPORTS.split(",") if imp.strip()
]

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "server.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("WebSocket_Server")

app = FastAPI(title="Agent WebSocket Server")


# WebSocket Agent subclass that overrides _output_json for real-time streaming
class WebSocketAgent(Agent):
    """Agent implementation with WebSocket output streaming."""

    def __init__(
        self,
        task: str,
        output_callback: Callable,
        authorized_imports=None,
        model_id=None,
    ):
        self.output_callback = output_callback
        super().__init__(
            task=task, authorized_imports=authorized_imports, model_id=model_id
        )

    def _output_json(self, data: Dict[str, Any]) -> None:
        """Override to forward outputs to WebSocket instead of stdout."""
        json_str = json.dumps(data)
        # Call the callback directly - it will handle the async parts
        self.output_callback(json_str)
        # Also log to server logs
        logger.info(f"Agent output: {json_str[:100]}...")


# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.agents: Dict[str, WebSocketAgent] = {}
        # For managing output callbacks
        self.output_callbacks: Dict[str, Callable] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.agents:
            del self.agents[client_id]
        if client_id in self.output_callbacks:
            del self.output_callbacks[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

    def register_output_callback(self, client_id: str, callback: Callable):
        """Register a callback for agent outputs."""
        self.output_callbacks[client_id] = callback

    def create_agent(self, client_id: str, task: str, output_callback: Callable):
        """Create a WebSocketAgent with output callback."""
        self.agents[client_id] = WebSocketAgent(
            task=task,
            authorized_imports=AUTHORIZED_IMPORTS,
            model_id=MODEL_ID,
            output_callback=output_callback,
        )
        return self.agents[client_id]

    def get_agent(self, client_id: str):
        return self.agents.get(client_id)


manager = ConnectionManager()

# Mount static files for UI
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def redirect_to_ui():
    """Redirect root to the UI page"""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/static/index.html")


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    # Create a shared event loop reference
    loop = asyncio.get_running_loop()

    try:
        while True:
            data = await websocket.receive_json()
            command = data.get("command")

            if command == "initialize":
                task = data.get("task", "")
                if not task:
                    await manager.send_message(
                        client_id, {"type": "error", "content": "No task provided"}
                    )
                    continue

                try:
                    # Create a synchronous callback that forwards to our async method
                    def sync_output_callback(message_str):
                        """Synchronous callback that directly sends to websocket."""
                        try:
                            # Parse the JSON here
                            message_data = json.loads(message_str)
                            # Use run_coroutine_threadsafe to safely run async code from sync context
                            asyncio.run_coroutine_threadsafe(
                                websocket.send_json(message_data), loop
                            )
                        except Exception as e:
                            logger.error(f"Error in output callback: {e}")
                            # For error cases, still try to send something
                            asyncio.run_coroutine_threadsafe(
                                websocket.send_json(
                                    {"type": "log", "content": str(message_str)}
                                ),
                                loop,
                            )

                    # Initialize the agent with our custom output callback
                    agent = manager.create_agent(client_id, task, sync_output_callback)

                    # Send initial status
                    await manager.send_message(
                        client_id,
                        {
                            "type": "status",
                            "content": "Agent initialized and running...",
                        },
                    )

                    # Run the agent in a separate thread to avoid blocking the event loop
                    import threading

                    def run_agent():
                        try:
                            final_answer = agent.run()
                            # Signal completion in the event loop
                            asyncio.run_coroutine_threadsafe(
                                websocket.send_json(
                                    {
                                        "type": "execution_complete",
                                        "content": "Agent execution completed",
                                    }
                                ),
                                loop,
                            )
                        except Exception as e:
                            logger.error(f"Error running agent: {e}")
                            asyncio.run_coroutine_threadsafe(
                                websocket.send_json(
                                    {
                                        "type": "error",
                                        "content": f"Error running agent: {e}",
                                    }
                                ),
                                loop,
                            )

                    # Start agent in background thread
                    threading.Thread(target=run_agent, daemon=True).start()

                except Exception as e:
                    logger.error(f"Error initializing agent: {e}", exc_info=True)
                    await manager.send_message(
                        client_id,
                        {"type": "error", "content": f"Error initializing agent: {e}"},
                    )

            elif command == "terminate":
                # Clean up agent if needed
                await manager.send_message(
                    client_id, {"type": "status", "content": "Agent terminated"}
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(client_id)


def main():
    parser = argparse.ArgumentParser(description="Run Agent WebSocket server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to"
    )

    args = parser.parse_args()

    if not ANTHROPIC_API_KEY:
        logger.error(
            "No Anthropic credentials found in environment variables. Please set ANTHROPIC_API_KEY."
        )
        exit(1)

    # Start the server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
