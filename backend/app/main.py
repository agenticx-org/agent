from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
from typing import Dict, Any
from .stream_handler import process_message

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for handling chat messages and streaming responses.
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                # Parse the received data
                event = json.loads(data)
                
                # Define the send_chunk callback function
                async def send_chunk(chunk: Dict[str, Any]) -> bool:
                    """
                    Send a chunk of data back to the client.
                    Returns True if successful, False if client disconnected.
                    """
                    try:
                        await websocket.send_json(chunk)
                        return True
                    except WebSocketDisconnect:
                        logger.warning("Client disconnected while sending chunk")
                        return False
                    except Exception as e:
                        logger.error(f"Error sending chunk: {e}")
                        return False
                
                # Process the message and stream the response
                await process_message(event, send_chunk)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON data: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Internal server error"
                })
                
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the server is running.
    """
    return {"status": "healthy"} 