import uvicorn
from search.search_api import *
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
from typing import Dict, Any
from backend.app.stream_handler import process_message

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

signal_search_instance = SignalSearch(load_local_model=True, model_path="search/bert_topic.pkl")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/search/{user_query}")
async def read_item(user_query: str):
    collected_data = signal_search_instance.crawl(
            query=user_query,
            n_seed_topics=5,       # Start with 5 most relevant topics
            max_steps=15,          # Limit exploration depth/breadth per seed
            novelty_threshold=0.75, # Lower threshold -> more docs considered novel
            stop_patience=4,       # Stop if avg collection < 1 doc/step for 4 steps
            min_new_docs_rate=1,
            max_docs_per_topic=20, # Process max 20 docs from each visited node
            max_buffer_check=300   # Compare new docs against max 300 buffer docs for speed
        )
    return {"data": collected_data}

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

def main():
    """Entry point for the application using Gunicorn."""
    return app

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
