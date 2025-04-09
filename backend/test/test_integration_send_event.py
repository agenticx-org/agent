import pytest
import asyncio
import websockets
import json
import requests
from pathlib import Path

# Use consistent host and port
HOST = "0.0.0.0"
PORT = 8000

async def check_websocket_endpoint(url, timeout=2):
    """Check if a WebSocket endpoint is available and accepting connections."""
    print(f"Testing WebSocket availability at {url}")
    try:
        # Set a short timeout to avoid blocking for too long
        async with websockets.connect(url, open_timeout=timeout) as ws:
            await ws.send(json.dumps({"type": "ping"}))
            await asyncio.wait_for(ws.recv(), timeout=timeout)
            print(f"WebSocket at {url} is available and responding")
            return True
    except (websockets.exceptions.WebSocketException, 
            asyncio.TimeoutError, 
            ConnectionRefusedError) as e:
        print(f"WebSocket check failed: {type(e).__name__}: {str(e)}")
        return False

@pytest.mark.asyncio
async def test_send_event_and_receive_response():

    # test WebSocket communication
    websocket_url = f"ws://{HOST}:{PORT}/ws/chat"
    try:
        # Connect to the WebSocket
        async with websockets.connect(websocket_url, open_timeout=5) as ws:
            # Send a test event that matches the expected format
            test_event = {
                "name" : "chat_message",
                "message": {
                    "content" : "Hi, I would like to create a research report about gold please",
                    "role": "user"
                },
                "model":"claude sonnet 3.7 thinking",
                "mode":"chat",
                "block_ids":"234234234AWEFAWEF",
                "document_id":"AWEFAWE3453453453",
                "chat_id":"34234DFAWFA", # to load the context of the conversation
                "user_id":"23423423AWEFAWEFAWEF", # current authenticated user
            }
            print(f"Sending test event: {test_event}")
            await ws.send(json.dumps(test_event))
            
            
            # Now wait for the content response (could be multiple chunks)
            received_complete = False
            received_chunks = []
            
            # Set a maximum wait time for the complete response
            max_wait_time = 15  # seconds
            start_time = asyncio.get_event_loop().time()
            
            while not received_complete and (asyncio.get_event_loop().time() - start_time) < max_wait_time:
                response = await asyncio.wait_for(ws.recv(), timeout=2)
                chunk = json.loads(response)
                print(f"Received chunk: {chunk}")
                received_chunks.append(chunk)
                
                # Check if this is the completion message
                if chunk.get("type") == "status" and chunk.get("status") == "complete":
                    received_complete = True
                    break
            
            # Validate that we received at least some chunks and the complete status
            assert len(received_chunks) > 0, "Did not receive any response chunks"
            assert received_complete, "Did not receive completion status"
            
            print(f"WebSocket send and receive test successful! Received {len(received_chunks)} chunks")
            return True
    except (websockets.exceptions.WebSocketException, 
            asyncio.TimeoutError, 
            ConnectionRefusedError) as e:
        pytest.fail(f"WebSocket test failed: {type(e).__name__}: {str(e)}")
