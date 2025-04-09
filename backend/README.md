# WebSocket Chat Backend

This is a FastAPI-based WebSocket server for handling real-time chat communications with streaming responses.

## Features

- WebSocket-based communication
- Streaming responses
- Support for different LLM models
- Stateful connection management

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   python run.py
   ```

The server will be available at `http://localhost:8000`.

## WebSocket Endpoint

- `/ws/chat` - Main WebSocket endpoint for chat communications

## API Documentation

Once the server is running, visit:
- `http://localhost:8000/docs` for Swagger UI documentation
- `http://localhost:8000/redoc` for ReDoc documentation 


## Testing

python -m pytest -xvs test/test_integration_send_event.py

