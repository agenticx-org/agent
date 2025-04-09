# Agent API Server with SSE Streaming

This project provides a FastAPI-based server that runs an agentic system with Server-Sent Events (SSE) streaming for real-time communication with the frontend.

## Features

- **JSON-structured Output**: All agent outputs are formatted as JSON with consistent fields, making it easy to parse and display in the frontend.
- **SSE Streaming**: Real-time updates from the agent are streamed to the frontend as they occur.
- **Web Interface**: Simple web interface included for testing the API.
- **Configurable**: Easily configure the agent behavior through environment variables.

## Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables (create a `.env` file):

```
ANTHROPIC_API_KEY=your_api_key_here
MODEL_ID=claude-3-7-sonnet-latest
AUTHORIZED_IMPORTS=math,random,datetime,json,re
```

## Running the Server

Start the server with:

```bash
python server.py
```

The server will be available at `http://localhost:8000`

## API Endpoints

### GET /

Serves the web interface for testing the API.

### POST /api/query

Sends a query to the agent and streams back the responses.

**Request Body:**

```json
{
  "query": "Your question or task here"
}
```

**Response:**
Server-Sent Events stream with JSON payloads in the following format:

```json
{
  "type": "status|thought|tool_call|tool_result|plan|findings|final_answer|error|warning|execution_complete",
  "content": "Content of the message",
  ...additional fields depending on type
}
```

## JSON Response Structure

Depending on the `type` field, JSON responses will have different structures:

- **status**: Simple status messages

  ```json
  { "type": "status", "content": "Agent initialized successfully." }
  ```

- **thought**: Agent's reasoning process

  ```json
  { "type": "thought", "content": "I need to calculate 2+2..." }
  ```

- **tool_call**: When the agent calls a tool

  ```json
  {
    "type": "tool_call",
    "tool": "execute_python",
    "args": { "code": "print(2+2)" }
  }
  ```

- **tool_result**: Result of a tool execution

  ```json
  {
    "type": "tool_result",
    "tool": "execute_python",
    "success": true,
    "stdout": "4\n"
  }
  ```

- **code**: Code the agent will execute

  ```json
  { "type": "code", "content": "print(2+2)" }
  ```

- **plan**: Agent's plan in markdown format

  ```json
  {
    "type": "plan",
    "content": "# Plan\n- [ ] Calculate 2+2\n- [ ] Return result"
  }
  ```

- **findings**: Research findings in markdown

  ```json
  { "type": "findings", "content": "# Findings\n- The result of 2+2 is 4" }
  ```

- **final_answer**: The agent's final response

  ```json
  { "type": "final_answer", "content": "The answer is 4." }
  ```

- **error/warning**: Error or warning messages

  ```json
  { "type": "error", "content": "An error occurred while executing tool." }
  ```

- **execution_complete**: Signals the end of execution
  ```json
  { "type": "execution_complete" }
  ```

## CLI Usage

You can also use the command-line interface:

```bash
python main.py "Your question or task here"
```

Add the `--verbose` flag to see detailed logs:

```bash
python main.py "Your question or task here" --verbose
```

## Development

The server uses FastAPI's auto-reload feature. Any changes to the server code will automatically restart the server.

For frontend development, edit the files in the `static` directory.
