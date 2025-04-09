# Anthropic API Setup

## Overview

The backend uses the Anthropic API to generate streaming responses for agent mode. The implementation is in `agent_handler.py`.

## Setup

1. Get an API key from the [Anthropic Console](https://console.anthropic.com/)
2. Set the API key in your environment:

```bash
# Add this to your .env file
ANTHROPIC_API_KEY=your_api_key_here
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The default model is `claude-3-sonnet-20240229`, but you can specify a different model in the request payload:

```json
{
  "model": "claude-3-haiku-20240307",
  "mode": "agent",
  "message": {
    "content": "What is machine learning?"
  },
  "system_prompt": "You are a helpful assistant."
}
```

## Stream Handler Integration

When a request specifies `mode: "agent"`, the system automatically uses the Anthropic API for message processing:

1. The `stream_handler.py` detects agent mode requests
2. It delegates processing to `agent_handler.process_agent_message`
3. The Anthropic API streams responses which are sent to the client in real-time

## Error Handling

If the Anthropic API key is not set or invalid, the system will return an appropriate error message. API errors are logged and returned as part of the response stream.
