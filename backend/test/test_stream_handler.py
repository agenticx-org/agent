import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import AsyncAnthropic

from app.stream_handler import StreamingLLM

@pytest.fixture
def mock_anthropic_client():
    with patch("app.stream_handler.AsyncAnthropic") as mock:
        client = AsyncMock()
        mock.return_value = client
        yield client

@pytest.fixture
def streaming_llm(mock_anthropic_client):
    return StreamingLLM(model_id="default_model", is_agent=False)

@pytest.fixture
def streaming_llm_no_api():
    # Temporarily remove API key if it exists
    api_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    llm = StreamingLLM(model_id="default_model", is_agent=False)
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    return llm

@pytest.mark.asyncio
async def test_generate_streaming_response_with_api(streaming_llm, mock_anthropic_client):
    # Mock the streaming response
    mock_stream = AsyncMock()
    mock_stream.__aiter__.return_value = [
        MagicMock(type="content_block_delta", delta=MagicMock(text="Hello")),
        MagicMock(type="content_block_delta", delta=MagicMock(text=" world")),
        MagicMock(type="content_block_delta", delta=MagicMock(text="!")),
    ]
    mock_anthropic_client.messages.create.return_value = mock_stream

    # Test data
    message = "Hello"
    chat_history = [
        {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
    ]

    # Collect all chunks
    chunks = []
    async for chunk in streaming_llm.generate_streaming_response(message, chat_history):
        chunks.append(chunk)

    # Verify chunks
    assert len(chunks) == 3
    assert chunks[0] == {"type": "text", "text": "Hello"}
    assert chunks[1] == {"type": "text", "text": " world"}
    assert chunks[2] == {"type": "text", "text": "!"}

    # Verify API call
    mock_anthropic_client.messages.create.assert_called_once()
    call_args = mock_anthropic_client.messages.create.call_args[1]
    assert call_args["model"] == "claude-3-5-sonnet-latest"
    assert call_args["max_tokens"] == 1024
    assert call_args["stream"] is True
    assert len(call_args["messages"]) == 3  # chat history + current message

@pytest.mark.asyncio
async def test_generate_streaming_response_no_api(streaming_llm_no_api):
    # Test data
    message = "Hello"
    chat_history = []

    # Collect chunks
    chunks = []
    async for chunk in streaming_llm_no_api.generate_streaming_response(message, chat_history):
        chunks.append(chunk)

    # Verify fallback behavior
    assert len(chunks) == 1
    assert chunks[0] == {"type": "text", "text": "No Anthropic API key found"}

@pytest.mark.asyncio
async def test_generate_streaming_response_api_error(streaming_llm, mock_anthropic_client):
    # Mock API error
    mock_anthropic_client.messages.create.side_effect = Exception("API Error")

    # Test data
    message = "Hello"
    chat_history = []

    # Collect chunks
    chunks = []
    async for chunk in streaming_llm.generate_streaming_response(message, chat_history):
        chunks.append(chunk)

    # Verify error handling
    assert len(chunks) == 1
    assert "Error using Anthropic API" in chunks[0]["text"]

@pytest.mark.asyncio
async def test_generate_streaming_response_empty_chat_history(streaming_llm, mock_anthropic_client):
    # Mock the streaming response
    mock_stream = AsyncMock()
    mock_stream.__aiter__.return_value = [
        MagicMock(type="content_block_delta", delta=MagicMock(text="Response")),
    ]
    mock_anthropic_client.messages.create.return_value = mock_stream

    # Test with empty chat history
    message = "Hello"
    chat_history = []

    # Collect chunks
    chunks = []
    async for chunk in streaming_llm.generate_streaming_response(message, chat_history):
        chunks.append(chunk)

    # Verify API call with empty chat history
    mock_anthropic_client.messages.create.assert_called_once()
    call_args = mock_anthropic_client.messages.create.call_args[1]
    assert len(call_args["messages"]) == 1  # Only current message 

@pytest.mark.asyncio
async def test_generate_streaming_response_direct():
    """Test that directly calls generate_streaming_response without mocks"""
    # Create StreamingLLM instance
    llm = StreamingLLM(model_id="default_model", is_agent=False)
    
    # Test data
    message = "What is 2+2? Give a short answer."
    chat_history = []
    
    # Collect chunks
    chunks = []
    async for chunk in llm.generate_streaming_response(message, chat_history):
        chunks.append(chunk)
    
    # Basic validation
    assert len(chunks) > 0, "Should receive at least one chunk"
    
    # If we have an API key, we should get actual responses
    if os.environ.get("ANTHROPIC_API_KEY"):
        # Verify the response structure
        for chunk in chunks:
            assert isinstance(chunk, dict), "Each chunk should be a dictionary"
            assert "type" in chunk, "Each chunk should have a type"
            assert "text" in chunk, "Each chunk should have text content"
            assert isinstance(chunk["text"], str), "Text content should be a string"
            assert len(chunk["text"]) > 0, "Text content should not be empty"
    else:
        # If no API key, we should get the fallback message
        assert len(chunks) == 1, "Should get exactly one chunk for fallback"
        assert chunks[0]["text"] == "No Anthropic API key found" 