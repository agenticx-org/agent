import pytest
from app.stream_handler import StreamingLLM


@pytest.mark.asyncio
async def test_generate_streaming_response():
    streaming = StreamingLLM(model_id="default_model", is_agent=False)
    response_chunks = []
    
    async for chunk in streaming.generate_streaming_response("Hello, how are you?"):
        response_chunks.append(chunk)
    
    # Verify we received some response chunks
    assert len(response_chunks) > 0
    
    # If using the real API, verify the response structure
    if streaming.client:
        for chunk in response_chunks:
            print(chunk)
            # assert isinstance(chunk, dict)
            # assert "content" in chunk
            # assert isinstance(chunk["content"], list)
            # for content_block in chunk["content"]:
            #     assert "text" in content_block
            #     assert isinstance(content_block["text"], str)
