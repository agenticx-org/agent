import asyncio
import json
import logging
import os
import random
import time
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List

from anthropic import AsyncAnthropic

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingLLM:
    """
    Handles streaming responses from Anthropic's Claude models.
    Falls back to simulated responses if API key is not available.
    """

    def __init__(self, model_id: str, is_agent: bool):
        self.model_id = model_id
        self.is_agent = is_agent
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.client = AsyncAnthropic(api_key=self.api_key) if self.api_key else None

        # Map model_id to Anthropic model names if needed
        self.anthropic_model = self._map_model_to_anthropic(model_id)

    def _map_model_to_anthropic(self, model_id: str) -> str:
        """Map internal model IDs to Anthropic model names"""
        model_mapping = {
            "default_model": "claude-3-5-sonnet-latest",
            # Add other model mappings as needed
        }
        return model_mapping.get(model_id, "claude-3-5-sonnet-latest")

    async def generate_streaming_response(
        self, message: str, chat_history: List[Dict[str, Any]] = []
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate a streaming response using Anthropic's Claude API.

        Args:
            message: The current user message
            chat_history: Previous messages in the conversation
        """
        # If no API key is available, fall back to simulated responses
        if not self.client:
            logger.warning("No Anthropic API key found")
            yield {"type": "text", "text": "No Anthropic API key found"}
        try:
            # Format chat history for Anthropic API
            messages = []

            # Add chat history
            for msg in chat_history:
                role = "user" if msg.get("role") == "user" else "assistant"
                # Extract text content from the message
                content_parts = msg.get("content", [])
                text_parts = []
                for part in content_parts:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))

                if text_parts:
                    messages.append({"role": role, "content": " ".join(text_parts)})

            # Add current message
            messages.append(
                {
                    "role": "user",
                    "content": message,
                }
            )

            # Create a streaming response from Anthropic
            stream = await self.client.messages.create(
                max_tokens=1024,
                messages=messages,
                model=self.anthropic_model,
                stream=True
            )

            # Process the streaming response
            async for event in stream:
                print(event)
                if event.type == "content_block_delta":
                    # Extract the text from the content block delta
                    yield {"type": "text", "text": event.delta.text}
                elif event.type == "message_delta":
                    # Handle message delta events if needed
                    continue
                elif event.type == "content_block_start":
                    # Handle content block start events if needed
                    continue
                elif event.type == "content_block_stop":
                    # Handle content block stop events if needed
                    continue

                # Add a small sleep to improve cancellation responsiveness
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Error using Anthropic API: {str(e)}")
            # Fall back to simulated responses in case of API error
            logger.info("Falling back to simulated responses")
            yield {"type": "text", "text": f"Error using Anthropic API: {str(e)}"}


async def process_message(
    event: Dict[str, Any], send_chunk: Callable[[Dict[str, Any]], Awaitable[bool]]
) -> None:
    """
    Process a message and stream the response using the provided callback.

    Args:
        event: The event data, including message content, model, and block_ids
        send_chunk: Async function to send chunks back to the client
    """
    # Set a max time for the entire process
    MAX_PROCESS_TIME = 60  # seconds
    start_time = time.time()

    logger.info(f"Processing message for block_ids: {event.get('block_ids', '')}")
    logger.info(f"Event: {event}")

    try:
        # Extract event data
        model = event.get("model", "default_model")
        mode = event.get("mode", "chat")
        message_content = event.get("message", {}).get("content", "")
        block_ids = event.get("block_ids", "")
        chat_history = event.get("chat_history", [])

        logger.info(f"Processing message for block_ids: {block_ids}")
        logger.info(f"Chat history length: {len(chat_history)}")

        # Create simulator
        simulator = StreamingLLM(model, mode == "agent")

        logger.info(f"About to send thinking status")

        # Send thinking state
        sent = await send_chunk({"type": "status", "status": "thinking"})
        if not sent:
            logger.warning("Failed to send thinking status, client likely disconnected")
            return

        # Simulate processing delay but check for cancellation
        for _ in range(8):  # 0.8 seconds in 0.1s increments
            await asyncio.sleep(0.1)
            # Regular check if we should exit (e.g., due to timeout)
            if time.time() - start_time > MAX_PROCESS_TIME:
                logger.warning(
                    f"Processing exceeded max time ({MAX_PROCESS_TIME}s), exiting"
                )
                return

        # Stream the response
        try:
            async for chunk in simulator.generate_streaming_response(
                message_content, chat_history
            ):
                # Check if streaming is taking too long
                if time.time() - start_time > MAX_PROCESS_TIME:
                    logger.warning(
                        f"Streaming exceeded max time ({MAX_PROCESS_TIME}s), stopping"
                    )
                    break

                # Check if sending was successful, break if not
                result = await send_chunk({"type": "md", "content": chunk})

                if result is False:
                    logger.warning(
                        "Client disconnected during streaming, stopping response generation"
                    )
                    return

                # Add a small sleep to improve cancellation responsiveness
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            # If cancelled during the streaming loop, propagate
            logger.info(f"Task cancelled during streaming for block_ids: {block_ids}")
            raise

        # Send completion message if we made it through
        try:
            await send_chunk({"type": "status", "status": "complete"})
        except Exception as e:
            logger.warning(f"Failed to send completion status: {e}")

    except asyncio.CancelledError:
        # Handle cancellation gracefully
        logger.info(f"Task cancelled for block_ids: {event.get('block_ids', '')}")
        # Don't send any more messages
        raise

    except Exception as e:
        logger.error(f"Error in process_message: {str(e)}")
        # Try to send error status if possible
        try:
            await send_chunk({"type": "status", "status": "error"})
        except:
            pass
