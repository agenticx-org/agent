import logging
import os
import uuid
from typing import Any, Dict, Iterator, List, Tuple

from anthropic import Anthropic

logger = logging.getLogger("CLI_Agent")


class LLMInteraction:
    """Handles communication with the Anthropic API."""

    def __init__(self, api_key=None, model_id=None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if self.api_key:
            logger.info("Initializing Anthropic API client.")
            self.client = Anthropic(api_key=self.api_key)
        else:
            raise ValueError("No API credentials configured for Anthropic.")

        self.model_id = model_id or os.getenv("MODEL_ID", "claude-3-7-sonnet-latest")
        logger.info(f"Using model: {self.model_id}")

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: List[Dict[str, Any]],
    ) -> Any:
        """Sends messages to the Anthropic API and gets the response."""
        logger.info(
            f"Sending request to {self.model_id} with {len(messages)} messages and {len(tools)} tools."
        )
        try:
            response = self.client.messages.create(
                model=self.model_id,
                system=system_prompt,
                messages=messages,
                tools=tools,
                tool_choice={"type": "auto"},
                max_tokens=4096,
                temperature=0.1,
            )
            logger.info(f"Received response. Stop reason: {response.stop_reason}")
            return response
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}", exc_info=True)
            return None

    def generate_response_stream(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: List[Dict[str, Any]],
    ) -> Iterator[Dict[str, Any]]:
        """Sends messages to the Anthropic API and streams the response, yielding chunks.
        The generator finally returns a tuple: (llm_call_id, final_message_object).
        """
        llm_call_id = str(uuid.uuid4())
        logger.info(
            f"Sending streaming request to {self.model_id} (ID: {llm_call_id}) with {len(messages)} messages and {len(tools)} tools."
        )
        final_message = None
        try:
            with self.client.messages.stream(
                model=self.model_id,
                system=system_prompt,
                messages=messages,
                tools=tools,
                tool_choice={"type": "auto"},
                max_tokens=4096,
                temperature=0.1,  # Adjust temperature as needed
            ) as stream:
                for event in stream:
                    # Yield text chunks as they arrive
                    if (
                        event.type == "content_block_delta"
                        and event.delta.type == "text_delta"
                    ):
                        yield {
                            "id": llm_call_id,
                            "type": "llm_chunk",
                            "content": event.delta.text,
                        }
                    # Handle other event types if needed (e.g., tool_use start/delta/stop)
                    # For now, we focus on text streaming and getting the final message.

            # After the stream context manager exits, get the final message
            final_message = stream.get_final_message()
            logger.info(
                f"Stream finished (ID: {llm_call_id}). Stop reason: {final_message.stop_reason}"
            )

        except Exception as e:
            logger.error(
                f"Anthropic API stream (ID: {llm_call_id}) failed: {e}", exc_info=True
            )
            # Yield an error chunk to inform the consumer
            yield {
                "id": llm_call_id,
                "type": "error",
                "content": f"LLM stream failed: {e}",
            }
            # Ensure the return value indicates failure
            return llm_call_id, None

        # Return the ID and the complete message object via the generator's return value
        return llm_call_id, final_message
