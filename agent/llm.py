import logging
import os
from typing import Any, Dict, List

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
