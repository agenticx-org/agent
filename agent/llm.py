import os

import anthropic
from dotenv import load_dotenv

load_dotenv()


class LLM:
    def __init__(
        self,
        model_id="claude-3-7-sonnet-latest",
        api_key=None,
        **kwargs,
    ):
        """
        Initialize an LLM instance that uses the Anthropic SDK under the hood.

        Args:
            model_id (str, optional): The model ID to use. Defaults to "claude-3-7-sonnet-latest".
            api_key (str, optional): Anthropic API key. Defaults to None (will use ANTHROPIC_API_KEY env var).
            **kwargs: Additional parameters to pass to the Anthropic client
        """
        self.model_id = model_id
        # Use provided API key or fall back to environment variable
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.kwargs = kwargs

        # Filter out 'organization' from kwargs if present
        if "organization" in kwargs:
            kwargs_copy = kwargs.copy()
            kwargs_copy.pop("organization")
            self.client = anthropic.Anthropic(api_key=self.api_key, **kwargs_copy)
        else:
            self.client = anthropic.Anthropic(api_key=self.api_key, **kwargs)

    def generate(self, prompt, system_prompt=None, max_tokens=4096, temperature=0.7):
        """
        Generate a response from the LLM.

        Args:
            prompt (str): The user prompt to send to the LLM
            system_prompt (str, optional): System instructions for the LLM. Defaults to None.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 4096.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.

        Returns:
            dict: The response from the LLM
        """
        try:
            # Prepare parameters for the call
            params = {
                "model": self.model_id,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }

            # Add optional parameters if provided
            if system_prompt:
                params["system"] = system_prompt

            # Call Anthropic message creation
            response = self.client.messages.create(**params)

            return {
                "status": "success",
                "response": response.content[0].text,
                "raw_response": response,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def generate_stream(
        self, prompt, system_prompt=None, max_tokens=4096, temperature=0.7
    ):
        """
        Generate a streaming response from the LLM.

        Args:
            prompt (str): The user prompt to send to the LLM
            system_prompt (str, optional): System instructions for the LLM. Defaults to None.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 4096.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.

        Returns:
            generator: A generator that yields chunks of the response text
        """
        try:
            # Prepare parameters for the call
            params = {
                "model": self.model_id,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "max_tokens": max_tokens,
            }

            # Add optional parameters if provided
            if system_prompt:
                params["system"] = system_prompt

            # Call Anthropic message creation with streaming
            stream = self.client.messages.create(**params)

            # Process the streaming response
            for chunk in stream:
                if len(chunk.delta.text) > 0:
                    yield {
                        "status": "success",
                        "chunk": chunk.delta.text,
                        "raw_chunk": chunk,
                    }
        except Exception as e:
            yield {"status": "error", "message": str(e)}
