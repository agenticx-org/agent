import json
import os

from anthropic import Anthropic
from firecrawl import FirecrawlApp


def crawl_website(url: str, limit: int = None) -> str:
    """Crawls a website using Firecrawl.

    Args:
        url (str): The URL to crawl
        limit (int, optional): The maximum number of pages to crawl

    Returns:
        str: JSON string containing the crawl results
    """
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        return json.dumps(
            {"error": "FIRECRAWL_API_KEY not set in environment variables"}
        )

    app = FirecrawlApp(api_key=api_key)
    params = {}
    if limit:
        params["limit"] = limit

    try:
        crawl_result = app.crawl_url(url, params=params, poll_interval=30)
        return json.dumps(crawl_result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def main():
    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    # Initialize conversation history
    conversation_history = []

    # Add the initial message to history
    user_message = {
        "role": "user",
        "content": "Crawl the website, https://docs.anthropic.com/en/docs/about-claude/models/all-models",
    }
    conversation_history.append(user_message)

    message = client.messages.create(
        max_tokens=1024,
        messages=conversation_history,
        tools=[
            {
                "name": "crawl_website",
                "description": "Use this function to crawl a website",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to crawl",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "The maximum number of pages to crawl",
                            "optional": True,
                        },
                    },
                    "required": ["url"],
                },
            }
        ],
        model="claude-3-7-sonnet-latest",
    )

    # Add assistant's response to history
    assistant_message = {
        "role": "assistant",
        "content": [
            content.text for content in message.content if content.type == "text"
        ][0],
    }
    conversation_history.append(assistant_message)

    print("Full message response:", message)
    print("\nContent blocks:")

    for content in message.content:
        print(f"\nProcessing content block type: {content.type}")

        if content.type == "text":
            print("Text content:", content.text)
        elif content.type == "tool_use":
            print(f"Tool use found: {content.name}")
            print(f"Parameters: {content.input}")
            if content.name == "crawl_website":
                try:
                    result = crawl_website(**content.input)
                    print(f"\nCrawl results:\n{result}")

                    # Add tool result to history
                    tool_result_message = {
                        "role": "tool",
                        "tool_name": "crawl_website",
                        "content": result,
                    }
                    conversation_history.append(tool_result_message)
                except Exception as e:
                    print(f"Error executing crawl: {str(e)}")

    print("\nConversation History:")
    for msg in conversation_history:
        print(f"\n{msg['role'].upper()}: {msg['content']}")


if __name__ == "__main__":
    main()
