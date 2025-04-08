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


def terminate() -> str:
    """Signals that the agent has completed its task.

    Returns:
        str: A confirmation message
    """
    return json.dumps({"status": "Task completed successfully"})


def main():
    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    # Initialize conversation history
    conversation_history = []
    max_iterations = 10  # Maximum number of iterations
    current_iteration = 0
    task_completed = False

    # Add the initial message to history
    user_message = {
        "role": "user",
        "content": "Crawl the website, https://docs.anthropic.com/en/docs/about-claude/models/all-models, and give me a summary.",
    }
    conversation_history.append(user_message)

    while not task_completed and current_iteration < max_iterations:
        current_iteration += 1
        print(f"\n=== Iteration {current_iteration} ===")

        message = client.messages.create(
            max_tokens=8096,
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
                },
                {
                    "name": "terminate",
                    "description": "Call this function when the task is complete to end the agent loop",
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
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

        # Process each content block
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

                        # Add tool result as part of assistant's next message
                        user_message = {
                            "role": "user",
                            "content": f"Tool '{content.name}' returned: {result}",
                        }
                        conversation_history.append(user_message)
                    except Exception as e:
                        print(f"Error executing crawl: {str(e)}")
                        user_message = {
                            "role": "user",
                            "content": f"Error executing {content.name}: {str(e)}",
                        }
                        conversation_history.append(user_message)

                elif content.name == "terminate":
                    result = terminate()
                    print("\nTerminating agent loop...")
                    task_completed = True
                    user_message = {
                        "role": "user",
                        "content": f"Tool '{content.name}' returned: {result}",
                    }
                    conversation_history.append(user_message)

        if current_iteration >= max_iterations:
            print("\nReached maximum number of iterations without completing the task.")

    print("\nFinal Conversation History:")
    for msg in conversation_history:
        print(f"\n{msg['role'].upper()}: {msg['content']}")


if __name__ == "__main__":
    main()
