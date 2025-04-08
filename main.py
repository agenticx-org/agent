import json
import os
from pathlib import Path
from textwrap import dedent

from anthropic import Anthropic
from firecrawl import FirecrawlApp

import agent_prompt
import agent_rules
import system_prompt
from editor_tool import EditorTool
from shell_tool import ShellTool


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
        crawl_result = app.scrape_url(url, params={'formats': ['markdown']})
        return json.dumps(crawl_result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def idle() -> str:
    """Signals that the agent has completed its task.

    Returns:
        str: A confirmation message
    """
    return json.dumps({"status": "Task completed successfully"})


def main():
    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    # Initialize conversation history and editor tool
    conversation_history = []
    max_iterations = 10  # Maximum number of iterations
    current_iteration = 0
    task_completed = False

    # Initialize editor with workspace root path
    workspace_root = Path(os.getcwd())
    editor = EditorTool(root_path=workspace_root)
    shell = ShellTool(root_path=workspace_root)

    # Add the initial message to history
    # user_message = {
    #     "role": "user",
    #     "content": agent_prompt.agent_prompt + agent_rules.agent_rules
    #     + "Crawl the website, https://docs.anthropic.com/en/docs/about-claude/models/all-models, and give me a summary, and save it to 'summary.md'",
    # }
    user_message = {
        "role": "user",
        "content": agent_prompt.agent_prompt + agent_rules.agent_rules
        + "Crawl the website, https://docs.anthropic.com/en/docs/about-claude/models/all-models, and give me a summary, write a python script called process.py to process the crawl results and output a file called 'summary.md', and save it to 'process.py', then execute the script using shell_tool",
    }
    conversation_history.append(user_message)

    while not task_completed and current_iteration < max_iterations:
        current_iteration += 1
        # print(f"\n=== Iteration {current_iteration} ===")

        message = client.messages.create(
            max_tokens=8096,
            messages=conversation_history,
            system=system_prompt.system_prompt,
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
                        },
                        "required": ["url"],
                    },
                },
                editor.to_params(),
                {
                    "name": "idle",
                    "description": "Call this function when the current task is complete",
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            ] + shell.to_params(),
            model="claude-3-5-sonnet-latest",
        )

        # Add assistant's response to history
        text_contents = [
            content.text.rstrip() for content in message.content if content.type == "text"
        ]
        assistant_message = {
            "role": "assistant",
            "content": text_contents[0] if text_contents else "",
        }
        conversation_history.append(assistant_message)

        # print("Full message response:", message)
        # print("\nContent blocks:")
        print(message.content)

        # Process each content block
        for content in message.content:
            # print(f"\nProcessing content block type: {content.type}")

            if content.type == "text":
                # print("Text content:", content.text)
                pass
            elif content.type == "tool_use":
                # print(f"Tool use found: {content.name}")
                # print(f"Parameters: {content.input}")

                if content.name == "crawl_website":
                    try:
                        result = crawl_website(**content.input)
                        # print(f"\nCrawl results:\n{result}")

                        # Add tool result as part of assistant's next message
                        user_message = {
                            "role": "user",
                            "content": f"Tool '{content.name}' returned: {result}",
                        }
                        conversation_history.append(user_message)
                    except Exception as e:
                        # print(f"Error executing crawl: {str(e)}")
                        user_message = {
                            "role": "user",
                            "content": f"Error executing {content.name}: {str(e)}",
                        }
                        conversation_history.append(user_message)
                elif content.name == editor.name:  # Use the editor's name constant
                    try:
                        result = editor(**content.input)
                        print(f"\nEditor results:\n{result}")
                        user_message = {
                            "role": "user",
                            "content": f"Tool '{content.name}' returned: {result}",
                        }
                        conversation_history.append(user_message)
                    except Exception as e:
                        # print(f"Error executing editor: {str(e)}")
                        user_message = {
                            "role": "user",
                            "content": f"Error executing {content.name}: {str(e)}",
                        }
                        conversation_history.append(user_message)
                elif content.name == "shell_tool":
                    try:
                        result = shell(**content.input)
                        print(f"\nShell results:\n{result}")
                        user_message = {
                            "role": "user",
                            "content": f"Tool '{content.name}' returned: {result}",
                        }
                        conversation_history.append(user_message)
                    except Exception as e:
                        # print(f"Error executing shell: {str(e)}")
                        user_message = {
                            "role": "user",
                            "content": f"Error executing {content.name}: {str(e)}",
                        }
                        conversation_history.append(user_message)
                elif content.name == "idle":
                    result = idle()
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
