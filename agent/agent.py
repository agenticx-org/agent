from agent.llm import LLM
from agent.tools.registry import ToolRegistry


class Agent:
    def __init__(
        self,
        name,
        description,
        system_prompt,
        tool_registry=None,
        max_steps=10,
        model_id="claude-3-7-sonnet-latest",
        api_key=None,
        **llm_kwargs,
    ):
        """
        Initialize an Agent.

        Args:
            name (str): The name of the agent
            description (str): A description of what the agent does
            system_prompt (str): The system prompt to guide the agent's behavior
            tool_registry (ToolRegistry, optional): Registry of tools the agent can use. Defaults to None.
            max_steps (int, optional): Maximum number of execution steps. Defaults to 10.
            model_id (str, optional): The model ID to use. Defaults to "claude-3-7-sonnet-latest".
            api_key (str, optional): API key for the LLM provider. Defaults to None.
            **llm_kwargs: Additional parameters to pass to the LLM
        """
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_steps = max_steps
        self.step_count = 0

        # Initialize the LLM
        self.llm = LLM(model_id=model_id, api_key=api_key, **llm_kwargs)

    def run(self, input_text):
        """
        Run the agent on the given input.

        Args:
            input_text (str): The input text to process

        Returns:
            dict: The result of the agent's execution
        """
        if self.step_count >= self.max_steps:
            return {"status": "error", "message": "Maximum steps reached"}

        # Use the LLM to generate a response
        enhanced_prompt = self._create_prompt(input_text)
        llm_response = self.llm.generate(
            prompt=enhanced_prompt, system_prompt=self.system_prompt
        )

        self.step_count += 1

        if llm_response["status"] == "error":
            return llm_response

        return {
            "status": "success",
            "agent": self.name,
            "response": llm_response["response"],
        }

    def process_loop(self, input_text):
        """
        Run a complete process loop with the agent, handling tool calls until completion.

        Args:
            input_text (str): The initial input text to process

        Returns:
            dict: The final result of the agent's execution including full conversation history
        """
        # Initialize conversation history
        conversation = [{"role": "user", "content": input_text}]

        print(f"Starting process loop with initial prompt: {input_text}")
        turn_count = 0
        max_tool_turns = self.max_steps - 1  # Leave one turn for final response

        # Create enhanced system prompt for better tool handling
        enhanced_system_prompt = self._create_tool_using_system_prompt()

        # Flag to track if we're in the final summarization turn
        final_turn = False
        tool_used = False

        while turn_count < max_tool_turns:
            turn_count += 1
            self.step_count += 1

            # Create enhanced prompt from conversation history
            current_prompt = self._create_prompt_from_history(conversation)

            # For the final turn, use a different prompt to force summarization
            if final_turn:
                # Use a special system prompt to force summarization
                summary_system_prompt = self._create_summary_system_prompt()

                # Get response from LLM
                llm_response = self.llm.generate(
                    prompt=current_prompt, system_prompt=summary_system_prompt
                )
            else:
                # Get response from LLM with normal tool-using system prompt
                llm_response = self.llm.generate(
                    prompt=current_prompt, system_prompt=enhanced_system_prompt
                )

            if llm_response["status"] == "error":
                return {
                    "status": "error",
                    "message": llm_response["message"],
                    "conversation": conversation,
                }

            # Add assistant response to conversation
            assistant_message = {
                "role": "assistant",
                "content": llm_response["response"],
            }
            conversation.append(assistant_message)

            # If this is the final summarization turn, we're done
            if final_turn:
                print(f"Final response generated after {turn_count} turns.")
                break

            # Check for tool calls in the response
            tool_calls = self._extract_tool_calls(llm_response["response"])

            if not tool_calls:
                if tool_used:
                    # No more tool calls but we've used tools before,
                    # so do one more turn for final summarization
                    print(f"No more tool calls detected. Generating final summary...")
                    final_turn = True
                    continue
                else:
                    # No tool calls detected at all - we're done
                    print(
                        f"No tool calls detected. Process complete after {turn_count} turns."
                    )
                    break

            # Process each tool call
            tool_used = True  # Mark that we've used a tool
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]

                print(f"Executing tool: '{tool_name}' with args: {args}")

                try:
                    # Execute the tool using the tool registry
                    tool_result = self.tool_registry.execute_tool(tool_name, args)
                    tool_response = {
                        "role": "function",
                        "name": tool_name,
                        "content": tool_result,
                    }
                    conversation.append(tool_response)
                    print(f"Tool '{tool_name}' executed successfully.")
                except Exception as e:
                    error_msg = f"Tool execution failed: {type(e).__name__}: {str(e)}"
                    tool_response = {
                        "role": "function",
                        "name": tool_name,
                        "content": {"error": error_msg},
                    }
                    conversation.append(tool_response)
                    print(error_msg)

        if turn_count >= max_tool_turns and not final_turn:
            print(
                f"Maximum tool turns ({max_tool_turns}) reached. Generating final summary..."
            )

            # Add one last turn to generate a summary
            current_prompt = self._create_prompt_from_history(conversation)
            summary_system_prompt = self._create_summary_system_prompt()

            # Get final summary response from LLM
            llm_response = self.llm.generate(
                prompt=current_prompt, system_prompt=summary_system_prompt
            )

            if llm_response["status"] == "success":
                # Add final assistant response to conversation
                assistant_message = {
                    "role": "assistant",
                    "content": llm_response["response"],
                }
                conversation.append(assistant_message)
                print(f"Final response generated after reaching max turns.")

        # Return final result with conversation history
        return {
            "status": "success",
            "agent": self.name,
            "final_response": (
                conversation[-1]["content"]
                if conversation[-1]["role"] == "assistant"
                else None
            ),
            "conversation": conversation,
            "turns": turn_count,
        }

    def _create_summary_system_prompt(self):
        """
        Create a system prompt that instructs the LLM to provide a final summary response.

        Returns:
            str: The summary system prompt
        """
        summary_instructions = f"""
{self.system_prompt}

You have collected information using various tools. Now, STOP using any more tools and provide a comprehensive 
final response to the user's original request. Summarize what you found using the tools and answer their questions directly.

DO NOT suggest using more tools. DO NOT call any more tools.
JUST provide a final, helpful response using the information you've already gathered.
"""
        return summary_instructions

    def _create_tool_using_system_prompt(self):
        """
        Create an enhanced system prompt that instructs the LLM on how to use tools.

        Returns:
            str: The enhanced system prompt
        """
        tools_description = ""

        if self.tool_registry.list_tools():
            tools_description = "You have access to the following tools:\n\n"

            for tool in self.tool_registry.list_tools():
                tools_description += f"Tool name: {tool.name}\n"
                tools_description += f"Description: {tool.description}\n"

                if tool.parameters:
                    tools_description += "Parameters:\n"
                    for param_name, param_details in tool.parameters.items():
                        param_type = param_details.get("type", "any")
                        param_desc = param_details.get("description", "")
                        required = param_details.get("required", False)
                        tools_description += f"  - {param_name} ({param_type}): {param_desc}{' (required)' if required else ''}\n"

                tools_description += "\n"

        tool_usage_instructions = """
When you need to use a tool, format your response in one of these ways:

Format 1:
I need to [explain reasoning]
tool_name({"param1": "value1", "param2": "value2"})

Format 2:
Function to call: tool_name
Arguments: {"param1": "value1", "param2": "value2"}

For example:
I need to search for apartments in Paris.
airbnb_search({"location": "Paris", "checkin": "2024-03-28", "checkout": "2024-03-30"})

Wait for the tool's response before continuing. After receiving the tool's response, provide a helpful answer to the user based on that information.
"""

        # Combine the original system prompt with tool instructions
        enhanced_prompt = (
            f"{self.system_prompt}\n\n{tools_description}\n{tool_usage_instructions}"
        )
        return enhanced_prompt

    def _extract_tool_calls(self, response_text):
        """
        Extract tool calls from the LLM response text.
        This implementation looks for tool calling patterns in the response.

        Args:
            response_text (str): The response text from the LLM

        Returns:
            list: A list of dictionaries containing tool calls with 'name' and 'args' keys
        """
        import json
        import re

        tool_calls = []

        # Get list of available tool names
        tool_names = [tool.name for tool in self.tool_registry.list_tools()]

        # Check if any of the available tools are mentioned
        for tool_name in tool_names:
            # Pattern 1: Look for function call format like: function_name({"param": "value"})
            pattern1 = rf"{tool_name}\s*\(\s*(\{{.*?\}})\s*\)"
            matches1 = re.findall(pattern1, response_text, re.DOTALL)

            for match in matches1:
                try:
                    args = json.loads(match)
                    tool_calls.append({"name": tool_name, "args": args})
                    # If we found a valid call, we can break from the loop for this tool
                    break
                except json.JSONDecodeError:
                    pass  # Not valid JSON, continue to next pattern

            # If we already found a valid call for this tool, move to the next tool
            if any(call["name"] == tool_name for call in tool_calls):
                continue

            # Pattern 2: Look for 'Function to call: function_name' format
            pattern2 = rf"Function to call:\s*{tool_name}.*?Arguments:\s*(\{{.*?\}})"
            matches2 = re.findall(pattern2, response_text, re.DOTALL)

            for match in matches2:
                try:
                    args = json.loads(match)
                    tool_calls.append({"name": tool_name, "args": args})
                    break
                except json.JSONDecodeError:
                    pass

            # If we already found a valid call for this tool, move to the next tool
            if any(call["name"] == tool_name for call in tool_calls):
                continue

            # Pattern 3: Look for key-value pairs in text format
            if not any(call["name"] == tool_name for call in tool_calls):
                # Example: "Using tool_name with location=Paris, date=2023-01-01"
                kv_pattern = rf'{tool_name}.*?(\w+)\s*=\s*["\']?([\w\s\.-]+)["\']?'
                kv_matches = re.findall(kv_pattern, response_text)

                if kv_matches:
                    args = {}
                    for key, value in kv_matches:
                        args[key] = value.strip()

                    if args:  # Only add if we found at least one argument
                        tool_calls.append({"name": tool_name, "args": args})

        return tool_calls

    def _create_prompt_from_history(self, conversation):
        """
        Create a prompt from the conversation history.

        Args:
            conversation (list): List of conversation messages

        Returns:
            str: The prompt to send to the LLM
        """
        prompt = ""
        for message in conversation:
            role = message["role"]
            content = message["content"]

            if role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
            elif role == "function":
                # Function responses might be dictionaries
                if isinstance(content, dict):
                    if "result" in content:
                        prompt += f"Tool ({message['name']}): {content['result']}\n\n"
                    elif "error" in content:
                        prompt += (
                            f"Tool ({message['name']}) Error: {content['error']}\n\n"
                        )
                else:
                    prompt += f"Tool ({message['name']}): {content}\n\n"

        # Add tools information
        tools = self.tool_registry.list_tools()
        if tools:
            prompt += "\nAvailable tools:\n"
            for tool in tools:
                prompt += f"- {tool.name}: {tool.description}\n"

        return prompt.strip()

    def _create_prompt(self, input_text):
        """
        Create a prompt for the LLM that includes available tools and context.

        Args:
            input_text (str): The user input

        Returns:
            str: The enhanced prompt
        """
        tools = self.tool_registry.list_tools()
        if not tools:
            return input_text

        # Create a description of available tools
        tools_description = "Available tools:\n"
        for tool in tools:
            tools_description += f"- {tool.name}: {tool.description}\n"

        # Combine everything into a single prompt
        prompt = f"{tools_description}\n\nUser input: {input_text}\n\nPlease respond to the user input, using tools where appropriate."
        return prompt

    def reset(self):
        """Reset the agent's state."""
        self.step_count = 0

    def add_tool(self, tool):
        """
        Add a tool to the agent's tool registry.

        Args:
            tool: The tool to add

        Returns:
            Agent: The agent (for method chaining)
        """
        self.tool_registry.register_tool(tool)
        return self
