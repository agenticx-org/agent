import importlib
import json
import logging
import sys
import time
from typing import Any, Dict, Iterator, Optional

from agent.code_executor import CodeExecutor
from agent.llm import LLMInteraction
from agent.prompts import get_system_prompt
from agent.state_manager import StateManager
from agent.tools import ToolManager

logger = logging.getLogger("CLI_Agent")


class Agent:
    """Orchestrates the agent's lifecycle for CLI interaction."""

    def __init__(self, task: str, authorized_imports=None, model_id=None):
        self.task = task
        self.authorized_imports = authorized_imports or []
        self._output_json(
            {"type": "status", "content": f"Initializing Agent for task: {self.task}"}
        )

        # Initialize components
        try:
            self.llm = LLMInteraction(model_id=model_id)
        except ValueError as e:
            self._output_json(
                {"type": "error", "content": f"Failed to initialize LLM: {e}"}
            )
            sys.exit(1)

        # Init StateManager with temporary prompt
        self.state_manager = StateManager(
            initial_task=task, system_prompt="Initializing..."
        )

        # Init CodeExecutor with empty dict first
        self.code_executor = CodeExecutor(initial_globals={})

        # Init ToolManager, which needs state and code executor
        self.tool_manager = ToolManager(
            state_manager=self.state_manager,
            code_executor=self.code_executor,
            allowed_imports=self.authorized_imports,
        )

        # Generate the real system prompt
        tool_definitions = self.tool_manager.get_tool_definitions()
        system_prompt = get_system_prompt(tool_definitions, self.authorized_imports)
        self.state_manager.system_prompt = system_prompt  # Update state

        # Prepare and set initial globals for code execution
        initial_globals = self._prepare_initial_globals()
        self.state_manager.set_initial_globals(initial_globals)  # Set in state
        self.code_executor.globals_locals = initial_globals  # Update executor directly

        self._output_json(
            {"type": "status", "content": "Agent initialized successfully."}
        )
        self._output_json(
            {"id": "initial", "type": "plan", "content": self.state_manager.get_plan()}
        )

    def _prepare_initial_globals(self) -> Dict[str, Any]:
        """Prepares the initial global scope for the CodeExecutor."""
        initial_globals = {}
        # Import allowed modules
        for import_name in self.authorized_imports:
            try:
                module = importlib.import_module(import_name)
                initial_globals[import_name] = module
                logger.info(f"Made module '{import_name}' available to code execution.")
            except ImportError:
                logger.warning(
                    f"Could not import module '{import_name}' for code execution."
                )

        # Add callable *custom* tools
        callable_tools = self.tool_manager.get_callable_tools_for_eval()
        initial_globals.update(callable_tools)
        logger.info(
            f"Made tools {list(callable_tools.keys())} available to code execution."
        )

        return initial_globals

    def _output_json(self, data: Dict[str, Any]) -> None:
        """Prints data in a uniform JSON format."""
        print(json.dumps(data))

    def run(self, max_iterations=15):
        """Runs the agent's main loop for CLI."""
        self._output_json({"type": "status", "content": "Agent starting run loop..."})
        iterations = 0

        while not self.state_manager.check_done() and iterations < max_iterations:
            iterations += 1
            self._output_json({"type": "iteration_start", "iteration": iterations})
            self._output_json({"type": "status", "content": "Preparing LLM request..."})

            # 1. Get state for LLM
            messages = self.state_manager.get_history()
            system_prompt = self.state_manager.get_system_prompt()
            tool_definitions = self.tool_manager.get_tool_definitions()

            # 2. Call LLM (Streaming)
            self._output_json({"type": "status", "content": "Calling LLM..."})
            llm_call_id = None
            final_message = None
            stream_generator = self.llm.generate_response_stream(
                messages, system_prompt, tool_definitions
            )

            # Consume the stream and output chunks
            try:
                while True:
                    chunk = next(stream_generator)
                    self._output_json(chunk)  # Output each chunk (llm_chunk or error)
                    # Keep track of the ID from the first chunk
                    if llm_call_id is None and "id" in chunk:
                        llm_call_id = chunk["id"]

            except StopIteration as e:
                # The generator is exhausted, capture the return value
                # The return value is a tuple (llm_call_id, final_message_object)
                if e.value:
                    llm_call_id_from_return, final_message = e.value
                    # Ensure we have the ID, either from chunks or return value
                    if llm_call_id is None:
                        llm_call_id = llm_call_id_from_return
                else:
                    # Handle case where StopIteration has no value (e.g., an error occurred before yielding anything)
                    # If llm_call_id wasn't set from an error chunk, generate one for consistency? Or rely on logger?
                    # For now, assume an error chunk was yielded if e.value is None
                    if llm_call_id is None:
                        llm_call_id = "error-no-id"  # Fallback ID
                    final_message = None  # Ensure final_message is None on stream error

            # Add LLM call ID to status message for clarity
            self._output_json(
                {"id": llm_call_id, "type": "status", "content": "LLM stream finished."}
            )

            # Check if the stream failed or returned empty content
            if final_message is None or final_message.content is None:
                self._output_json(
                    {
                        "id": llm_call_id,  # Add ID
                        "type": "error",
                        "content": "LLM stream failed or returned empty content. Terminating.",
                    }
                )
                break

            # Add assistant's *complete* response message to history before processing
            # Important: Use final_message here, which is the complete Anthropic message object
            self.state_manager.add_assistant_message(final_message.content)

            # 3. Process LLM response blocks from the final message
            executed_tool_this_turn = False
            for block in final_message.content:
                if block.type == "text":
                    # We no longer output the full thought here, it was streamed as chunks
                    # self._output_json({"type": "thought", "content": block.text})
                    pass  # Thought content was already streamed

                elif block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id  # Keep the tool_use_id from Anthropic
                    executed_tool_this_turn = True

                    # Log the tool call with the LLM call ID
                    self._output_json(
                        {
                            "id": llm_call_id,
                            "type": "tool_call",
                            "tool": tool_name,
                            "args": tool_input,
                        }  # Add ID
                    )

                    # Special handling for code execution
                    if tool_name == "execute_python":
                        self._output_json(
                            {
                                "id": llm_call_id,
                                "type": "code",
                                "content": tool_input.get("code", ""),
                            }  # Add ID
                        )

                    self._output_json(
                        {
                            "id": llm_call_id,
                            "type": "status",
                            "content": f"Executing tool: {tool_name}...",
                        }  # Add ID
                    )

                    # Execute the tool
                    result = self.tool_manager.execute_tool(tool_name, tool_input)

                    # Determine if result indicates an error
                    is_error = False
                    result_content_for_llm = result
                    if isinstance(result, str) and result.lower().startswith("error:"):
                        is_error = True
                    # Check dict format from execute_python_impl
                    if isinstance(result, dict) and result.get("error"):
                        is_error = True
                        result_content_for_llm = f"Error during execution: {result['error']}"  # Pass error string to LLM

                    # Format tool result as JSON, including LLM call ID
                    result_output = {
                        "id": llm_call_id,  # Add ID
                        "type": "tool_result",
                        "tool": tool_name,
                        "success": not is_error,
                    }

                    # Format the result content based on its type
                    if isinstance(result, dict):
                        if "stdout" in result:
                            result_output["stdout"] = result.get("stdout")
                        if "error" in result:
                            result_output["error"] = result.get("error")
                    else:
                        result_output["content"] = str(result)

                    self._output_json(result_output)

                    # Add tool result message to state for the *next* LLM call
                    # Use the original tool_use_id provided by Anthropic
                    self.state_manager.add_tool_result(
                        tool_use_id=tool_use_id,
                        result=result_content_for_llm,  # Send stringified/error detail to LLM
                        is_error=is_error,
                    )

            if (
                not executed_tool_this_turn
                and final_message.stop_reason == "stop_sequence"  # Use final_message
            ):
                self._output_json(
                    {
                        "id": llm_call_id,  # Add ID
                        "type": "warning",
                        "content": "LLM finished turn without using a tool. Task may be stalled.",
                    }
                )

            self._output_json({"type": "iteration_end", "iteration": iterations})
            # Optional delay
            time.sleep(0.5)

        # Loop finished
        # Add ID 'final' to these terminal messages
        self._output_json({"id": "final", "type": "execution_complete"})
        if self.state_manager.check_done():
            final_answer = self.state_manager.get_final_answer()
            self._output_json(
                {"id": "final", "type": "final_answer", "content": final_answer}
            )
            self._output_json(
                {
                    "id": "final",
                    "type": "status",
                    "content": "Task completed successfully.",
                }
            )
        elif iterations >= max_iterations:
            self._output_json(
                {
                    "id": "final",
                    "type": "error",
                    "content": "Agent reached maximum iterations.",
                }
            )
            self._output_json(
                {
                    "id": "final",
                    "type": "status",
                    "content": "Task incomplete (max iterations).",
                }
            )
        else:
            self._output_json(
                {
                    "id": "final",
                    "type": "warning",
                    "content": "Agent loop exited unexpectedly.",
                }
            )
            self._output_json(
                {"id": "final", "type": "status", "content": "Task incomplete."}
            )

        return self.state_manager.get_final_answer()
