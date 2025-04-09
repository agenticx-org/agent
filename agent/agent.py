import importlib
import json
import logging
import sys
import time
from typing import Any, Dict, Optional

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
        self._output_json({"type": "plan", "content": self.state_manager.get_plan()})

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

            # 2. Call LLM
            self._output_json({"type": "status", "content": "Calling LLM..."})
            response_message = self.llm.generate_response(
                messages, system_prompt, tool_definitions
            )

            if response_message is None or response_message.content is None:
                self._output_json(
                    {
                        "type": "error",
                        "content": "LLM interaction failed or returned empty content. Terminating.",
                    }
                )
                break

            # Add assistant's response message to history before processing
            self.state_manager.add_assistant_message(response_message.content)

            # 3. Process LLM response blocks
            executed_tool_this_turn = False
            for block in response_message.content:
                if block.type == "text":
                    self._output_json({"type": "thought", "content": block.text})

                elif block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id
                    executed_tool_this_turn = True

                    # Log the tool call
                    self._output_json(
                        {"type": "tool_call", "tool": tool_name, "args": tool_input}
                    )

                    # Special handling for code execution
                    if tool_name == "execute_python":
                        self._output_json(
                            {"type": "code", "content": tool_input.get("code", "")}
                        )

                    self._output_json(
                        {"type": "status", "content": f"Executing tool: {tool_name}..."}
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

                    # Format tool result as JSON
                    result_output = {
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
                    self.state_manager.add_tool_result(
                        tool_use_id=tool_use_id,
                        result=result_content_for_llm,  # Send stringified/error detail to LLM
                        is_error=is_error,
                    )

            if (
                not executed_tool_this_turn
                and response_message.stop_reason == "stop_sequence"
            ):
                self._output_json(
                    {
                        "type": "warning",
                        "content": "LLM finished turn without using a tool. Task may be stalled.",
                    }
                )

            self._output_json({"type": "iteration_end", "iteration": iterations})
            # Optional delay
            time.sleep(0.5)

        # Loop finished
        self._output_json({"type": "execution_complete"})
        if self.state_manager.check_done():
            final_answer = self.state_manager.get_final_answer()
            self._output_json({"type": "final_answer", "content": final_answer})
            self._output_json(
                {"type": "status", "content": "Task completed successfully."}
            )
        elif iterations >= max_iterations:
            self._output_json(
                {"type": "error", "content": "Agent reached maximum iterations."}
            )
            self._output_json(
                {"type": "status", "content": "Task incomplete (max iterations)."}
            )
        else:
            self._output_json(
                {"type": "warning", "content": "Agent loop exited unexpectedly."}
            )
            self._output_json({"type": "status", "content": "Task incomplete."})

        return self.state_manager.get_final_answer()
