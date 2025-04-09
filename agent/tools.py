import inspect
import json
import logging
import random
from typing import TYPE_CHECKING, Any, Callable, Dict, List

logger = logging.getLogger("CLI_Agent")

# Import for type checking only to avoid circular imports
if TYPE_CHECKING:
    from agent.code_executor import CodeExecutor
    from agent.state_manager import StateManager

# --- Tool Implementations ---


# --- Built-in Tool Implementations ---
def execute_python_impl(
    code_executor: "CodeExecutor", state_manager: "StateManager", code: str
) -> Dict[str, Any]:
    """
    Implementation for the 'execute_python' tool.
    Executes the code using CodeExecutor and updates the StateManager's globals.
    """
    logger.info("Executing python code via tool.")
    result = code_executor.execute(code)
    if result.get("updated_globals"):
        state_manager.update_globals(result["updated_globals"])
    return {
        "stdout": result["stdout"],
        "error": result["error"],  # Will be None if successful
    }


def update_plan_impl(state_manager: "StateManager", plan_markdown: str) -> str:
    """Implementation for the 'update_plan' tool."""
    logger.info("Updating plan via tool.")
    state_manager.update_plan(plan_markdown)
    return "Plan updated successfully."


def record_findings_impl(state_manager: "StateManager", findings_markdown: str) -> str:
    """Implementation for the 'record_findings' tool."""
    logger.info("Recording findings via tool.")
    state_manager.update_findings(findings_markdown)
    return "Findings recorded successfully."


def final_answer_impl(state_manager: "StateManager", result: Any) -> Any:
    """Implementation for the 'final_answer' tool."""
    logger.info(f"Final answer received via tool: {result}")
    state_manager.set_done(result)
    return result


# --- Custom Tool Implementations ---
def search(query: str) -> str:
    """
    Simulates a web search for a given query. Returns simulated results.
    """
    logger.info(f"Executing search tool with query: '{query}'")
    results = [
        f"Result 1 for '{query}': Details about the query.",
        f"Result 2 for '{query}': Related information link.",
        f"Result 3 for '{query}': A relevant fact.",
    ]
    if "unknown" in query.lower() or random.random() < 0.1:
        logger.warning(f"No results found for query: '{query}'")
        return f"No results found for query: '{query}'"
    return "\n".join(random.sample(results, k=random.randint(1, len(results))))


class ToolManager:
    """Manages tool definitions, schemas, and execution mapping."""

    def __init__(
        self,
        state_manager: "StateManager",
        code_executor: "CodeExecutor",
        allowed_imports: List[str],
    ):
        # No need to import here since we have TYPE_CHECKING imports at the module level
        self.state_manager = state_manager
        self.code_executor = code_executor
        self.allowed_imports = allowed_imports
        self._tools: Dict[str, Callable] = {}
        self._tool_implementations: Dict[str, Callable] = {}  # Store actual functions
        self._tool_definitions: List[Dict[str, Any]] = []
        self._load_tools()
        self._generate_tool_definitions()
        logger.info(
            f"ToolManager initialized with tools: {list(self._tool_implementations.keys())}"
        )

    def _load_tools(self):
        """Loads tool implementation functions from the current script scope."""
        # Map tool name to implementation function defined globally in this script
        self._tool_implementations["execute_python"] = execute_python_impl
        self._tool_implementations["update_plan"] = update_plan_impl
        self._tool_implementations["record_findings"] = record_findings_impl
        self._tool_implementations["final_answer"] = final_answer_impl
        self._tool_implementations["search"] = search  # Custom tool

    def _generate_tool_definitions(self):
        """Generates Anthropic-compatible tool definitions."""
        definitions = []
        # Built-in tools (manual schema definition)
        definitions.append(
            {
                "name": "execute_python",
                "description": f"Executes a snippet of Python code. State persists. Allowed imports: {', '.join(self.allowed_imports) or 'None'}. Use print() for output.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "The Python code."}
                    },
                    "required": ["code"],
                },
            }
        )
        definitions.append(
            {
                "name": "update_plan",
                "description": "Updates the agent's plan (Markdown checklist). Call at start of each reasoning step.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "plan_markdown": {
                            "type": "string",
                            "description": "Complete updated plan in Markdown.",
                        }
                    },
                    "required": ["plan_markdown"],
                },
            }
        )
        definitions.append(
            {
                "name": "record_findings",
                "description": "Records final findings/conclusions (Markdown). Call before final_answer.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "findings_markdown": {
                            "type": "string",
                            "description": "Summary of findings in Markdown.",
                        }
                    },
                    "required": ["findings_markdown"],
                },
            }
        )
        definitions.append(
            {
                "name": "final_answer",
                "description": "Provides the final answer to the user's task and concludes operation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "string", "description": "The final answer."}
                    },
                    "required": ["result"],
                },
            }
        )

        # Custom tools (generate schema from signature/docstring)
        for name, func in self._tool_implementations.items():
            if name in [
                "execute_python",
                "update_plan",
                "record_findings",
                "final_answer",
            ]:
                continue  # Skip built-ins already defined

            docstring = inspect.getdoc(func) or f"Executes the {name} tool."
            sig = inspect.signature(func)
            properties = {}
            required = []
            for param_name, param in sig.parameters.items():
                param_type = "string"  # Default
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"
                properties[param_name] = {
                    "type": param_type,
                    "description": f"Parameter '{param_name}'",
                }
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            definitions.append(
                {
                    "name": name,
                    "description": docstring.split("\n")[0],
                    "input_schema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                }
            )
            logger.debug(f"Generated definition for custom tool: {name}")

        self._tool_definitions = definitions

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return self._tool_definitions

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Finds and executes the appropriate tool implementation."""
        tool_function = self._tool_implementations.get(tool_name)
        if not tool_function:
            logger.error(f"Tool '{tool_name}' not found.")
            return f"Error: Tool '{tool_name}' not found."

        logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")

        try:
            # Inject dependencies for built-in tools requiring state/executor
            if tool_name == "execute_python":
                result = tool_function(
                    self.code_executor, self.state_manager, **tool_args
                )
            elif tool_name in ["update_plan", "record_findings", "final_answer"]:
                result = tool_function(self.state_manager, **tool_args)
            else:
                # Execute custom tools directly
                result = tool_function(**tool_args)

            return result  # Return the actual result object/value

        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
            error_msg = f"Error executing tool '{tool_name}': {type(e).__name__}: {e}"
            return error_msg  # Return error message string for the LLM

    def get_callable_tools_for_eval(self) -> Dict[str, Callable]:
        """Returns custom tools suitable for CodeExecutor's global scope."""
        eval_tools = {}
        for name, func in self._tool_implementations.items():
            # Exclude built-ins that manage agent state/execution
            if name not in [
                "execute_python",
                "update_plan",
                "record_findings",
                "final_answer",
            ]:
                eval_tools[name] = func
        return eval_tools
