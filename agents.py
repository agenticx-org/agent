import os
import sys
import logging
import json
import inspect
import importlib
import io
import contextlib
import time
import argparse
import random # For example search tool
from typing import Dict, Any, List, Callable, Optional

from dotenv import load_dotenv
from anthropic import Anthropic # Use only the regular Anthropic client

# --- Configuration Loading ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CLI_Agent")

# --- Anthropic Configuration ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "claude-3-haiku-20240307")

# --- Agent Configuration ---
RAW_AUTHORIZED_IMPORTS = os.getenv("AUTHORIZED_IMPORTS", "math,random,datetime,json,re")
AUTHORIZED_IMPORTS = [imp.strip() for imp in RAW_AUTHORIZED_IMPORTS.split(',') if imp.strip()]

# --- Prompt Templates ---
PLAN_TEMPLATE = """
# Research Agent Progress Tracking

## Completed Milestones
- [ ] Initial Task Analysis

## Current Research Focus
### Step 1
- [ ] Define initial approach

*(Agent will update this structure)*
"""

FINDINGS_TEMPLATE = """
# Research Findings

## Summary
*(Agent should summarize key findings here)*

## Details
*(Agent should list detailed results, observations, or data points)*

## Confidence Score
*(Agent should assess confidence, e.g., High/Medium/Low)*
"""

# --- Tool Implementations ---

# --- Built-in Tool Implementations ---
# These need access to specific agent components (CodeExecutor, StateManager)

def execute_python_impl(code_executor: 'CodeExecutor', state_manager: 'StateManager', code: str) -> Dict[str, Any]:
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
        "error": result["error"] # Will be None if successful
    }

def update_plan_impl(state_manager: 'StateManager', plan_markdown: str) -> str:
    """Implementation for the 'update_plan' tool."""
    logger.info("Updating plan via tool.")
    state_manager.update_plan(plan_markdown)
    return "Plan updated successfully."

def record_findings_impl(state_manager: 'StateManager', findings_markdown: str) -> str:
    """Implementation for the 'record_findings' tool."""
    logger.info("Recording findings via tool.")
    state_manager.update_findings(findings_markdown)
    return "Findings recorded successfully."

def final_answer_impl(state_manager: 'StateManager', result: Any) -> Any:
    """Implementation for the 'final_answer' tool."""
    logger.info(f"Final answer received via tool: {result}")
    state_manager.set_done(result)
    return result

# --- Custom Tool Implementations ---
# These are standalone functions

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

# --- Core Classes ---

class LLMInteraction:
    """Handles communication with the Anthropic API."""
    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError("No API key configured for Anthropic. Please set ANTHROPIC_API_KEY in environment variables.")

        logger.info("Initializing Anthropic API client.")
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model_id = MODEL_ID
        logger.info(f"Using model: {self.model_id}")

    def generate_response(self, messages: List[Dict[str, Any]], system_prompt: str, tools: List[Dict[str, Any]]) -> Any:
        """Sends messages to the Anthropic API and gets the response."""
        logger.info(f"Sending request to {self.model_id} with {len(messages)} messages and {len(tools)} tools.")
        # logger.debug(f"Message History (last 2): {json.dumps(messages[-2:], indent=2)}") # Log snippet
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
            # logger.debug(f"Response content types: {[block.type for block in response.content] if response.content else 'None'}")
            return response
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}", exc_info=True)
            return None

class CodeExecutor:
    """Executes Python code snippets within a controlled, persistent context."""
    def __init__(self, initial_globals: Dict[str, Any]):
        self.globals_locals: Dict[str, Any] = initial_globals.copy()
        logger.info(f"CodeExecutor initialized with globals: {list(self.globals_locals.keys())}")

    def execute(self, code_string: str) -> Dict[str, Any]:
        """Executes Python code, captures stdout/errors, updates scope."""
        logger.info(f"Executing code:\n---\n{code_string}\n---")
        stdout_capture = io.StringIO()
        error_message = None
        globals_before = self.globals_locals.copy()

        try:
            with contextlib.redirect_stdout(stdout_capture):
                compiled_code = compile(code_string, '<string>', 'exec')
                exec(compiled_code, self.globals_locals)
        except Exception as e:
            error_message = f"{type(e).__name__}: {e}"
            logger.error(f"Code execution failed: {error_message}")

        stdout_result = stdout_capture.getvalue()
        if stdout_result:
             logger.info(f"Execution stdout:\n{stdout_result}")

        updated_globals = {
            k: v for k, v in self.globals_locals.items()
            if k not in globals_before or globals_before[k] is not v
        }
        if '__builtins__' in updated_globals and updated_globals['__builtins__'] == globals_before.get('__builtins__'):
             del updated_globals['__builtins__']
        # logger.debug(f"Globals updated by execution: {list(updated_globals.keys())}")

        return {
            "stdout": stdout_result,
            "error": error_message,
            "updated_globals": updated_globals
        }

    def get_current_globals(self) -> Dict[str, Any]:
        return self.globals_locals.copy()

class StateManager:
    """Manages the agent's state: history, plan, findings, execution scope."""
    def __init__(self, initial_task: str, system_prompt: str):
        self.message_history: List[Dict[str, Any]] = [
            {"role": "user", "content": initial_task}
        ]
        self.system_prompt = system_prompt
        self.plan: str = PLAN_TEMPLATE
        self.findings: str = FINDINGS_TEMPLATE
        self.execution_globals: Dict[str, Any] = {}
        self._is_done: bool = False
        self.final_answer: Optional[Any] = None

    def add_message(self, role: str, content: Any):
        """Adds a message (or list of content blocks) to the history."""
        if not content:
             logger.warning(f"Attempted to add empty message for role {role}")
             return
        # Ensure content is list for assistant, handle tool results correctly
        if role == "assistant":
             if not isinstance(content, list):
                 content = [{"type": "text", "text": str(content)}]
        elif role == "user":
             # Handle tool result additions specifically via add_tool_result
             if isinstance(content, list) and content and content[0].get("type") == "tool_result":
                 # Already formatted correctly
                 pass
             elif not isinstance(content, list):
                  content = [{"type": "text", "text": str(content)}] # Simple user text

        self.message_history.append({"role": role, "content": content})

    def add_assistant_message(self, content_blocks: List[Dict[str, Any]]):
        """Adds the assistant's response (potentially multiple blocks) to history."""
        if content_blocks:
             self.add_message(role="assistant", content=content_blocks)

    def add_tool_result(self, tool_use_id: str, result: Any, is_error: bool = False):
         """Adds a tool result message linked to a tool_use request."""
         content_block = {
             "type": "tool_result",
             "tool_use_id": tool_use_id,
             "is_error": is_error,
         }
         # Content must be string or list of blocks (e.g. text block) for Anthropic
         if isinstance(result, (dict, list)):
             # Convert complex results to string (JSON) for the LLM
             content_block["content"] = json.dumps(result, indent=2)
         else:
             content_block["content"] = str(result)

         # Add as a user message containing the single tool result block
         self.add_message(role="user", content=[content_block])

    def get_history(self) -> List[Dict[str, Any]]:
        return self.message_history

    def get_system_prompt(self) -> str:
        return self.system_prompt

    def update_plan(self, plan_markdown: str):
        self.plan = plan_markdown
        logger.info("Plan updated.")
        # Print update to CLI
        print("\n--- PLAN UPDATED ---")
        print(self.plan)
        print("--------------------\n")


    def get_plan(self) -> str:
        return self.plan

    def update_findings(self, findings_markdown: str):
        self.findings = findings_markdown
        logger.info("Findings updated.")
        # Print update to CLI
        print("\n--- FINDINGS RECORDED ---")
        print(self.findings)
        print("-------------------------\n")

    def get_findings(self) -> str:
        return self.findings

    def update_globals(self, new_globals: Dict[str, Any]):
        self.execution_globals.update(new_globals)
        # logger.debug(f"Execution globals updated: {list(new_globals.keys())}")

    def get_globals(self) -> Dict[str, Any]:
        return self.execution_globals

    def set_initial_globals(self, initial_globals: Dict[str, Any]):
        self.execution_globals = initial_globals
        logger.info("Initial execution globals set.")

    def set_done(self, final_answer: Any):
        self._is_done = True
        self.final_answer = final_answer
        logger.info(f"Agent marked as done. Final Answer: {final_answer}")

    def check_done(self) -> bool:
        return self._is_done

    def get_final_answer(self) -> Optional[Any]:
        return self.final_answer

class ToolManager:
    """Manages tool definitions, schemas, and execution mapping."""
    def __init__(self, state_manager: 'StateManager', code_executor: 'CodeExecutor', allowed_imports: List[str]):
        self.state_manager = state_manager
        self.code_executor = code_executor
        self.allowed_imports = allowed_imports
        self._tools: Dict[str, Callable] = {}
        self._tool_implementations: Dict[str, Callable] = {} # Store actual functions
        self._tool_definitions: List[Dict[str, Any]] = []
        self._load_tools()
        self._generate_tool_definitions()
        logger.info(f"ToolManager initialized with tools: {list(self._tool_implementations.keys())}")

    def _load_tools(self):
        """Loads tool implementation functions from the current script scope."""
        # Map tool name to implementation function defined globally in this script
        self._tool_implementations["execute_python"] = execute_python_impl
        self._tool_implementations["update_plan"] = update_plan_impl
        self._tool_implementations["record_findings"] = record_findings_impl
        self._tool_implementations["final_answer"] = final_answer_impl
        self._tool_implementations["search"] = search # Custom tool

        # Add more custom tools here if defined globally
        # e.g., self._tool_implementations["get_stock_price"] = get_stock_price

    def _generate_tool_definitions(self):
        """Generates Anthropic-compatible tool definitions."""
        definitions = []
        # Built-in tools (manual schema definition)
        definitions.append({
            "name": "execute_python",
            "description": f"Executes a snippet of Python code. State persists. Allowed imports: {', '.join(self.allowed_imports) or 'None'}. Use print() for output.",
            "input_schema": {
                "type": "object", "properties": {"code": {"type": "string", "description": "The Python code."}}, "required": ["code"]
            }
        })
        definitions.append({
            "name": "update_plan",
            "description": "Updates the agent's plan (Markdown checklist). Call at start of each reasoning step.",
            "input_schema": {
                "type": "object", "properties": {"plan_markdown": {"type": "string", "description": "Complete updated plan in Markdown."}}, "required": ["plan_markdown"]
            }
        })
        definitions.append({
            "name": "record_findings",
            "description": "Records final findings/conclusions (Markdown). Call before final_answer.",
            "input_schema": {
                "type": "object", "properties": {"findings_markdown": {"type": "string", "description": "Summary of findings in Markdown."}}, "required": ["findings_markdown"]
            }
        })
        definitions.append({
            "name": "final_answer",
            "description": "Provides the final answer to the user's task and concludes operation.",
            "input_schema": {
                "type": "object", "properties": {"result": {"type": "string", "description": "The final answer."}}, "required": ["result"]
            }
        })

        # Custom tools (generate schema from signature/docstring)
        for name, func in self._tool_implementations.items():
            if name in ["execute_python", "update_plan", "record_findings", "final_answer"]:
                 continue # Skip built-ins already defined

            docstring = inspect.getdoc(func) or f"Executes the {name} tool."
            sig = inspect.signature(func)
            properties = {}
            required = []
            for param_name, param in sig.parameters.items():
                 param_type = "string" # Default
                 if param.annotation == int: param_type = "integer"
                 elif param.annotation == float: param_type = "number"
                 elif param.annotation == bool: param_type = "boolean"
                 elif param.annotation == list: param_type = "array"
                 elif param.annotation == dict: param_type = "object"
                 properties[param_name] = {"type": param_type, "description": f"Parameter '{param_name}'"}
                 if param.default == inspect.Parameter.empty:
                     required.append(param_name)

            definitions.append({
                "name": name,
                "description": docstring.split('\n')[0],
                "input_schema": {"type": "object", "properties": properties, "required": required}
            })
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
        # Print to CLI
        print(f"\n[TOOL CALL] -> {tool_name}")
        print(f"  Args: {json.dumps(tool_args)}")


        try:
            # Inject dependencies for built-in tools requiring state/executor
            if tool_name == "execute_python":
                # Special CLI print for pending code
                print(f"  Code:\n```python\n{tool_args.get('code', '')}\n```")
                print("[STATUS] Executing Python code...")
                result = tool_function(self.code_executor, self.state_manager, **tool_args)
            elif tool_name in ["update_plan", "record_findings", "final_answer"]:
                print(f"[STATUS] Executing tool: {tool_name}...")
                result = tool_function(self.state_manager, **tool_args)
            else:
                # Execute custom tools directly
                print(f"[STATUS] Executing tool: {tool_name}...")
                result = tool_function(**tool_args)

            # Print result to CLI
            print(f"[OBSERVATION] <- [{tool_name}] Result:")
            if isinstance(result, dict): # Nicer print for dicts (like python exec result)
                print(f"  stdout: {result.get('stdout')}")
                print(f"  error: {result.get('error')}")
            elif isinstance(result, str) and '\n' in result: # Print multi-line strings nicely
                 print("```")
                 print(result)
                 print("```")
            else:
                 print(f"  {result}")
            print("-" * 20)

            return result # Return the actual result object/value

        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
            error_msg = f"Error executing tool '{tool_name}': {type(e).__name__}: {e}"
            # Print error to CLI
            print(f"[OBSERVATION] <- [{tool_name}] ERROR:")
            print(f"  {error_msg}")
            print("-" * 20)
            return error_msg # Return error message string for the LLM

    def get_callable_tools_for_eval(self) -> Dict[str, Callable]:
        """Returns custom tools suitable for CodeExecutor's global scope."""
        eval_tools = {}
        for name, func in self._tool_implementations.items():
             # Exclude built-ins that manage agent state/execution
            if name not in ["execute_python", "update_plan", "record_findings", "final_answer"]:
                 eval_tools[name] = func
        # logger.debug(f"Providing tools for eval context: {list(eval_tools.keys())}")
        return eval_tools

def get_system_prompt(tool_definitions: List[dict], authorized_imports: List[str]) -> str:
    """Generates the system prompt including dynamic tool and import info."""
    formatted_tool_descriptions = "\n\n".join([
        f"**Tool: `{tool['name']}`**\nDescription: {tool['description']}\nInput Schema: {json.dumps(tool['input_schema'])}"
        for tool in tool_definitions
    ])
    auth_imports_list = ", ".join(authorized_imports) if authorized_imports else "None"

    # Using the same detailed prompt structure
    prompt = """You are an expert research assistant agent designed to solve complex tasks step-by-step using a limited set of tools. Your goal is to fully address the user's TASK.

**Workflow:**
1.  **Think:** Analyze the task and your current progress. Update your plan using the `update_plan` tool. Maintain the markdown checklist format. Check off completed items '[x]' and detail the next steps '[ ]'. Output your reasoning.
2.  **Act:** Choose the *single best tool* from the available list to execute the next logical step in your plan. Provide the required arguments for the chosen tool.
3.  **Observe:** You will receive the result of the tool execution.
4.  **Repeat:** Use the observation to inform your next Thought/Plan update and subsequent action. Continue until the task is fully resolved.

**Available Tools:**
You have access to the following tools. Use them strictly according to their descriptions and input schemas:""" + formatted_tool_descriptions + """**Python Execution (`execute_python` tool):**
- The Python execution environment is stateful. Variables and imports persist across `execute_python` calls within the same task.
- Only use imports from this allowed list: """ + auth_imports_list + """- Use `print()` within your Python code to output intermediate results or data you need for subsequent steps. These print outputs will be returned as the observation's 'stdout'.
- Handle potential errors gracefully within your Python code if possible.

**Python Helper Functions:**
Here is a list of helper functions that you can import to use in your code:

1. Firecrawl:
```python
from firecrawl import FirecrawlApp
import os
url = "https://www.google.com"

# Initialize the FirecrawlApp with your API key
app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
scrape_result = app.scrape_url(url, params={'formats': ['markdown']})
print(scrape_result["markdown"])```

2. File output:
```python
with open("output.md", "w") as f:
    f.write("Hello, world!")
```


**Planning and Findings:**
- Use the `update_plan` tool *at the beginning of each reasoning step* to keep track of your progress using the markdown checklist format.
- Before calling `final_answer`, use the `record_findings` tool to summarize your key results and conclusions in markdown format.

**Important Rules:**
- Always reason step-by-step before selecting a tool. Explain *why* you are choosing a specific tool in your thought process.
- Only call one tool at a time. Wait for the result before proceeding.
- Ensure you provide arguments exactly matching the tool's `input_schema`.
- If a tool fails, analyze the error message in the observation, adjust your plan, and try a different approach or tool call.
- Aim for clarity and conciseness in your reasoning and planning.
- Your final output *must* be provided using the `final_answer` tool.

Now, begin! Analyze the user's TASK and start the process. First, update the plan, then decide on your first action (tool call).
"""
    return prompt


class Agent:
    """Orchestrates the agent's lifecycle for CLI interaction."""
    def __init__(self, task: str):
        self.task = task
        print(f"[INFO] Initializing Agent for task: {self.task}")

        # Initialize components
        try:
            self.llm = LLMInteraction()
        except ValueError as e:
             print(f"[ERROR] Failed to initialize LLM: {e}")
             sys.exit(1)


        # Init StateManager with temporary prompt
        self.state_manager = StateManager(initial_task=task, system_prompt="Initializing...")

        # Init CodeExecutor with empty dict first
        self.code_executor = CodeExecutor(initial_globals={})

        # Init ToolManager, which needs state and code executor
        self.tool_manager = ToolManager(
            state_manager=self.state_manager,
            code_executor=self.code_executor,
            allowed_imports=AUTHORIZED_IMPORTS
        )

        # Generate the real system prompt
        tool_definitions = self.tool_manager.get_tool_definitions()
        system_prompt = get_system_prompt(tool_definitions, AUTHORIZED_IMPORTS)
        self.state_manager.system_prompt = system_prompt # Update state

        # Prepare and set initial globals for code execution
        initial_globals = self._prepare_initial_globals()
        self.state_manager.set_initial_globals(initial_globals) # Set in state
        self.code_executor.globals_locals = initial_globals # Update executor directly

        print("[INFO] Agent initialized successfully.")
        print("--- INITIAL PLAN ---")
        print(self.state_manager.get_plan())
        print("--------------------\n")


    def _prepare_initial_globals(self) -> Dict[str, Any]:
        """Prepares the initial global scope for the CodeExecutor."""
        initial_globals = {}
        # Import allowed modules
        for import_name in AUTHORIZED_IMPORTS:
            try:
                module = importlib.import_module(import_name)
                initial_globals[import_name] = module
                logger.info(f"Made module '{import_name}' available to code execution.")
            except ImportError:
                logger.warning(f"Could not import module '{import_name}' for code execution.")

        # Add callable *custom* tools
        callable_tools = self.tool_manager.get_callable_tools_for_eval()
        initial_globals.update(callable_tools)
        logger.info(f"Made tools {list(callable_tools.keys())} available to code execution.")

        # Add __builtins__ cautiously if needed, often implicitly available
        # initial_globals['__builtins__'] = __builtins__
        return initial_globals

    def run(self):
        """Runs the agent's main loop for CLI."""
        print("[STATUS] Agent starting run loop...")
        max_iterations = 15
        iterations = 0

        while not self.state_manager.check_done() and iterations < max_iterations:
            iterations += 1
            print(f"\n=========== AGENT ITERATION {iterations} ===========")
            print("[STATUS] Preparing LLM request...")

            # 1. Get state for LLM
            messages = self.state_manager.get_history()
            system_prompt = self.state_manager.get_system_prompt()
            tool_definitions = self.tool_manager.get_tool_definitions()

            # 2. Call LLM
            print("[STATUS] Calling LLM...")
            response_message = self.llm.generate_response(messages, system_prompt, tool_definitions)

            if response_message is None or response_message.content is None:
                print("[ERROR] LLM interaction failed or returned empty content. Terminating.")
                break

            # Add assistant's response message to history before processing
            self.state_manager.add_assistant_message(response_message.content)

            # 3. Process LLM response blocks
            executed_tool_this_turn = False
            for block in response_message.content:
                if block.type == "text":
                    print("\n[THOUGHT]")
                    print(block.text)
                    print("-" * 20)

                elif block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id
                    executed_tool_this_turn = True

                    # Execute the tool (prints call/status/observation within execute_tool)
                    result = self.tool_manager.execute_tool(tool_name, tool_input)

                    # Determine if result indicates an error
                    is_error = False
                    result_content_for_llm = result
                    if isinstance(result, str) and result.lower().startswith("error:"):
                         is_error = True
                    # Check dict format from execute_python_impl
                    if isinstance(result, dict) and result.get("error"):
                         is_error = True
                         result_content_for_llm = f"Error during execution: {result['error']}" # Pass error string to LLM

                    # Add tool result message to state for the *next* LLM call
                    self.state_manager.add_tool_result(
                         tool_use_id=tool_use_id,
                         result=result_content_for_llm, # Send stringified/error detail to LLM
                         is_error=is_error
                    )
                    # Plan/Findings updates are printed within their state_manager methods

            if not executed_tool_this_turn and response_message.stop_reason == 'stop_sequence':
                 print("[WARNING] LLM finished turn without using a tool. Task may be stalled.")
                 # Potential loop break or re-prompt logic could go here

            # Optional delay
            time.sleep(0.5)


        # Loop finished
        print("\n=========== AGENT FINISHED ===========")
        if self.state_manager.check_done():
            final_answer = self.state_manager.get_final_answer()
            print("\n[FINAL ANSWER]")
            print(final_answer)
            print("-" * 20)
            print("[STATUS] Task completed successfully.")
        elif iterations >= max_iterations:
            print("[ERROR] Agent reached maximum iterations.")
            print("[STATUS] Task incomplete (max iterations).")
        else:
             print("[WARNING] Agent loop exited unexpectedly.")
             print("[STATUS] Task incomplete.")
        print("======================================")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Agentic Library from the command line.")
    parser.add_argument("task", help="The task for the agent to perform.")
    args = parser.parse_args()

    print("*" * 50)
    print("         Command Line Agent Runner")
    print("*" * 50)

    if not ANTHROPIC_API_KEY:
        print("[ERROR] No Anthropic API key found in environment variables.")
        print("Please set ANTHROPIC_API_KEY in your environment.")
        sys.exit(1)

    try:
        agent = Agent(task=args.task)
        agent.run()
    except Exception as e:
        logger.error(f"An unexpected error occurred during agent execution: {e}", exc_info=True)
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")
        sys.exit(1)