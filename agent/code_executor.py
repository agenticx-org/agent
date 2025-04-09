import contextlib
import io
import logging
from typing import Any, Dict

logger = logging.getLogger("CLI_Agent")


class CodeExecutor:
    """Executes Python code snippets within a controlled, persistent context."""

    def __init__(self, initial_globals: Dict[str, Any]):
        self.globals_locals: Dict[str, Any] = initial_globals.copy()
        logger.info(
            f"CodeExecutor initialized with globals: {list(self.globals_locals.keys())}"
        )

    def execute(self, code_string: str) -> Dict[str, Any]:
        """Executes Python code, captures stdout/errors, updates scope."""
        logger.info(f"Executing code:\n---\n{code_string}\n---")
        stdout_capture = io.StringIO()
        error_message = None
        globals_before = self.globals_locals.copy()

        try:
            with contextlib.redirect_stdout(stdout_capture):
                compiled_code = compile(code_string, "<string>", "exec")
                exec(compiled_code, self.globals_locals)
        except Exception as e:
            error_message = f"{type(e).__name__}: {e}"
            logger.error(f"Code execution failed: {error_message}")

        stdout_result = stdout_capture.getvalue()
        if stdout_result:
            logger.info(f"Execution stdout:\n{stdout_result}")

        updated_globals = {
            k: v
            for k, v in self.globals_locals.items()
            if k not in globals_before or globals_before[k] is not v
        }
        if "__builtins__" in updated_globals and updated_globals[
            "__builtins__"
        ] == globals_before.get("__builtins__"):
            del updated_globals["__builtins__"]

        return {
            "stdout": stdout_result,
            "error": error_message,
            "updated_globals": updated_globals,
        }

    def get_current_globals(self) -> Dict[str, Any]:
        return self.globals_locals.copy()
