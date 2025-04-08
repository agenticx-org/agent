import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union, Any
from contextlib import contextmanager

ShellCommand = Literal["shell_exec", "shell_view", "shell_wait", "shell_write_to_process", "shell_kill_process"]


class ShellTool:
    """
    A shell tool that allows the agent to execute commands in shell sessions,
    view outputs, write to processes, wait for processes, and kill processes.
    """

    api_type: Literal["shell_tool_20250124"] = "shell_tool_20250124"
    name: Literal["shell_tool"] = "shell_tool"

    def __init__(self, root_path: Union[str, Path]):
        """
        Initialize the shell tool with a root path where all shell operations will be restricted to.

        Args:
            root_path: The root directory path where all file operations will be restricted to.
        """
        self.root_path = Path(root_path).resolve()
        if not self.root_path.exists():
            raise ValueError(f"Root path {self.root_path} does not exist")
        if not self.root_path.is_dir():
            raise ValueError(f"Root path {self.root_path} is not a directory")
        
        # Dictionary to store active shell sessions
        self._shell_sessions: Dict[str, Dict[str, Any]] = {}

    def to_params(self) -> List[Dict[str, Any]]:
        """Return the tool parameters for Anthropic's API."""
        return [
            {
                "name": "shell_exec",
                "description": "Execute commands in a specified shell session. Use for running code, installing packages, or managing files.",
                "input_schema": {
                "type": "object",
                "properties": {
                    "id": {
                    "type": "string",
                    "description": "Unique identifier of the target shell session"
                    },
                    "exec_dir": {
                    "type": "string",
                    "description": "Working directory for command execution (must use absolute path)"
                    },
                    "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                    }
                },
                "required": ["id", "exec_dir", "command"]
                }
            },
            {
                "name": "shell_view",
                "description": "View the content of a specified shell session. Use for checking command execution results or monitoring output.",
                "input_schema": {
                "type": "object",
                "properties": {
                    "id": {
                    "type": "string",
                    "description": "Unique identifier of the target shell session"
                    }
                },
                "required": ["id"]
                }
            },
            {
                "name": "shell_wait",
                "description": "Wait for the running process in a specified shell session to return. Use after running commands that require longer runtime.",
                "input_schema": {
                "type": "object",
                "properties": {
                    "id": {
                    "type": "string",
                    "description": "Unique identifier of the target shell session"
                    },
                    "seconds": {
                    "type": "integer",
                    "description": "Wait duration in seconds"
                    }
                },
                "required": ["id", "seconds"]
                }
            },
            {
                "name": "shell_write_to_process",
                "description": "Write input to a running process in a specified shell session. Use for responding to interactive command prompts.",
                "input_schema": {
                "type": "object",
                "properties": {
                    "id": {
                    "type": "string",
                    "description": "Unique identifier of the target shell session"
                    },
                    "input": {
                    "type": "string",
                    "description": "Input content to write to the process"
                    },
                    "press_enter": {
                    "type": "boolean",
                    "description": "Whether to press Enter key after input"
                    }
                },
                "required": ["id", "input", "press_enter"]
                }
            },
            {
                "name": "shell_kill_process",
                "description": "Terminate a running process in a specified shell session. Use for stopping long-running processes or handling frozen commands.",
                "input_schema": {
                "type": "object",
                "properties": {
                    "id": {
                    "type": "string",
                    "description": "Unique identifier of the target shell session"
                    }
                },
                "required": ["id"]
                }
            }
            ]


    def validate_path(self, exec_dir: Path) -> None:
        """Validate that the execution directory is valid and within the root path."""
        try:
            # Resolve the path to its absolute form
            abs_path = exec_dir.resolve()
            # Check if the path is within the root directory
            if not str(abs_path).startswith(str(self.root_path)):
                raise ValueError(
                    f"Path {exec_dir} is outside the root directory {self.root_path}"
                )

            if not exec_dir.exists():
                raise ValueError(f"Directory {exec_dir} does not exist")

            if not exec_dir.is_dir():
                raise ValueError(f"Path {exec_dir} is not a directory")
                
        except Exception as e:
            if not isinstance(e, ValueError):
                raise ValueError(f"Error validating path: {str(e)}")
            raise

    def shell_exec(self, id: str, exec_dir: str, command: str) -> str:
        """
        Execute a shell command in the specified working directory.
        
        Args:
            id: Unique identifier for this shell session
            exec_dir: Directory where the command will be executed
            command: The shell command to execute
            
        Returns:
            JSON string with status information
        """
        try:
            exec_path = Path(exec_dir)
            if not exec_path.is_absolute():
                raise ValueError(f"exec_dir must be an absolute path: {exec_dir}")
                
            self.validate_path(exec_path)
            
            # Check if session already exists
            if id in self._shell_sessions:
                # If there's an existing process, terminate it
                self.shell_kill_process(id)
            
            # Create new shell process
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=str(exec_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Store the process info
            self._shell_sessions[id] = {
                "process": process,
                "command": command,
                "exec_dir": str(exec_path),
                "output": [],
                "running": True,
                "start_time": time.time()
            }
            
            return json.dumps({
                "status": "success", 
                "message": f"Command started in session {id}",
                "pid": process.pid
            })
            
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    def shell_view(self, id: str) -> str:
        """
        View the output of a shell session.
        
        Args:
            id: Unique identifier of the shell session
            
        Returns:
            JSON string with session output
        """
        try:
            if id not in self._shell_sessions:
                return json.dumps({"status": "error", "message": f"Session {id} does not exist"})
                
            session = self._shell_sessions[id]
            process = session["process"]
            
            # Collect any new output
            output = []
            while True:
                # Check if there's data to read without blocking
                if process.poll() is not None:
                    # Process has finished
                    session["running"] = False
                
                line = process.stdout.readline()
                if not line:
                    break
                output.append(line.rstrip())
                session["output"].append(line.rstrip())
            
            # Check if process is still running
            is_running = process.poll() is None
            session["running"] = is_running
            
            return json.dumps({
                "status": "success",
                "output": session["output"],
                "running": is_running,
                "exit_code": process.returncode if not is_running else None,
                "command": session["command"],
                "exec_dir": session["exec_dir"]
            })
            
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    def shell_wait(self, id: str, seconds: Optional[int] = None) -> str:
        """
        Wait for a shell process to complete or until timeout.
        
        Args:
            id: Unique identifier of the shell session
            seconds: Optional timeout in seconds
            
        Returns:
            JSON string with status information
        """
        try:
            if id not in self._shell_sessions:
                return json.dumps({"status": "error", "message": f"Session {id} does not exist"})
                
            session = self._shell_sessions[id]
            process = session["process"]
            
            # Set default timeout
            timeout = seconds if seconds is not None else 60
            start_time = time.time()
            
            # Wait for process to complete or timeout
            output = []
            try:
                while process.poll() is None:
                    # Read any available output
                    line = process.stdout.readline()
                    if line:
                        output.append(line.rstrip())
                        session["output"].append(line.rstrip())
                    
                    # Check for timeout
                    if time.time() - start_time > timeout:
                        return json.dumps({
                            "status": "timeout",
                            "message": f"Process did not complete within {timeout} seconds",
                            "output": output
                        })
                    
                    # Small sleep to prevent CPU hogging
                    time.sleep(0.1)
                
                # Read any remaining output
                for line in process.stdout:
                    output.append(line.rstrip())
                    session["output"].append(line.rstrip())
                
                session["running"] = False
                return json.dumps({
                    "status": "success",
                    "message": f"Process completed with exit code {process.returncode}",
                    "exit_code": process.returncode,
                    "output": output
                })
                
            except subprocess.TimeoutExpired:
                return json.dumps({
                    "status": "timeout",
                    "message": f"Process did not complete within {timeout} seconds",
                    "output": output
                })
                
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    def shell_write_to_process(self, id: str, input: str, press_enter: bool) -> str:
        """
        Write input to a running process.
        
        Args:
            id: Unique identifier of the shell session
            input: Text to write to the process
            press_enter: Whether to append a newline character
            
        Returns:
            JSON string with status information
        """
        try:
            if id not in self._shell_sessions:
                return json.dumps({"status": "error", "message": f"Session {id} does not exist"})
                
            session = self._shell_sessions[id]
            process = session["process"]
            
            # Check if process is running
            if process.poll() is not None:
                return json.dumps({
                    "status": "error",
                    "message": "Process is not running"
                })
            
            # Prepare input with optional newline
            input_text = input
            if press_enter:
                input_text += "\n"
            
            # Write to stdin
            process.stdin.write(input_text)
            process.stdin.flush()
            
            return json.dumps({
                "status": "success",
                "message": f"Input sent to process in session {id}"
            })
            
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    def shell_kill_process(self, id: str) -> str:
        """
        Kill a running process.
        
        Args:
            id: Unique identifier of the shell session
            
        Returns:
            JSON string with status information
        """
        try:
            if id not in self._shell_sessions:
                return json.dumps({"status": "error", "message": f"Session {id} does not exist"})
                
            session = self._shell_sessions[id]
            process = session["process"]
            
            # Check if process is running
            if process.poll() is None:
                # Try to terminate gracefully first
                process.terminate()
                
                # Give it a moment to terminate
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate
                    process.kill()
            
            session["running"] = False
            return json.dumps({
                "status": "success",
                "message": f"Process in session {id} terminated"
            })
            
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    def __call__(
        self,
        *,
        command: ShellCommand,
        id: str,
        exec_dir: Optional[str] = None,
        shell_command: Optional[str] = None,
        seconds: Optional[int] = None,
        input: Optional[str] = None,
        press_enter: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """Execute a shell command with given parameters."""
        commands = {
            "shell_exec": self.shell_exec,
            "shell_view": self.shell_view,
            "shell_wait": self.shell_wait,
            "shell_write_to_process": self.shell_write_to_process,
            "shell_kill_process": self.shell_kill_process,
        }

        if command not in commands:
            return json.dumps(
                {"status": "error", "message": f"Unknown command: {command}"}
            )

        try:
            if command == "shell_exec":
                if exec_dir is None or shell_command is None:
                    raise ValueError(
                        "Parameters `exec_dir` and `command` are required for command: shell_exec"
                    )
                return commands[command](id, exec_dir, shell_command)
            elif command == "shell_view":
                return commands[command](id)
            elif command == "shell_wait":
                return commands[command](id, seconds)
            elif command == "shell_write_to_process":
                if input is None or press_enter is None:
                    raise ValueError(
                        "Parameters `input` and `press_enter` are required for command: shell_write_to_process"
                    )
                return commands[command](id, input, press_enter)
            elif command == "shell_kill_process":
                return commands[command](id)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)}) 