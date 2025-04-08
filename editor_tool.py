import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

Command = Literal["view", "create", "str_replace", "insert", "undo_edit"]


class EditorTool:
    """
    An filesystem editor tool that allows the agent to view, create, and edit files.
    The tool parameters are defined by Anthropic and are not editable.
    """

    api_type: Literal["text_editor_20250124"] = "text_editor_20250124"
    name: Literal["str_replace_editor"] = "str_replace_editor"

    def __init__(self, root_path: Union[str, Path]):
        """
        Initialize the editor tool with a root path where all file operations will be performed.

        Args:
            root_path: The root directory path where all file operations will be restricted to.
        """
        self._file_history: Dict[Path, List[str]] = defaultdict(list)
        self.root_path = Path(root_path).resolve()
        if not self.root_path.exists():
            raise ValueError(f"Root path {self.root_path} does not exist")
        if not self.root_path.is_dir():
            raise ValueError(f"Root path {self.root_path} is not a directory")

    def to_params(self) -> Dict[str, str]:
        """Return the tool parameters for Anthropic's API."""
        return {
            "name": self.name,
            "type": self.api_type,
        }

    def validate_path(self, command: str, path: Path) -> None:
        """Validate that the path and command combination is valid."""
        try:
            # Resolve the path to its absolute form
            abs_path = path.resolve()
            # Check if the path is within the root directory
            if not str(abs_path).startswith(str(self.root_path)):
                raise ValueError(
                    f"Path {path} is outside the root directory {self.root_path}"
                )

            # Create parent directories if this is a create command
            if command == "create":
                parent = abs_path.parent
                if not parent.exists():
                    parent.mkdir(parents=True)

            if not path.is_absolute():
                suggested_path = self.root_path / path
                raise ValueError(
                    f"Path {path} is not absolute. Did you mean {suggested_path}?"
                )

            if not path.exists() and command != "create":
                raise ValueError(f"Path {path} does not exist")

            if path.exists() and command == "create":
                raise ValueError(f"File already exists at {path}")

            if path.is_dir() and command != "view":
                raise ValueError(
                    f"Path {path} is a directory and only view command is allowed"
                )
        except Exception as e:
            if not isinstance(e, ValueError):
                raise ValueError(f"Error validating path: {str(e)}")
            raise

    def normalize_path(self, path: str) -> Path:
        """Convert a path string to an absolute Path object within the root directory."""
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = self.root_path / path_obj
        return path_obj

    def read_file(self, path: Path) -> str:
        """Read contents of a file."""
        try:
            return path.read_text()
        except Exception as e:
            raise ValueError(f"Error reading {path}: {str(e)}")

    def write_file(self, path: Path, content: str) -> None:
        """Write content to a file."""
        try:
            path.write_text(content)
        except Exception as e:
            raise ValueError(f"Error writing to {path}: {str(e)}")

    def view(self, path: str, view_range: Optional[List[int]] = None) -> str:
        """View contents of a file or directory."""
        _path = self.normalize_path(path)
        self.validate_path("view", _path)

        if _path.is_dir():
            if view_range:
                raise ValueError("view_range not allowed for directories")
            try:
                files = list(_path.glob("**/*"))
                return json.dumps(
                    {
                        "status": "success",
                        "type": "directory",
                        "contents": [str(f.relative_to(_path)) for f in files],
                    }
                )
            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)})

        content = self.read_file(_path)
        lines = content.split("\n")

        if view_range:
            start, end = view_range
            if start < 1 or start > len(lines):
                raise ValueError(f"Invalid start line {start}")
            if end != -1 and (end < start or end > len(lines)):
                raise ValueError(f"Invalid end line {end}")

            if end == -1:
                content = "\n".join(lines[start - 1 :])
            else:
                content = "\n".join(lines[start - 1 : end])

        return json.dumps({"status": "success", "type": "file", "content": content})

    def create(self, path: str, content: str) -> str:
        """Create a new file with given content."""
        _path = self.normalize_path(path)
        self.validate_path("create", _path)

        self.write_file(_path, content)
        return json.dumps({"status": "success", "message": f"File created at {path}"})

    def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        """Replace occurrences of old_str with new_str in file."""
        _path = self.normalize_path(path)
        self.validate_path("str_replace", _path)

        content = self.read_file(_path)
        occurrences = content.count(old_str)

        if occurrences == 0:
            return json.dumps(
                {"status": "error", "message": f"String '{old_str}' not found in file"}
            )
        elif occurrences > 1:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Multiple occurrences ({occurrences}) of '{old_str}' found",
                }
            )

        # Store old content in history
        self._file_history[_path].append(content)

        # Replace and write new content
        new_content = content.replace(old_str, new_str)
        self.write_file(_path, new_content)

        return json.dumps(
            {"status": "success", "message": f"Replaced '{old_str}' with '{new_str}'"}
        )

    def insert(self, path: str, insert_line: int, new_str: str) -> str:
        """Insert text at a specific line in the file."""
        _path = self.normalize_path(path)
        self.validate_path("insert", _path)

        content = self.read_file(_path)
        lines = content.split("\n")

        if insert_line < 0 or insert_line > len(lines):
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Invalid insert line {insert_line}. File has {len(lines)} lines.",
                }
            )

        # Store old content in history
        self._file_history[_path].append(content)

        # Insert the new string at the specified line
        lines.insert(insert_line, new_str)
        new_content = "\n".join(lines)
        self.write_file(_path, new_content)

        return json.dumps(
            {"status": "success", "message": f"Inserted text at line {insert_line}"}
        )

    def undo_edit(self, path: str) -> str:
        """Undo last edit to file."""
        _path = self.normalize_path(path)
        self.validate_path("undo_edit", _path)

        if not self._file_history[_path]:
            return json.dumps(
                {"status": "error", "message": "No edit history available"}
            )

        previous_content = self._file_history[_path].pop()
        self.write_file(_path, previous_content)

        return json.dumps(
            {"status": "success", "message": "Last edit undone successfully"}
        )

    def __call__(
        self,
        *,
        command: Command,
        path: str,
        file_text: Optional[str] = None,
        view_range: Optional[List[int]] = None,
        old_str: Optional[str] = None,
        new_str: Optional[str] = None,
        insert_line: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Execute a command with given parameters."""
        commands: Dict[str, Any] = {
            "view": self.view,
            "create": self.create,
            "str_replace": self.str_replace,
            "insert": self.insert,
            "undo_edit": self.undo_edit,
        }

        if command not in commands:
            return json.dumps(
                {"status": "error", "message": f"Unknown command: {command}"}
            )

        try:
            if command == "view":
                return commands[command](path, view_range)
            elif command == "create":
                if file_text is None:
                    raise ValueError(
                        "Parameter `file_text` is required for command: create"
                    )
                return commands[command](path, file_text)
            elif command == "str_replace":
                if old_str is None or new_str is None:
                    raise ValueError(
                        "Parameters `old_str` and `new_str` are required for command: str_replace"
                    )
                return commands[command](path, old_str, new_str)
            elif command == "insert":
                if insert_line is None or new_str is None:
                    raise ValueError(
                        "Parameters `insert_line` and `new_str` are required for command: insert"
                    )
                return commands[command](path, insert_line, new_str)
            else:  # undo_edit
                return commands[command](path)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})
