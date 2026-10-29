"""Checkpoint and console logging manager for CAMIR phases.

Ensures outputs, tabular results, and full console logs are stored
in structured .txt files organized by phase for future reference and reproducibility.
"""

from datetime import datetime, timezone
import io
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional
from camir.utils.hardware import get_hardware_info, get_thermal_state


DEFAULT_CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


class TeeCapture:
    """Context manager to tee stdout and stderr to both console and a .txt log file."""

    def __init__(self, log_filepath: Path):
        self.log_filepath = Path(log_filepath)
        self.log_filepath.parent.mkdir(parents=True, exist_ok=True)
        self._stdout_original = sys.stdout
        self._stderr_original = sys.stderr
        self._buffer = io.StringIO()
        self._file = None

    def __enter__(self):
        self._file = open(self.log_filepath, "w", encoding="utf-8")
        
        class _TeeStream:
            def __init__(self, primary, secondary, buffer):
                self.primary = primary
                self.secondary = secondary
                self.buffer = buffer

            def write(self, data):
                self.primary.write(data)
                self.secondary.write(data)
                self.buffer.write(data)

            def flush(self):
                self.primary.flush()
                self.secondary.flush()

        sys.stdout = _TeeStream(self._stdout_original, self._file, self._buffer)
        sys.stderr = _TeeStream(self._stderr_original, self._file, self._buffer)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout_original
        sys.stderr = self._stderr_original
        if self._file:
            self._file.flush()
            self._file.close()

    @property
    def captured_text(self) -> str:
        return self._buffer.getvalue()


class CheckpointManager:
    """Manages saving phase execution outputs and console logs into dedicated .txt files."""

    def __init__(self, phase: int = 1, base_dir: Optional[Path] = None):
        self.phase = phase
        self.base_dir = Path(base_dir) if base_dir else (DEFAULT_CHECKPOINTS_DIR / f"phase_{phase}")
        self.console_logs_dir = self.base_dir / "console_logs"
        self.outputs_dir = self.base_dir / "outputs"
        
        self.console_logs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def get_console_log_path(self, task_name: str) -> Path:
        """Return canonical path for a task console log .txt file."""
        return self.console_logs_dir / f"{task_name}.txt"

    def get_output_path(self, output_name: str) -> Path:
        """Return canonical path for an output .txt file."""
        return self.outputs_dir / f"{output_name}.txt"

    def save_console_log(self, task_name: str, content: str) -> Path:
        """Save captured console log string into a .txt file."""
        path = self.get_console_log_path(task_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def save_output(
        self,
        output_name: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save an output checkpoint, stripping any trailing .txt if present."""
        clean_name = output_name[:-4] if output_name.endswith(".txt") else output_name
        return self.save_output_checkpoint(clean_name, content, metadata=metadata)

    def save_output_checkpoint(
        self,
        output_name: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:

        """Save a human-readable text checkpoint with system metadata header."""
        path = self.get_output_path(output_name)
        timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        hw_info = get_hardware_info()
        thermal = get_thermal_state()

        header_lines = [
            "=" * 80,
            f"CAMIR PHASE {self.phase} CHECKPOINT: {output_name.upper()}",
            "=" * 80,
            f"Timestamp (UTC): {timestamp_utc}",
            f"Hardware:        {hw_info}",
            f"Thermal State:   {thermal}",
        ]
        if metadata:
            header_lines.append("-" * 80)
            header_lines.append("METADATA & PARAMETERS:")
            for k, v in metadata.items():
                header_lines.append(f"  {k}: {v}")
        header_lines.append("=" * 80)
        header_lines.append("")

        full_content = "\n".join(header_lines) + content.strip() + "\n"

        with open(path, "w", encoding="utf-8") as f:
            f.write(full_content)

        return path
