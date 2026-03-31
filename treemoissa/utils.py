"""Shared utilities — no heavy dependencies."""

from __future__ import annotations

import re
import subprocess
from contextlib import contextmanager
from pathlib import Path


def _sanitize(name: str) -> str:
    """Sanitize a name for use as a directory component."""
    name = name.strip().lower()
    # Remove characters not safe for filenames
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name or "unknown"


def is_wsl() -> bool:
    """Detect if running under WSL2."""
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


@contextmanager
def wsl_keep_awake():
    """Prevent Windows from sleeping while running under WSL2.

    Launches a background PowerShell process that calls
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) and keeps
    it alive until the context manager exits.
    """
    if not is_wsl():
        yield
        return

    ps_script = (
        "[System.Runtime.InteropServices.Marshal]::"
        "PrelinkAll([PowerState]);"
        "Add-Type @\"\n"
        "using System;\n"
        "using System.Runtime.InteropServices;\n"
        "public class PowerState {\n"
        "    [DllImport(\"kernel32.dll\")]\n"
        "    public static extern uint SetThreadExecutionState(uint esFlags);\n"
        "}\n"
        "\"@;\n"
        "[PowerState]::SetThreadExecutionState(0x80000003);\n"
        "while($true){Start-Sleep -Seconds 60;"
        "[PowerState]::SetThreadExecutionState(0x80000003)}"
    )

    proc = None
    try:
        proc = subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-Command", ps_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        yield
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=5)
