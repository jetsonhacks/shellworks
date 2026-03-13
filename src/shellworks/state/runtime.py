"""
directory: src/shellworks/state/
file:      runtime.py

RuntimeLLMState — the live model selection for a specific run of Shellworks.

Runtime state is not persisted automatically. If a user explicitly requests
that a model become the new default, that change is written back to the
endpoint file. A temporary model switch during a run updates runtime state
only and does not touch disk.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeLLMState:
    active_endpoint_name: str
    active_model: str
