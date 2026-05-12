"""Base class and process-management utilities for toggle-mode recorders."""

from __future__ import annotations

import os
import signal
import time
from abc import ABC, abstractmethod


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _kill_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    for _ in range(20):
        time.sleep(0.1)
        if not _pid_alive(pid):
            return
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


class AudioRecorder(ABC):
    """Abstract base for toggle-mode background recorders."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier stored in the lock file 'recorder' field."""

    def is_available(self) -> bool:
        """Return True if this recorder's runtime dependencies are present."""
        return True

    @abstractmethod
    def start(self, audio_path: str, device: str | int | None) -> int | None:
        """Spawn a background recording process.

        Returns the PID on success, or None if the recorder failed to start.
        """

    def stop(self, pid: int) -> None:
        """Stop a previously started recording process (SIGTERM → SIGKILL)."""
        _kill_process(pid)
