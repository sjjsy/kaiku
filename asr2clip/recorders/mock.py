"""Mock recorder for toggle mode — copies a source WAV instead of recording.

Used with mock_devices: entries in the config. The child process copies the
source file immediately then sleeps until killed by the second --toggle call,
mirroring the life-cycle of a real recorder subprocess.
"""

from __future__ import annotations

import os
import shutil
import time

from .base import AudioRecorder


class MockRecorder(AudioRecorder):
    """Toggle-mode recorder that serves a pre-recorded WAV file.

    Behaves like a real recorder: start() forks a child that 'records'
    (i.e. copies the source file) and sleeps until SIGTERM. The lock-file
    protocol in toggle.py then treats this child's PID normally.

    device_info must carry a mock_source attribute pointing to the source WAV.
    """

    @property
    def name(self) -> str:
        return "mock"

    def is_available(self) -> bool:
        return True

    def start(self, audio_path: str, device_info) -> int | None:
        source_file = getattr(device_info, "mock_source", None)
        if not source_file:
            return None

        pid = os.fork()
        if pid == 0:
            # Child: copy source to audio_path then sleep until killed
            try:
                shutil.copy2(source_file, audio_path)
            except Exception:
                pass
            time.sleep(3600)
            os._exit(0)

        return pid
