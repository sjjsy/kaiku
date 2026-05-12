"""Unit tests for the asr2clip.recorders package."""

from __future__ import annotations

import os
import signal
import wave
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# _pid_alive
# ---------------------------------------------------------------------------

class TestPidAlive:
    def test_current_process_is_alive(self):
        from asr2clip.recorders.base import _pid_alive
        assert _pid_alive(os.getpid()) is True

    def test_nonexistent_pid_returns_false(self):
        from asr2clip.recorders.base import _pid_alive
        assert _pid_alive(999999) is False


# ---------------------------------------------------------------------------
# _kill_process
# ---------------------------------------------------------------------------

class TestKillProcess:
    def test_sigterm_only_when_process_exits_quickly(self):
        from asr2clip.recorders.base import _kill_process
        with patch("os.kill") as mock_kill, \
             patch("time.sleep"), \
             patch("asr2clip.recorders.base._pid_alive", return_value=False):
            _kill_process(12345)
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)

    def test_sigkill_sent_when_process_hangs(self):
        from asr2clip.recorders.base import _kill_process
        with patch("os.kill") as mock_kill, \
             patch("time.sleep"), \
             patch("asr2clip.recorders.base._pid_alive", return_value=True):
            _kill_process(12345)
        sent = [c.args[1] for c in mock_kill.call_args_list]
        assert signal.SIGTERM in sent
        assert signal.SIGKILL in sent

    def test_oserror_on_sigterm_is_swallowed(self):
        from asr2clip.recorders.base import _kill_process
        with patch("os.kill", side_effect=OSError):
            _kill_process(99999)  # must not raise


# ---------------------------------------------------------------------------
# ArecordRecorder
# ---------------------------------------------------------------------------

class TestArecordRecorder:
    def test_is_available_when_binary_on_path(self):
        from asr2clip.recorders.arecord import ArecordRecorder
        with patch("shutil.which", return_value="/usr/bin/arecord"):
            assert ArecordRecorder().is_available() is True

    def test_not_available_when_binary_missing(self):
        from asr2clip.recorders.arecord import ArecordRecorder
        with patch("shutil.which", return_value=None):
            assert ArecordRecorder().is_available() is False

    def test_start_returns_none_when_unavailable(self, tmp_path):
        from asr2clip.recorders.arecord import ArecordRecorder
        with patch("shutil.which", return_value=None):
            assert ArecordRecorder().start(str(tmp_path / "out.wav"), None) is None

    def test_start_returns_pid_on_success(self, tmp_path):
        from asr2clip.recorders.arecord import ArecordRecorder
        proc = MagicMock()
        proc.pid = 9999
        proc.poll.return_value = None
        with patch("shutil.which", return_value="/usr/bin/arecord"), \
             patch("asr2clip.recorders.arecord.popen_subprocess", return_value=proc) as mock_popen, \
             patch("time.sleep"), \
             patch("asr2clip.recorders.arecord._device_native_rate", return_value=44100):
            pid = ArecordRecorder().start(str(tmp_path / "out.wav"), None)
        assert pid == 9999
        mock_popen.assert_called_once()

    def test_start_returns_none_on_immediate_exit(self, tmp_path):
        from asr2clip.recorders.arecord import ArecordRecorder
        proc = MagicMock()
        proc.pid = 1111
        proc.poll.return_value = 1
        proc.stderr.read.return_value = b"arecord: Device or resource busy"
        with patch("shutil.which", return_value="/usr/bin/arecord"), \
             patch("asr2clip.recorders.arecord.popen_subprocess", return_value=proc), \
             patch("time.sleep"), \
             patch("asr2clip.recorders.arecord._device_native_rate", return_value=44100):
            pid = ArecordRecorder().start(str(tmp_path / "out.wav"), None)
        assert pid is None

    def test_friendly_name_resolved_to_alsa_flag(self, tmp_path):
        from asr2clip.recorders.arecord import ArecordRecorder
        proc = MagicMock()
        proc.pid = 7777
        proc.poll.return_value = None
        with patch("shutil.which", return_value="/usr/bin/arecord"), \
             patch("asr2clip.recorders.arecord._friendly_to_alsa", return_value="plughw:2,0"), \
             patch("asr2clip.recorders.arecord.popen_subprocess", return_value=proc) as mock_popen, \
             patch("time.sleep"), \
             patch("asr2clip.recorders.arecord._device_native_rate", return_value=44100):
            ArecordRecorder().start(str(tmp_path / "out.wav"), "Blue Snowball: USB Audio (hw:2,0)")
        cmd = mock_popen.call_args.args[0]
        assert "-D" in cmd
        idx = cmd.index("-D")
        assert cmd[idx + 1] == "plughw:2,0"

    def test_uses_popen_subprocess_not_raw_popen(self, tmp_path):
        from asr2clip.recorders.arecord import ArecordRecorder
        proc = MagicMock()
        proc.pid = 5555
        proc.poll.return_value = None
        with patch("shutil.which", return_value="/usr/bin/arecord"), \
             patch("asr2clip.recorders.arecord.popen_subprocess", return_value=proc) as mock_popen, \
             patch("subprocess.Popen") as mock_raw_popen, \
             patch("time.sleep"), \
             patch("asr2clip.recorders.arecord._device_native_rate", return_value=44100):
            ArecordRecorder().start(str(tmp_path / "out.wav"), None)
        mock_popen.assert_called_once()
        mock_raw_popen.assert_not_called()


# ---------------------------------------------------------------------------
# SounddeviceRecorder
# ---------------------------------------------------------------------------

class TestSounddeviceRecorder:
    def test_is_available_when_sounddevice_importable(self):
        from asr2clip.recorders.sounddevice_recorder import SounddeviceRecorder
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            assert SounddeviceRecorder().is_available() is True

    def test_not_available_when_sounddevice_missing(self):
        from asr2clip.recorders.sounddevice_recorder import SounddeviceRecorder
        with patch("importlib.util.find_spec", return_value=None):
            assert SounddeviceRecorder().is_available() is False

    def test_start_spawns_module_via_popen_subprocess(self, tmp_path):
        from asr2clip.recorders.sounddevice_recorder import SounddeviceRecorder
        proc = MagicMock()
        proc.pid = 5555
        proc.poll.return_value = None
        with patch("asr2clip.recorders.sounddevice_recorder.popen_subprocess", return_value=proc) as mock_popen, \
             patch("subprocess.Popen") as mock_raw_popen, \
             patch("time.sleep"), \
             patch("asr2clip.recorders.sounddevice_recorder._device_native_rate", return_value=44100):
            pid = SounddeviceRecorder().start(str(tmp_path / "out.wav"), "pulse")
        assert pid == 5555
        mock_popen.assert_called_once()
        mock_raw_popen.assert_not_called()
        cmd = mock_popen.call_args.args[0]
        assert "-m" in cmd
        assert "asr2clip.recorders.sounddevice_recorder" in cmd

    def test_start_returns_none_on_immediate_exit(self, tmp_path):
        from asr2clip.recorders.sounddevice_recorder import SounddeviceRecorder
        proc = MagicMock()
        proc.pid = 2222
        proc.poll.return_value = 1
        proc.stderr.read.return_value = b"sounddevice error: no device"
        with patch("asr2clip.recorders.sounddevice_recorder.popen_subprocess", return_value=proc), \
             patch("time.sleep"), \
             patch("asr2clip.recorders.sounddevice_recorder._device_native_rate", return_value=44100):
            pid = SounddeviceRecorder().start(str(tmp_path / "out.wav"), None)
        assert pid is None

    def test_device_none_passes_empty_string_arg(self, tmp_path):
        from asr2clip.recorders.sounddevice_recorder import SounddeviceRecorder
        proc = MagicMock()
        proc.pid = 3333
        proc.poll.return_value = None
        with patch("asr2clip.recorders.sounddevice_recorder.popen_subprocess", return_value=proc) as mock_popen, \
             patch("time.sleep"), \
             patch("asr2clip.recorders.sounddevice_recorder._device_native_rate", return_value=44100):
            SounddeviceRecorder().start(str(tmp_path / "out.wav"), None)
        cmd = mock_popen.call_args.args[0]
        # device arg is last; empty string means None
        assert cmd[-1] == ""

    def test_run_subprocess_writes_wav(self, tmp_path):
        from asr2clip.recorders.sounddevice_recorder import _run_subprocess

        fake_chunk = np.zeros((1024, 1), dtype="float32")

        class FakeStream:
            def __init__(self, **kwargs):
                self._cb = kwargs["callback"]

            def __enter__(self):
                self._cb(fake_chunk, 1024, None, None)
                raise SystemExit(0)

            def __exit__(self, *a):
                return False

        out = tmp_path / "recording.wav"
        with patch("sounddevice.InputStream", FakeStream), \
             patch("signal.signal"), \
             patch("signal.pause"):
            try:
                _run_subprocess(str(out), 44100, None)
            except SystemExit:
                pass

        assert out.exists()
        with wave.open(str(out)) as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 44100
            assert wf.getnframes() == 1024

    def test_run_subprocess_writes_empty_wav_when_no_audio(self, tmp_path):
        from asr2clip.recorders.sounddevice_recorder import _run_subprocess

        class FakeStream:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                raise SystemExit(0)

            def __exit__(self, *a):
                return False

        out = tmp_path / "empty.wav"
        with patch("sounddevice.InputStream", FakeStream), \
             patch("signal.signal"), \
             patch("signal.pause"):
            try:
                _run_subprocess(str(out), 44100, None)
            except SystemExit:
                pass

        assert out.exists()
        with wave.open(str(out)) as wf:
            assert wf.getnframes() == 0


# ---------------------------------------------------------------------------
# make_recorder
# ---------------------------------------------------------------------------

class TestMakeRecorder:
    def test_auto_picks_sounddevice_first(self):
        from asr2clip.recorders import ArecordRecorder, SounddeviceRecorder, make_recorder
        with patch.object(SounddeviceRecorder, "is_available", return_value=True):
            r = make_recorder("auto")
        assert isinstance(r, SounddeviceRecorder)

    def test_auto_falls_back_to_arecord(self):
        from asr2clip.recorders import ArecordRecorder, SounddeviceRecorder, make_recorder
        with patch.object(SounddeviceRecorder, "is_available", return_value=False), \
             patch.object(ArecordRecorder, "is_available", return_value=True):
            r = make_recorder("auto")
        assert isinstance(r, ArecordRecorder)

    def test_auto_exits_when_nothing_available(self):
        from asr2clip.recorders import ArecordRecorder, SounddeviceRecorder, make_recorder
        with patch.object(SounddeviceRecorder, "is_available", return_value=False), \
             patch.object(ArecordRecorder, "is_available", return_value=False):
            with pytest.raises(SystemExit):
                make_recorder("auto")

    def test_none_treated_as_auto(self):
        from asr2clip.recorders import SounddeviceRecorder, make_recorder
        with patch.object(SounddeviceRecorder, "is_available", return_value=True):
            r = make_recorder(None)
        assert isinstance(r, SounddeviceRecorder)

    def test_explicit_sounddevice(self):
        from asr2clip.recorders import SounddeviceRecorder, make_recorder
        assert isinstance(make_recorder("sounddevice"), SounddeviceRecorder)

    def test_explicit_arecord(self):
        from asr2clip.recorders import ArecordRecorder, make_recorder
        assert isinstance(make_recorder("arecord"), ArecordRecorder)

    def test_unknown_name_exits(self):
        from asr2clip.recorders import make_recorder
        with pytest.raises(SystemExit):
            make_recorder("nonexistent")


# ---------------------------------------------------------------------------
# probe_available
# ---------------------------------------------------------------------------

class TestProbeAvailable:
    def test_returns_both_when_all_available(self):
        from asr2clip.recorders import ArecordRecorder, SounddeviceRecorder, probe_available
        with patch.object(SounddeviceRecorder, "is_available", return_value=True), \
             patch.object(ArecordRecorder, "is_available", return_value=True):
            result = probe_available()
        assert result == ["sounddevice", "arecord"]

    def test_returns_only_sounddevice(self):
        from asr2clip.recorders import ArecordRecorder, SounddeviceRecorder, probe_available
        with patch.object(SounddeviceRecorder, "is_available", return_value=True), \
             patch.object(ArecordRecorder, "is_available", return_value=False):
            assert probe_available() == ["sounddevice"]

    def test_returns_empty_when_none_available(self):
        from asr2clip.recorders import ArecordRecorder, SounddeviceRecorder, probe_available
        with patch.object(SounddeviceRecorder, "is_available", return_value=False), \
             patch.object(ArecordRecorder, "is_available", return_value=False):
            assert probe_available() == []
