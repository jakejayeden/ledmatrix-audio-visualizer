"""LEDMatrix Audio Visualizer plugin.

Captures mono audio from a USB/default input with sounddevice, maintains a
small rolling sample buffer, and renders a logarithmic FFT spectrum to the
LEDMatrix PIL canvas.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw

from src.plugin_system.base_plugin import BasePlugin

try:
    import sounddevice as sd
except ImportError:  # Plugin stays loadable so the display service can report the problem.
    sd = None


class AudioVisualizerPlugin(BasePlugin):
    """Real-time microphone spectrum visualizer."""

    # Ask LEDMatrix to call display() on its high-FPS render loop.
    needs_high_fps = True

    def __init__(self, plugin_id, config, display_manager, cache_manager, plugin_manager):
        super().__init__(plugin_id, config, display_manager, cache_manager, plugin_manager)

        self._lock = threading.Lock()
        self._stream = None
        self._audio_buffer = np.zeros(4096, dtype=np.float32)
        self._buffer_write = 0
        self._buffer_filled = 0
        self._last_audio_active_at = float("-inf")
        self._current_rms = 0.0
        self._audio_error: Optional[str] = None
        self._warned_stream_error: Optional[str] = None
        self._levels = np.zeros(32, dtype=np.float32)
        self._peaks = np.zeros(32, dtype=np.float32)

        self._load_config(config)
        self._resize_analysis_state()
        self._resize_audio_buffer()
        self._start_audio_stream()

    # ------------------------------------------------------------------
    # Configuration / lifecycle
    # ------------------------------------------------------------------
    def _load_config(self, config):
        self.input_device = config.get("input_device")
        self.sensitivity = float(config.get("sensitivity", 1.0))
        self.hold_while_audio_active = bool(config.get("hold_while_audio_active", False))
        self.activity_threshold = float(config.get("activity_threshold", 0.025))
        self.release_delay_seconds = float(config.get("release_delay_seconds", 2.5))
        self.sample_rate = int(config.get("sample_rate", 44100))
        self.fft_size = int(config.get("fft_size", 2048))
        self.min_frequency = float(config.get("min_frequency", 60.0))
        self.max_frequency = float(config.get("max_frequency", 16000.0))
        self.bar_count = int(config.get("bar_count", 32))
        self.smoothing = float(config.get("smoothing", 0.65))
        self.peak_decay = float(config.get("peak_decay", 0.8))
        self.show_peaks = bool(config.get("show_peaks", True))
        self.color_low = tuple(config.get("color_low", [0, 255, 80]))
        self.color_high = tuple(config.get("color_high", [255, 30, 0]))
        self.display_duration = float(config.get("display_duration", 15.0))

    def on_config_change(self, new_config):
        old_audio_settings = (self.input_device, self.sample_rate, self.fft_size)
        super().on_config_change(new_config)
        self._load_config(self.config)
        self._resize_analysis_state()
        self._resize_audio_buffer()

        new_audio_settings = (self.input_device, self.sample_rate, self.fft_size)
        if new_audio_settings != old_audio_settings:
            self._restart_audio_stream()

    def on_enable(self):
        try:
            super().on_enable()
        except AttributeError:
            pass
        self._start_audio_stream()

    def on_disable(self):
        self._stop_audio_stream()
        try:
            super().on_disable()
        except AttributeError:
            pass

    def cleanup(self):
        self._stop_audio_stream()
        super().cleanup()

    def get_display_duration(self):
        return self.display_duration

    # ------------------------------------------------------------------
    # Live-priority integration
    # ------------------------------------------------------------------
    def has_live_priority(self):
        """The plugin's own checkbox controls whether audio may hold the screen."""
        return self.hold_while_audio_active

    def has_live_content(self):
        """Return true while sound is active, including the configured release tail."""
        if not self.hold_while_audio_active or self._stream is None:
            return False
        with self._lock:
            last_active = self._last_audio_active_at
        return (time.monotonic() - last_active) <= self.release_delay_seconds

    def get_live_modes(self):
        return ["audio-visualizer"]

    # ------------------------------------------------------------------
    # Audio capture
    # ------------------------------------------------------------------
    def _normalised_device(self) -> Any:
        device = self.input_device
        if device in (None, "", "default"):
            return None
        if isinstance(device, str):
            stripped = device.strip()
            if stripped.isdigit():
                return int(stripped)
            return stripped
        return device

    def _resize_audio_buffer(self):
        wanted = max(self.fft_size * 4, 4096)
        with self._lock:
            if len(self._audio_buffer) == wanted:
                return
            self._audio_buffer = np.zeros(wanted, dtype=np.float32)
            self._buffer_write = 0
            self._buffer_filled = 0

    def _resize_analysis_state(self):
        if len(self._levels) != self.bar_count:
            self._levels = np.zeros(self.bar_count, dtype=np.float32)
            self._peaks = np.zeros(self.bar_count, dtype=np.float32)

    def _start_audio_stream(self):
        if self._stream is not None:
            return
        if sd is None:
            self._audio_error = "Python package 'sounddevice' is not installed"
            self.logger.error(self._audio_error)
            return

        try:
            self._audio_error = None
            self._stream = sd.InputStream(
                device=self._normalised_device(),
                channels=1,
                samplerate=self.sample_rate,
                dtype="float32",
                blocksize=512,
                callback=self._audio_callback,
            )
            self._stream.start()
            self.logger.info(
                "Audio input started: device=%r sample_rate=%d",
                self._normalised_device(), self.sample_rate,
            )
        except Exception as exc:  # PortAudio/device errors must not kill LEDMatrix.
            self._stream = None
            self._audio_error = str(exc)
            self.logger.error("Could not start audio input: %s", exc, exc_info=True)

    def _stop_audio_stream(self):
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            self.logger.debug("Error stopping audio stream", exc_info=True)
        try:
            stream.close()
        except Exception:
            self.logger.debug("Error closing audio stream", exc_info=True)

    def _restart_audio_stream(self):
        self._stop_audio_stream()
        self._start_audio_stream()

    def _audio_callback(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            status_text = str(status)
            if status_text != self._warned_stream_error:
                self._warned_stream_error = status_text
                self.logger.warning("Audio input status: %s", status_text)

        samples = np.asarray(indata[:, 0], dtype=np.float32)
        if samples.size == 0:
            return

        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float32))))
        now = time.monotonic()

        with self._lock:
            self._current_rms = rms
            if rms >= self.activity_threshold:
                self._last_audio_active_at = now

            capacity = len(self._audio_buffer)
            count = min(samples.size, capacity)
            samples = samples[-count:]
            end = self._buffer_write + count
            if end <= capacity:
                self._audio_buffer[self._buffer_write:end] = samples
            else:
                first = capacity - self._buffer_write
                self._audio_buffer[self._buffer_write:] = samples[:first]
                self._audio_buffer[:count - first] = samples[first:]
            self._buffer_write = (self._buffer_write + count) % capacity
            self._buffer_filled = min(capacity, self._buffer_filled + count)

    def _latest_samples(self) -> np.ndarray:
        with self._lock:
            available = min(self._buffer_filled, self.fft_size)
            if available <= 0:
                return np.zeros(self.fft_size, dtype=np.float32)

            result = np.zeros(self.fft_size, dtype=np.float32)
            start_out = self.fft_size - available
            start = (self._buffer_write - available) % len(self._audio_buffer)
            end = start + available
            if end <= len(self._audio_buffer):
                result[start_out:] = self._audio_buffer[start:end]
            else:
                first = len(self._audio_buffer) - start
                result[start_out:start_out + first] = self._audio_buffer[start:]
                result[start_out + first:] = self._audio_buffer[:available - first]
            return result

    # ------------------------------------------------------------------
    # Spectrum analysis / rendering
    # ------------------------------------------------------------------
    def update(self):
        """No polling work is needed; PortAudio feeds the rolling buffer."""
        if self._stream is None and sd is not None:
            # A USB microphone may have appeared after startup.
            self._start_audio_stream()

    def _spectrum_levels(self) -> np.ndarray:
        samples = self._latest_samples()
        samples -= float(np.mean(samples))
        windowed = samples * np.hanning(len(samples)).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(windowed))
        spectrum /= max(1.0, len(samples) / 2.0)
        freqs = np.fft.rfftfreq(len(samples), d=1.0 / self.sample_rate)

        nyquist = self.sample_rate / 2.0
        low = max(20.0, min(self.min_frequency, nyquist - 1.0))
        high = max(low + 1.0, min(self.max_frequency, nyquist))
        edges = np.geomspace(low, high, self.bar_count + 1)

        raw = np.zeros(self.bar_count, dtype=np.float32)
        for i in range(self.bar_count):
            mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
            if np.any(mask):
                # RMS within each logarithmic band gives a less twitchy result than max().
                values = spectrum[mask]
                raw[i] = float(np.sqrt(np.mean(values * values)))

        gain_db = 20.0 * math.log10(max(0.01, self.sensitivity))
        db = 20.0 * np.log10(np.maximum(raw, 1e-7)) + gain_db
        # Map approximately -65 dBFS..0 dBFS to 0..1.
        normalised = np.clip((db + 65.0) / 65.0, 0.0, 1.0).astype(np.float32)

        alpha = np.clip(self.smoothing, 0.0, 0.95)
        self._levels = (alpha * self._levels) + ((1.0 - alpha) * normalised)
        return self._levels

    @staticmethod
    def _mix_color(low, high, amount):
        amount = max(0.0, min(1.0, float(amount)))
        return tuple(int(a + (b - a) * amount) for a, b in zip(low, high))

    def _render_frame(self) -> Image.Image:
        width = int(self.display_manager.width)
        height = int(self.display_manager.height)
        image = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image)

        if self._stream is None:
            # Simple visible fault indicator without depending on font assets.
            draw.rectangle((0, 0, width - 1, height - 1), outline=(100, 0, 0))
            draw.line((0, 0, width - 1, height - 1), fill=(255, 0, 0))
            draw.line((width - 1, 0, 0, height - 1), fill=(255, 0, 0))
            return image

        levels = self._spectrum_levels()
        usable_height = max(1, height - 1)
        slot_width = width / float(self.bar_count)

        for i, level in enumerate(levels):
            bar_height = max(0, min(usable_height, int(round(level * usable_height))))
            x0 = int(round(i * slot_width))
            x1 = int(round((i + 1) * slot_width)) - 1
            if x1 < x0:
                x1 = x0
            if x1 > x0:
                x1 -= 1  # one-pixel gap where panel width permits it

            if bar_height > 0:
                for y in range(bar_height):
                    fraction = y / max(1, usable_height - 1)
                    color = self._mix_color(self.color_low, self.color_high, fraction)
                    py = height - 1 - y
                    draw.line((x0, py, x1, py), fill=color)

            if self.show_peaks:
                target = float(bar_height)
                if target >= self._peaks[i]:
                    self._peaks[i] = target
                else:
                    self._peaks[i] = max(0.0, self._peaks[i] - self.peak_decay)
                peak_y = height - 1 - int(round(self._peaks[i]))
                if 0 <= peak_y < height:
                    draw.line((x0, peak_y, x1, peak_y), fill=self.color_high)

        return image

    def display(self, force_clear=False):  # noqa: ARG002
        try:
            image = self._render_frame()
            self.display_manager.image = image
            self.display_manager.draw = ImageDraw.Draw(self.display_manager.image)
            self.display_manager.update_display()
        except Exception as exc:
            self.logger.error("Audio visualizer render failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Validation / diagnostics
    # ------------------------------------------------------------------
    def validate_config(self):
        if not super().validate_config():
            return False
        if self.min_frequency >= self.max_frequency:
            self.logger.error("min_frequency must be lower than max_frequency")
            return False
        if self.max_frequency > self.sample_rate / 2.0:
            self.logger.warning(
                "max_frequency %.0f is above Nyquist %.0f and will be clamped",
                self.max_frequency, self.sample_rate / 2.0,
            )
        return True

    def get_info(self):
        info = super().get_info()
        with self._lock:
            rms = self._current_rms
        info.update({
            "audio_input_running": self._stream is not None,
            "audio_error": self._audio_error,
            "rms_level": round(rms, 5),
            "audio_active": self.has_live_content(),
            "input_device": self.input_device,
            "sample_rate": self.sample_rate,
            "fft_size": self.fft_size,
        })
        return info
