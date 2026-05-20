"""
breathing_engine.py
-------------------
Breathing rate from chest ROI brightness changes, live webcam.

Same pipeline shape as rppg_engine.py, just slower band:
    heart rate band: 0.7 - 3.0 Hz
    breathing band:  0.1 - 0.5 Hz   (6 - 30 breaths/min)

Public API:
    engine = BreathingEngine()
    out = engine.update(frame, tracker_data)
    # out: { 'brpm': float|None, 'signal_quality': 0..1 }
"""

import time
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, detrend

FS_TARGET = 30.0
BR_MIN, BR_MAX = 6.0, 30.0        # breaths per minute
LOW_HZ, HIGH_HZ = 0.1, 0.5
BUFFER_SECONDS = 30               # breathing needs longer window than HR
MIN_SECONDS_FOR_BR = 20


def _butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype="band")


class BreathingEngine:
    def __init__(self):
        self.buffer = deque(maxlen=int(BUFFER_SECONDS * FS_TARGET))
        self.brpm_history = deque(maxlen=5)
        self.last_brpm = None
        self.last_quality = 0.0
        self.last_estimate_time = 0.0
        self._b, self._a = _butter_bandpass(LOW_HZ, HIGH_HZ, FS_TARGET)

    def _chest_brightness(self, frame, roi):
        if roi is None:
            return None
        x, y, w, h = roi
        if w <= 2 or h <= 2:
            return None
        patch = frame[y:y + h, x:x + w]
        if patch.size == 0:
            return None
        # Grayscale mean. Breathing modulates shadows/shading on the chest;
        # we don't care about color here, just intensity.
        gray = patch.mean(axis=2) if patch.ndim == 3 else patch
        return float(gray.mean())

    def update(self, frame, tracker_data):
        now = time.time()
        out = {"brpm": self.last_brpm, "signal_quality": self.last_quality}

        chest = tracker_data.get("chest_roi")
        val = self._chest_brightness(frame, chest)
        if val is None:
            return out

        self.buffer.append(val)

        if len(self.buffer) < int(FS_TARGET * MIN_SECONDS_FOR_BR):
            return out

        if now - self.last_estimate_time < 1.5:
            return out
        self.last_estimate_time = now

        raw = np.array(self.buffer, dtype=np.float64)
        sig = detrend(raw)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)

        try:
            filt = filtfilt(self._b, self._a, sig)
        except Exception:
            return out

        n = len(filt)
        freqs = np.fft.rfftfreq(n, d=1.0 / FS_TARGET)
        spec = np.abs(np.fft.rfft(filt))
        brpm_axis = freqs * 60.0

        mask = (brpm_axis >= BR_MIN) & (brpm_axis <= BR_MAX)
        if not np.any(mask):
            return out

        band_spec = spec[mask]
        peak_idx = int(np.argmax(band_spec))
        est_brpm = float(brpm_axis[mask][peak_idx])

        peak_energy = band_spec[peak_idx]
        total_energy = band_spec.sum() + 1e-8
        quality = float(min(1.0, peak_energy / total_energy * 5.0))

        self.brpm_history.append(est_brpm)
        smooth_brpm = float(np.mean(self.brpm_history))

        self.last_brpm = smooth_brpm
        self.last_quality = quality

        out["brpm"] = smooth_brpm
        out["signal_quality"] = quality
        return out
