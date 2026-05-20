"""
rppg_engine.py
--------------
Heart rate estimation via remote photoplethysmography (rPPG).

Upgraded from the original (which used a fixed center rectangle).
Now uses real forehead + cheek ROIs from face_tracker.

Public API:
    engine = RPPGEngine()
    out = engine.update(frame, tracker_data)
    # out is a dict, never raises:
    #   bpm           float or None
    #   signal_quality 0..1 (rough SNR proxy)
    #   peak_times    list[float] -- timestamps of detected beats
                                   (used by hrv_engine)
"""

import time
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, detrend, find_peaks

FS_TARGET = 30.0                # assumed/target frame rate
BPM_MIN, BPM_MAX = 42, 180
LOW_HZ, HIGH_HZ = 0.7, 3.0      # heart rate band
BUFFER_SECONDS = 20
MIN_SECONDS_FOR_BPM = 8


def _butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype="band")


class RPPGEngine:
    def __init__(self):
        self.green_buffer = deque(maxlen=int(BUFFER_SECONDS * FS_TARGET))
        self.time_buffer = deque(maxlen=int(BUFFER_SECONDS * FS_TARGET))
        self.bpm_history = deque(maxlen=5)
        self.last_bpm = None
        self.last_quality = 0.0
        self.last_peak_times = []
        self.last_estimate_time = 0.0
        self._b, self._a = _butter_bandpass(LOW_HZ, HIGH_HZ, FS_TARGET)

    def _extract_green_mean(self, frame, roi):
        if roi is None:
            return None
        x, y, w, h = roi
        if w <= 2 or h <= 2:
            return None
        patch = frame[y:y + h, x:x + w]
        if patch.size == 0:
            return None
        # Mean of green channel -- the channel most modulated by blood volume.
        return float(patch[:, :, 1].mean())

    def update(self, frame, tracker_data):
        """
        frame: BGR ndarray
        tracker_data: dict from FaceTracker.update()
        """
        now = time.time()
        out = {
            "bpm": self.last_bpm,
            "signal_quality": self.last_quality,
            "peak_times": list(self.last_peak_times),
        }

        # Prefer forehead, fall back to cheeks
        roi = tracker_data.get("forehead_roi") or tracker_data.get("cheek_roi")
        g = self._extract_green_mean(frame, roi)

        if g is None:
            # No face this frame; just return last value
            return out

        self.green_buffer.append(g)
        self.time_buffer.append(now)

        if len(self.green_buffer) < int(FS_TARGET * MIN_SECONDS_FOR_BPM):
            return out

        # Only re-estimate once a second; this is what the original did too.
        if now - self.last_estimate_time < 1.0:
            return out
        self.last_estimate_time = now

        raw = np.array(self.green_buffer, dtype=np.float64)
        sig = detrend(raw)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)

        try:
            filt = filtfilt(self._b, self._a, sig)
        except Exception:
            return out

        # ---- FFT ----
        n = len(filt)
        freqs = np.fft.rfftfreq(n, d=1.0 / FS_TARGET)
        spec = np.abs(np.fft.rfft(filt))
        bpm_axis = freqs * 60.0

        mask = (bpm_axis >= BPM_MIN) & (bpm_axis <= BPM_MAX)
        if not np.any(mask):
            return out

        band_spec = spec[mask]
        peak_idx = int(np.argmax(band_spec))
        est_bpm = float(bpm_axis[mask][peak_idx])

        # ---- Signal quality (rough): peak energy / total band energy ----
        peak_energy = band_spec[peak_idx]
        total_energy = band_spec.sum() + 1e-8
        quality = float(min(1.0, peak_energy / total_energy * 5.0))

        # ---- Smoothing ----
        self.bpm_history.append(est_bpm)
        smooth_bpm = float(np.mean(self.bpm_history))

        # ---- Peak detection on filtered signal (for HRV) ----
        # Convert filtered signal samples back to wall-clock times.
        peak_distance = max(1, int(FS_TARGET * 60.0 / BPM_MAX))
        peaks, _ = find_peaks(filt, distance=peak_distance)
        times_arr = np.array(self.time_buffer)
        if len(peaks) > 0 and len(times_arr) == len(filt):
            self.last_peak_times = times_arr[peaks].tolist()
        else:
            self.last_peak_times = []

        self.last_bpm = smooth_bpm
        self.last_quality = quality

        out["bpm"] = smooth_bpm
        out["signal_quality"] = quality
        out["peak_times"] = list(self.last_peak_times)
        return out
