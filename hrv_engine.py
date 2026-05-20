"""
hrv_engine.py
-------------
HRV (Heart Rate Variability) proxy via RMSSD from rPPG inter-beat intervals.

RMSSD = root mean square of successive differences between adjacent RR intervals.
Lower RMSSD generally correlates with higher sympathetic arousal (stress).

IMPORTANT honesty note: webcam-derived HRV is noisy compared to a chest strap
or finger PPG. We compute it because it's still useful as a *relative* signal
(within a single session, against baseline), and we feed signal_quality from
rppg_engine into the fusion layer so we can down-weight HRV when the underlying
pulse signal is weak.

Public API:
    engine = HRVEngine()
    out = engine.update(rppg_peak_times)
    # out: { 'rmssd_ms': float|None, 'sdnn_ms': float|None, 'n_beats': int }
"""

import numpy as np
from collections import deque

MIN_RR_S = 0.30   # ignore intervals shorter than this (200 BPM upper sanity)
MAX_RR_S = 1.5    # or longer than this (40 BPM lower sanity)
MAX_BEATS = 60    # keep last N RR intervals for rolling estimate


class HRVEngine:
    def __init__(self):
        self.rr_intervals_ms = deque(maxlen=MAX_BEATS)
        self.last_peak_seen = None
        self.last_rmssd = None
        self.last_sdnn = None

    def update(self, peak_times):
        """
        peak_times: list of wall-clock timestamps (seconds) of detected beats
                    from RPPGEngine.
        """
        out = {
            "rmssd_ms": self.last_rmssd,
            "sdnn_ms": self.last_sdnn,
            "n_beats": len(self.rr_intervals_ms),
        }

        if not peak_times or len(peak_times) < 2:
            return out

        # Process only peaks newer than what we've already seen
        new_peaks = peak_times
        if self.last_peak_seen is not None:
            new_peaks = [t for t in peak_times if t > self.last_peak_seen]

        # We need at least 2 peaks (or 1 new + the last we saw) to make an interval
        all_recent = peak_times[-min(len(peak_times), MAX_BEATS + 1):]
        # Recompute RR intervals from a fresh slice to avoid drift
        self.rr_intervals_ms.clear()
        for i in range(1, len(all_recent)):
            dt = all_recent[i] - all_recent[i - 1]
            if MIN_RR_S <= dt <= MAX_RR_S:
                self.rr_intervals_ms.append(dt * 1000.0)

        self.last_peak_seen = peak_times[-1]

        if len(self.rr_intervals_ms) < 5:
            return out

        rr = np.array(self.rr_intervals_ms, dtype=np.float64)
        # RMSSD
        diffs = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(diffs ** 2)))
        # SDNN
        sdnn = float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0

        self.last_rmssd = rmssd
        self.last_sdnn = sdnn

        out["rmssd_ms"] = rmssd
        out["sdnn_ms"] = sdnn
        out["n_beats"] = len(self.rr_intervals_ms)
        return out
