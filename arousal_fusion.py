"""
arousal_fusion.py
-----------------
Combines 5 physiological + behavioral signals into a single 0-100 arousal score.

CHANNELS (5 total):
    hr_bpm           -- heart rate (rPPG forehead)
    rmssd_ms         -- HRV (derived from rPPG beat intervals)
    brpm             -- breathing rate (chest brightness)
    motion_score     -- head jitter + facial frame-diff
    blink_rate_bpm   -- blinks per minute (EAR-based)

How it works:
    1. CALIBRATION (first 90s): record baseline mean/std per channel.
    2. SCORING: each channel becomes z = (current - mean) / std * direction.
    3. Combine z-scores with weights, sigmoid -> 0..100, smooth.

Weights:
    HR              40%
    HRV             25%
    Breathing       20%
    Motion          10%
    Blink            5%
"""

import time
import math
import numpy as np
from collections import defaultdict, deque


CHANNELS = [
    "hr_bpm",
    "brpm",
    "rmssd_ms",
    "motion_score",
    "blink_rate_bpm",
]

# +1 = higher value means more arousal
# -1 = lower value means more arousal (HRV)
DIRECTION = {
    "hr_bpm": +1,
    "brpm": +1,
    "rmssd_ms": -1,
    "motion_score": +1,
    "blink_rate_bpm": +1,
}

DEFAULT_WEIGHTS = {
    "hr_bpm":         0.40,
    "rmssd_ms":       0.25,
    "brpm":           0.20,
    "motion_score":   0.10,
    "blink_rate_bpm": 0.05,
}


class ArousalFusion:
    def __init__(self, calibration_seconds=90, weights=None):
        self.calibration_seconds = calibration_seconds
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.start_time = None
        self.calibration_samples = defaultdict(list)
        self.baseline_mean = {}
        self.baseline_std = {}
        self.calibrated = False
        self.score_history = deque(maxlen=15)

    def reset(self):
        self.start_time = None
        self.calibration_samples.clear()
        self.baseline_mean.clear()
        self.baseline_std.clear()
        self.calibrated = False
        self.score_history.clear()

    def _min_std_for(self, ch):
        return {
            "hr_bpm": 3.0,
            "brpm": 1.5,
            "rmssd_ms": 8.0,
            "motion_score": 0.05,
            "blink_rate_bpm": 3.0,
        }.get(ch, 1.0)

    def _finalize_calibration(self):
        for ch in CHANNELS:
            samples = [s for s in self.calibration_samples[ch] if s is not None]
            if len(samples) >= 5:
                self.baseline_mean[ch] = float(np.mean(samples))
                std = float(np.std(samples))
                self.baseline_std[ch] = max(std, self._min_std_for(ch))
            else:
                self.baseline_mean[ch] = None
                self.baseline_std[ch] = None
        self.calibrated = True

    def update(self, inputs):
        """
        inputs: dict with any of these keys (None or missing = skip channel):
            hr_bpm, hr_quality
            brpm, br_quality
            rmssd_ms
            motion_score
            blink_rate_bpm
        """
        now = time.time()
        if self.start_time is None:
            self.start_time = now
        elapsed = now - self.start_time

        if not self.calibrated:
            for ch in CHANNELS:
                v = inputs.get(ch)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    self.calibration_samples[ch].append(v)

            progress = min(1.0, elapsed / self.calibration_seconds)
            if elapsed >= self.calibration_seconds:
                self._finalize_calibration()
                return self._make_live_state(inputs, progress=1.0)

            return {
                "phase": "calibrating",
                "calibration_progress": progress,
                "elapsed_seconds": elapsed,
                "arousal_score": None,
                "baseline": None,
                "deltas": {},
                "channel_contributions": {},
                "z_scores": {},
            }

        return self._make_live_state(inputs, progress=1.0)

    def _make_live_state(self, inputs, progress):
        z_scores = {}
        contributions = {}
        deltas = {}
        used_weight_total = 0.0
        weighted_z_sum = 0.0

        hr_q = inputs.get("hr_quality", 1.0) or 0.0
        br_q = inputs.get("br_quality", 1.0) or 0.0

        quality_mult = {
            "hr_bpm":         max(0.0, min(1.0, hr_q)),
            "rmssd_ms":       max(0.0, min(1.0, hr_q)),
            "brpm":           max(0.0, min(1.0, br_q)),
            "motion_score":   1.0,
            "blink_rate_bpm": 1.0,
        }

        for ch in CHANNELS:
            v = inputs.get(ch)
            base_mu = self.baseline_mean.get(ch)
            base_sd = self.baseline_std.get(ch)
            if v is None or base_mu is None or base_sd is None or base_sd == 0:
                continue
            z = (v - base_mu) / base_sd * DIRECTION[ch]
            z_scores[ch] = z
            deltas[ch] = v - base_mu

            w = self.weights.get(ch, 0.0) * quality_mult[ch]
            if w > 0:
                weighted_z_sum += z * w
                used_weight_total += w
                contributions[ch] = z * w

        if used_weight_total <= 0:
            return {
                "phase": "live",
                "calibration_progress": progress,
                "arousal_score": None,
                "baseline": dict(self.baseline_mean),
                "deltas": deltas,
                "channel_contributions": {},
                "z_scores": {},
            }

        weighted_z = weighted_z_sum / used_weight_total
        score = 100.0 / (1.0 + math.exp(-weighted_z))

        self.score_history.append(score)
        smooth_score = float(np.mean(self.score_history))

        return {
            "phase": "live",
            "calibration_progress": progress,
            "arousal_score": smooth_score,
            "weighted_z": weighted_z,
            "baseline": dict(self.baseline_mean),
            "deltas": deltas,
            "channel_contributions": contributions,
            "z_scores": z_scores,
        }
