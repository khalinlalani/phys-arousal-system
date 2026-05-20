"""
blink_engine.py
---------------
Blink rate via Eye Aspect Ratio (EAR).

EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
Where p1..p6 are the 6 standard eye landmarks (outer, top1, top2, inner,
bottom2, bottom1). When the eye closes, EAR drops sharply.

We detect a blink as: EAR crosses below THRESH and then back above.
Public API:
    engine = BlinkEngine()
    out = engine.update(frame, tracker_data)
    # out: { 'blink_rate_bpm': float|None, 'ear': float|None, 'total_blinks': int }
    # blink_rate_bpm = blinks per minute (running, over last ~60s)
"""

import time
import numpy as np
from collections import deque

EAR_THRESH = 0.21               # below this = eye considered closed
CONSEC_FRAMES = 2               # require N consecutive low frames to count as a blink
WINDOW_SECONDS = 60.0           # rate computed over this trailing window


def _ear_from_pts(pts):
    """pts: np.array shape (6,2) in [outer, top1, top2, inner, bottom2, bottom1] order."""
    if pts is None or len(pts) != 6:
        return None
    p1, p2, p3, p4, p5, p6 = pts
    vert1 = np.linalg.norm(p2 - p6)
    vert2 = np.linalg.norm(p3 - p5)
    horiz = np.linalg.norm(p1 - p4) + 1e-8
    return float((vert1 + vert2) / (2.0 * horiz))


class BlinkEngine:
    def __init__(self):
        self.below_count = 0
        self.total_blinks = 0
        self.blink_times = deque()      # timestamps of blinks
        self.last_ear = None

    def update(self, frame, tracker_data):
        now = time.time()
        left = tracker_data.get("left_eye_pts")
        right = tracker_data.get("right_eye_pts")

        ear_l = _ear_from_pts(left)
        ear_r = _ear_from_pts(right)

        if ear_l is None and ear_r is None:
            ear = None
        elif ear_l is None:
            ear = ear_r
        elif ear_r is None:
            ear = ear_l
        else:
            ear = (ear_l + ear_r) / 2.0

        if ear is not None:
            self.last_ear = ear
            if ear < EAR_THRESH:
                self.below_count += 1
            else:
                if self.below_count >= CONSEC_FRAMES:
                    self.total_blinks += 1
                    self.blink_times.append(now)
                self.below_count = 0

        # Drop blinks outside trailing window
        while self.blink_times and (now - self.blink_times[0]) > WINDOW_SECONDS:
            self.blink_times.popleft()

        # Blinks per minute over the actual time we've been observing
        # (only meaningful after we've seen >= ~20s of data so it isn't spiky)
        rate = None
        if len(self.blink_times) > 0:
            window_used = min(WINDOW_SECONDS,
                              now - self.blink_times[0] + 1e-3)
            if window_used > 5.0:  # need at least a few seconds
                rate = float(len(self.blink_times) * 60.0 / window_used)

        return {
            "blink_rate_bpm": rate,
            "ear": self.last_ear,
            "total_blinks": self.total_blinks,
        }
