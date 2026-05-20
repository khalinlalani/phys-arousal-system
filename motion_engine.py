"""
motion_engine.py
----------------
Quantifies head motion / fidgeting.

Two signals combined:
    1. Nose-tip position variance (head jitter, in normalized pixels/sec)
    2. Frame-difference magnitude inside face bbox (micro-movements)

Output:
    motion_score: 0..1, rolling average over last ~5 seconds.
                  0 = stone still, 1 = lots of movement.
"""

import time
import numpy as np
import cv2
from collections import deque

WINDOW_SECONDS = 5.0
MAX_HISTORY = 300  # ~10s at 30fps


class MotionEngine:
    def __init__(self):
        self.nose_history = deque(maxlen=MAX_HISTORY)   # (t, x, y)
        self.frame_diff_history = deque(maxlen=MAX_HISTORY)  # (t, value)
        self.prev_face_gray = None
        self.prev_face_bbox = None

    def update(self, frame, tracker_data):
        now = time.time()

        # ---- 1. Head jitter via nose tip ----
        nose = tracker_data.get("nose_tip")
        if nose is not None:
            self.nose_history.append((now, nose[0], nose[1]))

        # ---- 2. Frame-difference inside face bbox ----
        bbox = tracker_data.get("face_bbox")
        diff_val = 0.0
        if bbox is not None:
            x, y, w, h = bbox
            face_patch = frame[y:y + h, x:x + w]
            if face_patch.size > 0:
                gray = cv2.cvtColor(face_patch, cv2.COLOR_BGR2GRAY)
                # Resize to a fixed size so comparison works across frames
                # even if face bbox size changes slightly
                gray_small = cv2.resize(gray, (64, 64))
                if self.prev_face_gray is not None:
                    d = cv2.absdiff(gray_small, self.prev_face_gray)
                    # Normalize by 255 so this lives in 0..1
                    diff_val = float(d.mean()) / 255.0
                self.prev_face_gray = gray_small
                self.frame_diff_history.append((now, diff_val))

        # Trim history to window
        while self.nose_history and (now - self.nose_history[0][0]) > WINDOW_SECONDS:
            self.nose_history.popleft()
        while self.frame_diff_history and (now - self.frame_diff_history[0][0]) > WINDOW_SECONDS:
            self.frame_diff_history.popleft()

        # ---- Compute jitter (std dev of nose position) ----
        jitter = 0.0
        if len(self.nose_history) >= 5:
            xs = np.array([n[1] for n in self.nose_history])
            ys = np.array([n[2] for n in self.nose_history])
            # Frame size from current frame
            h_f, w_f = frame.shape[:2]
            jitter = float((xs.std() / w_f + ys.std() / h_f))
            # Empirically this is roughly 0 (still) up to ~0.05+ (lots of moving)
            jitter = min(1.0, jitter / 0.05)

        # ---- Compute mean frame-diff over window ----
        mean_diff = 0.0
        if self.frame_diff_history:
            vals = np.array([d[1] for d in self.frame_diff_history])
            mean_diff = float(vals.mean())
            # Empirically ~0 still, ~0.03+ active
            mean_diff = min(1.0, mean_diff / 0.03)

        # Combined motion score: weighted blend
        motion_score = float(0.6 * jitter + 0.4 * mean_diff)
        motion_score = max(0.0, min(1.0, motion_score))

        return {
            "motion_score": motion_score,
            "head_jitter": jitter,
            "frame_diff": mean_diff,
        }
