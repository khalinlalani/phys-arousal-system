"""
face_tracker.py
---------------
Central face and pose tracker using MediaPipe FaceMesh + Pose.

Every other engine pulls ROIs from THIS module. Running MediaPipe once
per frame saves CPU vs running it inside every engine.

Public API:
    tracker = FaceTracker()
    data = tracker.update(frame)   # returns dict, never raises

Returned dict keys (always present, values may be None if no face/pose):
    face_found     bool
    forehead_roi   (x, y, w, h) or None
    cheek_roi      (x, y, w, h) or None
    left_eye_pts   np.array shape (6,2) or None     (for EAR/blink)
    right_eye_pts  np.array shape (6,2) or None
    nose_tip       (x, y) or None
    face_bbox      (x, y, w, h) or None
    chest_roi      (x, y, w, h) or None
"""

import cv2
import numpy as np
import mediapipe as mp


FOREHEAD_IDX = [10, 67, 297, 338, 109, 151]
LEFT_CHEEK_IDX = [50, 101, 118, 117, 123, 147]
RIGHT_CHEEK_IDX = [280, 330, 347, 346, 352, 376]

LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

NOSE_TIP_IDX = 1

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


class FaceTracker:
    def __init__(self,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 enable_pose=False):
        self.enable_pose = enable_pose

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        if enable_pose:
            self.pose = mp.solutions.pose.Pose(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.pose = None

    def _lm_to_px(self, landmarks, indices, w, h):
        return np.array(
            [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices],
            dtype=np.int32,
        )

    def _lm_single(self, landmarks, idx, w, h):
        lm = landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))

    def _bbox(self, pts, w, h, pad=0):
        x1 = max(0, int(pts[:, 0].min()) - pad)
        y1 = max(0, int(pts[:, 1].min()) - pad)
        x2 = min(w, int(pts[:, 0].max()) + pad)
        y2 = min(h, int(pts[:, 1].max()) + pad)
        return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    def update(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        result = {
            "face_found": False,
            "forehead_roi": None,
            "cheek_roi": None,
            "left_eye_pts": None,
            "right_eye_pts": None,
            "nose_tip": None,
            "face_bbox": None,
            "chest_roi": None,
        }

        fm_result = self.face_mesh.process(rgb)
        if fm_result.multi_face_landmarks:
            lms = fm_result.multi_face_landmarks[0].landmark
            result["face_found"] = True

            forehead_pts = self._lm_to_px(lms, FOREHEAD_IDX, w, h)
            result["forehead_roi"] = self._bbox(forehead_pts, w, h)

            left_cheek_pts = self._lm_to_px(lms, LEFT_CHEEK_IDX, w, h)
            right_cheek_pts = self._lm_to_px(lms, RIGHT_CHEEK_IDX, w, h)
            cheek_all = np.vstack([left_cheek_pts, right_cheek_pts])
            result["cheek_roi"] = self._bbox(cheek_all, w, h)

            result["left_eye_pts"] = self._lm_to_px(lms, LEFT_EYE_EAR_IDX, w, h)
            result["right_eye_pts"] = self._lm_to_px(lms, RIGHT_EYE_EAR_IDX, w, h)

            result["nose_tip"] = self._lm_single(lms, NOSE_TIP_IDX, w, h)

            all_pts = np.array(
                [(int(lm.x * w), int(lm.y * h)) for lm in lms], dtype=np.int32
            )
            result["face_bbox"] = self._bbox(all_pts, w, h)

        if self.enable_pose:
            pose_result = self.pose.process(rgb)
            if pose_result.pose_landmarks:
                plms = pose_result.pose_landmarks.landmark
                ls = plms[LEFT_SHOULDER]
                rs = plms[RIGHT_SHOULDER]

                if ls.visibility > 0.3 and rs.visibility > 0.3:
                    ls_x, ls_y = int(ls.x * w), int(ls.y * h)
                    rs_x, rs_y = int(rs.x * w), int(rs.y * h)

                    x1 = max(0, min(ls_x, rs_x))
                    x2 = min(w, max(ls_x, rs_x))
                    y1 = max(0, max(ls_y, rs_y))
                    y2 = min(h, y1 + int(0.25 * h))
                    if x2 - x1 > 10 and y2 - y1 > 10:
                        result["chest_roi"] = (x1, y1, x2 - x1, y2 - y1)

        if result["chest_roi"] is None and result["face_bbox"] is not None:
            fx, fy, fw, fh = result["face_bbox"]
            cy1 = min(h - 1, fy + fh + int(0.1 * fh))
            cy2 = min(h, cy1 + int(1.5 * fh))
            cx1 = max(0, fx - int(0.3 * fw))
            cx2 = min(w, fx + fw + int(0.3 * fw))
            if cy2 - cy1 > 10 and cx2 - cx1 > 10:
                result["chest_roi"] = (cx1, cy1, cx2 - cx1, cy2 - cy1)

        return result

    def close(self):
        try:
            self.face_mesh.close()
        except Exception:
            pass
        if self.pose is not None:
            try:
                self.pose.close()
            except Exception:
                pass
