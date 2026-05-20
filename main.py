"""
main.py
-------
Live demo: webcam in, all metrics overlaid, arousal score on screen.

Press 'q' to quit.
Press 'r' to reset calibration / baseline.
"""

import cv2
import time

from face_tracker import FaceTracker
from rppg_engine import RPPGEngine
from breathing_engine import BreathingEngine
from blink_engine import BlinkEngine
from motion_engine import MotionEngine
from hrv_engine import HRVEngine
from arousal_fusion import ArousalFusion


CALIBRATION_SECONDS = 90


def draw_text(img, text, pos, color=(255, 255, 255), scale=0.6, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_roi(img, roi, color, label=None):
    if roi is None:
        return
    x, y, w, h = roi
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
    if label:
        draw_text(img, label, (x, max(0, y - 5)), color, 0.45, 1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam")
        return

    tracker = FaceTracker()
    rppg = RPPGEngine()
    breathing = BreathingEngine()
    blink = BlinkEngine()
    motion = MotionEngine()
    hrv = HRVEngine()
    fusion = ArousalFusion(calibration_seconds=CALIBRATION_SECONDS)

    print("Starting webcam. Press 'q' to quit, 'r' to reset baseline.")
    print(f"Calibration baseline over first {CALIBRATION_SECONDS}s -- sit still and breathe normally.")

    frame_count = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        tdata = tracker.update(frame)
        rppg_out = rppg.update(frame, tdata)
        br_out = breathing.update(frame, tdata)
        blink_out = blink.update(frame, tdata)
        motion_out = motion.update(frame, tdata)
        hrv_out = hrv.update(rppg_out.get("peak_times", []))

        fusion_state = fusion.update({
            "hr_bpm": rppg_out.get("bpm"),
            "hr_quality": rppg_out.get("signal_quality"),
            "brpm": br_out.get("brpm"),
            "br_quality": br_out.get("signal_quality"),
            "rmssd_ms": hrv_out.get("rmssd_ms"),
            "motion_score": motion_out.get("motion_score"),
            "blink_rate_bpm": blink_out.get("blink_rate_bpm"),
        })

        draw_roi(display, tdata.get("forehead_roi"), (0, 255, 255), "rPPG")
        draw_roi(display, tdata.get("chest_roi"), (255, 200, 0), "breathing")
        if tdata.get("left_eye_pts") is not None:
            cv2.polylines(display, [tdata["left_eye_pts"]], True, (0, 200, 255), 1)
        if tdata.get("right_eye_pts") is not None:
            cv2.polylines(display, [tdata["right_eye_pts"]], True, (0, 200, 255), 1)

        y0 = 30
        if fusion_state["phase"] == "calibrating":
            prog = fusion_state["calibration_progress"]
            remaining = int(CALIBRATION_SECONDS * (1 - prog))
            draw_text(display, f"CALIBRATING BASELINE  {remaining}s left",
                      (10, y0), (0, 255, 255), 0.7, 2)
            bar_x, bar_y, bar_w, bar_h = 10, y0 + 10, 300, 12
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                          (100, 100, 100), 1)
            cv2.rectangle(display, (bar_x, bar_y),
                          (bar_x + int(bar_w * prog), bar_y + bar_h),
                          (0, 255, 255), -1)
        else:
            score = fusion_state["arousal_score"]
            if score is not None:
                if score < 40:
                    color = (0, 255, 0)
                elif score < 65:
                    color = (0, 220, 220)
                else:
                    color = (0, 80, 255)
                draw_text(display, f"AROUSAL: {score:5.1f} / 100",
                          (10, y0), color, 0.9, 2)
                bar_x, bar_y, bar_w, bar_h = 10, y0 + 10, 300, 14
                cv2.rectangle(display, (bar_x, bar_y),
                              (bar_x + bar_w, bar_y + bar_h),
                              (100, 100, 100), 1)
                cv2.rectangle(display, (bar_x, bar_y),
                              (bar_x + int(bar_w * (score / 100.0)), bar_y + bar_h),
                              color, -1)
            else:
                draw_text(display, "AROUSAL: warming up...", (10, y0),
                          (200, 200, 200), 0.7, 2)

        y = y0 + 50
        line_h = 22

        def fmt(v, suffix="", n=1):
            if v is None:
                return "--"
            return f"{v:.{n}f}{suffix}"

        draw_text(display, f"HR:        {fmt(rppg_out['bpm'], ' bpm')}     (q={fmt(rppg_out['signal_quality'], '', 2)})",
                  (10, y)); y += line_h
        draw_text(display, f"Breathing: {fmt(br_out['brpm'], ' /min')}  (q={fmt(br_out['signal_quality'], '', 2)})",
                  (10, y)); y += line_h
        draw_text(display, f"HRV RMSSD: {fmt(hrv_out['rmssd_ms'], ' ms')}    (n={hrv_out['n_beats']})",
                  (10, y)); y += line_h
        draw_text(display, f"Blinks:    {fmt(blink_out['blink_rate_bpm'], ' /min')}  (total={blink_out['total_blinks']})",
                  (10, y)); y += line_h
        draw_text(display, f"Motion:    {fmt(motion_out['motion_score'], '', 2)}",
                  (10, y)); y += line_h

        face_status = "FACE OK" if tdata["face_found"] else "NO FACE DETECTED"
        face_color = (0, 255, 0) if tdata["face_found"] else (0, 0, 255)
        draw_text(display, face_status, (10, display.shape[0] - 15),
                  face_color, 0.6, 2)

        frame_count += 1
        elapsed = time.time() - t_start
        fps = frame_count / max(elapsed, 1e-3)
        draw_text(display, f"{fps:.1f} fps", (display.shape[1] - 90, display.shape[0] - 15),
                  (180, 180, 180), 0.5, 1)

        cv2.imshow("Physiological Arousal System", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Resetting baseline calibration.")
            fusion.reset()

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()
