# Contactless Physiological Arousal Sensing System

TKS Create project. Real-time webcam-based estimation of physiological arousal from facial video — no contact sensors, no wearables. Built on the rPPG and respiration replicates documented at https://medium.com/@khalin.lalani.

## What it measures (5 channels)

| Signal | Method | Source ROI |
|---|---|---|
| **Heart rate** | rPPG, green channel + Butterworth + FFT | Forehead |
| **Breathing rate** | Chest brightness, slow bandpass + FFT | Upper chest |
| **HRV (RMSSD)** | Peak detection on rPPG signal, successive RR intervals | Derived from forehead |
| **Blink rate** | Eye Aspect Ratio (EAR) | Eye landmarks |
| **Motion** | Head jitter + frame-difference | Face bbox + nose tip |

All five feed into a single **arousal score (0–100)** that's z-scored against the user's own **90-second baseline** captured at startup.

## How the arousal score works

1. **Calibrate (first 90s):** subject sits still, breathes normally. System records mean and std of each channel.
2. **Live:** every channel is converted to a z-score against the baseline. HRV is inverted (lower RMSSD = higher arousal). Weighted blend:
   - HR 40%, HRV 25%, Breathing 20%, Motion 10%, Blink 5%
3. **Quality gating:** weak underlying rPPG signal auto-down-weights HR and HRV. Same for breathing.
4. **Squash + smooth:** sigmoid to 0–100, 15-frame moving average.

## Architecture

```
webcam frame
   │
   ▼
face_tracker.py   ──── MediaPipe FaceMesh + Pose (single source of truth)
   │
   ├──► rppg_engine.py          (HR)
   │       └──► hrv_engine.py        (HRV from rPPG peaks)
   ├──► breathing_engine.py     (BR)
   ├──► blink_engine.py         (blinks)
   └──► motion_engine.py        (motion)
              │
              ▼
        arousal_fusion.py   (90s baseline + weighted z-score → 0–100)
              │
              ▼
           main.py / streamlit_app.py
```

Every engine exposes the same shape: `engine.update(frame, tracker_data) -> dict`. Trivial to wire into Streamlit.

## Running

```bash
# Activate venv first
python main.py
```

Controls:
- `q` — quit
- `r` — reset baseline (re-run calibration)

## Limitations

- **Not a medical device.** Webcam HRV especially is noisy compared to a chest strap.
- Needs decent steady lighting and a mostly-still subject facing the camera.
- Baseline is per-session; restart = recalibrate.
- Tested on one subject. Skin tone, lighting, and camera variation affect signal quality.

## Files

- `face_tracker.py` — MediaPipe wrapper; returns all ROIs and landmarks per frame.
- `rppg_engine.py` — HR estimation.
- `breathing_engine.py` — Breathing rate.
- `hrv_engine.py` — RMSSD from rPPG peaks.
- `blink_engine.py` — Blink detection (EAR).
- `motion_engine.py` — Head jitter + frame-diff.
- `arousal_fusion.py` — Baseline calibration + weighted arousal score.
- `main.py` — Live OpenCV demo wiring everything together.
- `requirements.txt` — Pinned deps.
