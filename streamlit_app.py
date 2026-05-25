"""
streamlit_app.py - Physiological Arousal Monitor
"""

import av
import cv2
import os
import time
import threading
import numpy as np
import pandas as pd
import streamlit as st
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from face_tracker import FaceTracker
from rppg_engine import RPPGEngine
from breathing_engine import BreathingEngine
from blink_engine import BlinkEngine
from motion_engine import MotionEngine
from hrv_engine import HRVEngine
from arousal_fusion import ArousalFusion

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

# Track calibration start time in session state so reruns don't reset it
if "cal_start_time" not in st.session_state:
    st.session_state.cal_start_time = None

st.set_page_config(page_title="Arousal Monitor", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { background-color: #080c10 !important; color: #c8d8e8 !important; font-family: 'Rajdhani', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 1rem 2rem !important; max-width: 1400px; }
.title-block { text-align: center; padding: 0.5rem 0 1.5rem 0; border-bottom: 1px solid #1a2a3a; margin-bottom: 1.5rem; }
.title-block h1 { font-family: 'Share Tech Mono', monospace !important; font-size: 1.6rem !important; letter-spacing: 0.25em; color: #00e5ff !important; margin: 0 !important; text-transform: uppercase; }
.title-block p { font-size: 0.75rem; letter-spacing: 0.15em; color: #4a6a7a; margin: 0.2rem 0 0 0; text-transform: uppercase; }
.arousal-card { background: linear-gradient(135deg, #0d1f2d 0%, #0a1520 100%); border: 1px solid #1a3a4a; border-radius: 4px; padding: 1.5rem; text-align: center; position: relative; overflow: hidden; }
.arousal-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, #00e5ff, transparent); }
.arousal-label { font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; letter-spacing: 0.3em; color: #4a7a8a; text-transform: uppercase; margin-bottom: 0.3rem; }
.arousal-value { font-family: 'Share Tech Mono', monospace; font-size: 4.5rem; line-height: 1; font-weight: 400; margin: 0; }
.arousal-sub { font-size: 0.7rem; letter-spacing: 0.2em; color: #4a6a7a; margin-top: 0.3rem; text-transform: uppercase; }
.metric-card { background: #0a1520; border: 1px solid #1a2a3a; border-radius: 4px; padding: 1rem 1.2rem; margin-bottom: 0.6rem; position: relative; }
.metric-card::after { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 2px; border-radius: 4px 0 0 4px; }
.metric-card.hr::after { background: #00e5ff; }
.metric-card.br::after { background: #00ff88; }
.metric-card.hrv::after { background: #ff6b6b; }
.metric-card.blink::after { background: #ffd93d; }
.metric-card.motion::after { background: #c77dff; }
.metric-name { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; letter-spacing: 0.25em; color: #4a6a7a; text-transform: uppercase; }
.metric-value { font-family: 'Share Tech Mono', monospace; font-size: 1.8rem; line-height: 1.1; margin: 0.1rem 0; }
.metric-unit { font-size: 0.65rem; letter-spacing: 0.15em; color: #4a6a7a; text-transform: uppercase; }
.section-header { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; letter-spacing: 0.3em; color: #2a4a5a; text-transform: uppercase; border-bottom: 1px solid #1a2a3a; padding-bottom: 0.4rem; margin-bottom: 0.8rem; }
.stButton > button { background: transparent !important; border: 1px solid #1a3a4a !important; color: #4a8a9a !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.65rem !important; letter-spacing: 0.2em !important; text-transform: uppercase !important; padding: 0.4rem 1.2rem !important; border-radius: 2px !important; width: 100% !important; }
.stButton > button:hover { border-color: #00e5ff !important; color: #00e5ff !important; }
</style>
""", unsafe_allow_html=True)

CALIBRATION_SECONDS = 90
HISTORY_LEN = 150
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class ArousalProcessor(VideoProcessorBase):
    def __init__(self):
        os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
        self.tracker = FaceTracker(enable_pose=False)
        self.rppg = RPPGEngine()
        self.breathing = BreathingEngine()
        self.blink = BlinkEngine()
        self.motion = MotionEngine()
        self.hrv = HRVEngine()
        self.fusion = ArousalFusion(calibration_seconds=CALIBRATION_SECONDS)
        self._start_time = time.time()
        self._last_update = 0.0
        self._lock = threading.Lock()
        self._hr_hist = deque(maxlen=HISTORY_LEN)
        self._br_hist = deque(maxlen=HISTORY_LEN)
        self._arousal_hist = deque(maxlen=HISTORY_LEN)
        self.result = self._empty_result()

    def _empty_result(self):
        return {"hr": None, "br": None, "hrv": None, "blink": None, "motion": None,
                "arousal": None, "phase": "calibrating", "cal_progress": 0.0,
                "face_found": False, "hr_q": 0.0, "br_q": 0.0,
                "hr_hist": [], "br_hist": [], "arousal_hist": []}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        tdata = self.tracker.update(img)
        rppg_out = self.rppg.update(img, tdata)
        br_out = self.breathing.update(img, tdata)
        blink_out = self.blink.update(img, tdata)
        motion_out = self.motion.update(img, tdata)
        hrv_out = self.hrv.update(rppg_out.get("peak_times", []))
        fusion_state = self.fusion.update({
            "hr_bpm": rppg_out.get("bpm"), "hr_quality": rppg_out.get("signal_quality"),
            "brpm": br_out.get("brpm"), "br_quality": br_out.get("signal_quality"),
            "rmssd_ms": hrv_out.get("rmssd_ms"), "motion_score": motion_out.get("motion_score"),
            "blink_rate_bpm": blink_out.get("blink_rate_bpm"),
        })

        now = time.time()
        if now - self._last_update > 0.33:
            self._last_update = now
            hr = rppg_out.get("bpm")
            br = br_out.get("brpm")
            arousal = fusion_state.get("arousal_score")
            if hr is not None: self._hr_hist.append(hr)
            if br is not None: self._br_hist.append(br)
            if arousal is not None: self._arousal_hist.append(arousal)
            with self._lock:
                self.result = {
                    "hr": hr, "br": br, "hrv": hrv_out.get("rmssd_ms"),
                    "blink": blink_out.get("blink_rate_bpm"), "motion": motion_out.get("motion_score"),
                    "arousal": arousal, "phase": fusion_state["phase"],
                    "cal_progress": fusion_state.get("calibration_progress", 0.0),
                    "face_found": tdata["face_found"],
                    "hr_q": rppg_out.get("signal_quality") or 0.0,
                    "br_q": br_out.get("signal_quality") or 0.0,
                    "hr_hist": list(self._hr_hist), "br_hist": list(self._br_hist),
                    "arousal_hist": list(self._arousal_hist),
                }

        if tdata.get("forehead_roi"):
            x, y, w, h = tdata["forehead_roi"]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 229, 255), 1)
        if tdata.get("chest_roi"):
            x, y, w, h = tdata["chest_roi"]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 136), 1)
        if tdata.get("left_eye_pts") is not None:
            cv2.polylines(img, [tdata["left_eye_pts"]], True, (0, 200, 255), 1)
        if tdata.get("right_eye_pts") is not None:
            cv2.polylines(img, [tdata["right_eye_pts"]], True, (0, 200, 255), 1)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def fmt(v, decimals=1):
    return "---" if v is None else f"{v:.{decimals}f}"

def arousal_color(score):
    if score is None: return "#4a6a7a"
    if score < 35: return "#00ff88"
    if score < 55: return "#00e5ff"
    if score < 70: return "#ffd93d"
    return "#ff6b6b"

def metric_card(css_class, label, value, unit, color):
    return f'<div class="metric-card {css_class}"><div class="metric-name">{label}</div><div class="metric-value" style="color:{color}">{value}</div><div class="metric-unit">{unit}</div></div>'


# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="title-block"><h1>⬡ Physiological Arousal Monitor</h1><p>Contactless biometric sensing via webcam</p></div>', unsafe_allow_html=True)

# ── Instructions expander ──────────────────────────────────────────────────────
with st.expander("📖  How to use this — read before starting", expanded=False):
    st.markdown("""
<div style="font-family:'Rajdhani',sans-serif;color:#c8d8e8;line-height:1.7;font-size:0.95rem;">

<div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;letter-spacing:0.2em;color:#00e5ff;text-transform:uppercase;margin-bottom:0.8rem;">What is this?</div>

This system detects your physiological arousal level in real time using only your webcam — no wearables, no sensors. It tracks 5 signals simultaneously and combines them into a single <b style="color:#00e5ff;">Arousal Score (0–100)</b> that reflects how activated or stressed your nervous system is relative to your own calm baseline.

<br><br>

<div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;letter-spacing:0.2em;color:#00e5ff;text-transform:uppercase;margin-bottom:0.8rem;">The 5 signals</div>

<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
<tr style="border-bottom:1px solid #1a2a3a;">
<td style="padding:0.4rem 0.8rem 0.4rem 0;color:#00e5ff;font-family:'Share Tech Mono',monospace;font-size:0.7rem;">HEART RATE</td>
<td style="padding:0.4rem 0;color:#8a9ab0;">Green-channel color changes in your forehead skin, caused by blood pulsing through capillaries</td>
</tr>
<tr style="border-bottom:1px solid #1a2a3a;">
<td style="padding:0.4rem 0.8rem 0.4rem 0;color:#00ff88;font-family:'Share Tech Mono',monospace;font-size:0.7rem;">BREATHING</td>
<td style="padding:0.4rem 0;color:#8a9ab0;">Brightness changes in the chest/shoulder region as it rises and falls</td>
</tr>
<tr style="border-bottom:1px solid #1a2a3a;">
<td style="padding:0.4rem 0.8rem 0.4rem 0;color:#ff6b6b;font-family:'Share Tech Mono',monospace;font-size:0.7rem;">HRV</td>
<td style="padding:0.4rem 0;color:#8a9ab0;">Variation between individual heartbeat intervals — lower variability = higher stress</td>
</tr>
<tr style="border-bottom:1px solid #1a2a3a;">
<td style="padding:0.4rem 0.8rem 0.4rem 0;color:#ffd93d;font-family:'Share Tech Mono',monospace;font-size:0.7rem;">BLINK RATE</td>
<td style="padding:0.4rem 0;color:#8a9ab0;">Blinks per minute via Eye Aspect Ratio — stress and cognitive load affect blink frequency</td>
</tr>
<tr>
<td style="padding:0.4rem 0.8rem 0.4rem 0;color:#c77dff;font-family:'Share Tech Mono',monospace;font-size:0.7rem;">MOTION</td>
<td style="padding:0.4rem 0;color:#8a9ab0;">Head jitter and facial movement — fidgeting and restlessness are real arousal indicators</td>
</tr>
</table>

<br>

<div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;letter-spacing:0.2em;color:#00e5ff;text-transform:uppercase;margin-bottom:0.8rem;">How the score works</div>

The first <b style="color:#c8d8e8;">90 seconds</b> are a calibration phase. Sit still and breathe normally. The system records your personal baseline for each signal. After calibration, every reading is converted to a z-score — how far above or below YOUR normal is this right now? These are weighted and combined into the 0–100 arousal score. The score is relative to you, not absolute.

<br><br>

<div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;letter-spacing:0.2em;color:#00e5ff;text-transform:uppercase;margin-bottom:0.8rem;">Camera setup — important</div>

For best results:

<ul style="color:#8a9ab0;margin-top:0.4rem;padding-left:1.2rem;">
<li><b style="color:#c8d8e8;">Sit 40–70cm</b> from the camera — close enough for the face to be clearly visible</li>
<li><b style="color:#c8d8e8;">Face the light source</b> — a window or lamp in front of you, not behind. Backlit faces break the heart rate signal</li>
<li><b style="color:#c8d8e8;">Show your upper chest</b> — tilt the camera slightly down or sit back so shoulders are visible. This is needed for breathing detection</li>
<li><b style="color:#c8d8e8;">Stay relatively still</b> during calibration — movement during the first 90s skews your baseline</li>
<li><b style="color:#c8d8e8;">Use Chrome</b> for best WebRTC camera performance</li>
</ul>

<br>

<div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;letter-spacing:0.2em;color:#00e5ff;text-transform:uppercase;margin-bottom:0.8rem;">Limitations</div>

<span style="color:#4a6a7a;">This is a research-grade demo, not a medical device. Webcam signals are noisier than contact sensors. Results are most meaningful within a single session, compared to your own baseline. Lighting and camera quality affect accuracy significantly.</span>

</div>
""", unsafe_allow_html=True)

# ── Camera selector + webrtc (MUST be defined before columns use snap) ─────────
camera_options = {"Default camera": 0, "Camera 1": 1, "Camera 2": 2}
selected = st.selectbox("Select camera (stop stream first to switch)", list(camera_options.keys()), index=0)
device_id = camera_options[selected]

ctx = webrtc_streamer(
    key=f"arousal-{device_id}",
    video_processor_factory=ArousalProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": {"deviceId": {"ideal": str(device_id)}}, "audio": False},
    async_processing=False,
)

# ── Get snap (always defined, ctx always exists here) ──────────────────────────
_empty = {"hr": None, "br": None, "hrv": None, "blink": None, "motion": None,
          "arousal": None, "phase": "calibrating", "cal_progress": 0.0,
          "face_found": False, "hr_q": 0.0, "br_q": 0.0,
          "hr_hist": [], "br_hist": [], "arousal_hist": []}
try:
    if ctx.video_processor is not None:
        with ctx.video_processor._lock:
            snap = dict(ctx.video_processor.result)
    else:
        snap = _empty
except Exception:
    snap = _empty

# ── Layout columns ─────────────────────────────────────────────────────────────
col_feed, col_mid, col_right = st.columns([2.2, 1.4, 1.4])

with col_feed:
    st.markdown('<div class="section-header">Live Feed</div>', unsafe_allow_html=True)
    face_dot = "green" if snap["face_found"] else "red"
    face_txt = "Face detected" if snap["face_found"] else "No face detected"
    hr_q_pct = int((snap["hr_q"] or 0) * 100)
    br_q_pct = int((snap["br_q"] or 0) * 100)
    st.markdown(f"""
    <div style="display:flex;gap:1.5rem;margin-top:0.6rem;flex-wrap:wrap;">
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#4a6a7a;display:flex;align-items:center;gap:0.4rem;">
            <span style="width:6px;height:6px;border-radius:50%;background:{'#00ff88' if face_dot=='green' else '#ff4444'};display:inline-block;"></span>{face_txt}
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#4a6a7a;">rPPG {hr_q_pct}%</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#4a6a7a;">BR {br_q_pct}%</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("↺  Reset Baseline"):
        if ctx.video_processor is not None:
            ctx.video_processor.fusion.reset()
        st.session_state.cal_start_time = time.time()

with col_mid:
    st.markdown('<div class="section-header">Arousal Score</div>', unsafe_allow_html=True)
    score = snap["arousal"]
    color = arousal_color(score)
    score_txt = fmt(score, 1)

    if snap["phase"] == "calibrating":
        # Only start timer when stream is actually running
        stream_active = ctx.video_processor is not None
        if stream_active:
            if st.session_state.cal_start_time is None:
                st.session_state.cal_start_time = time.time()
            elapsed = time.time() - st.session_state.cal_start_time
        else:
            st.session_state.cal_start_time = None
            elapsed = 0
        prog = min(1.0, elapsed / CALIBRATION_SECONDS)
        remaining = max(0, int(CALIBRATION_SECONDS - elapsed))
        st.markdown(f'<div style="background:#0a1520;border:1px solid #1a2a3a;border-radius:4px;padding:1rem 1.2rem;margin-bottom:1rem;"><div style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;letter-spacing:0.25em;color:#00e5ff;text-transform:uppercase;margin-bottom:0.5rem;">▶ Calibrating — {remaining}s remaining</div><div style="font-family:\'Share Tech Mono\',monospace;font-size:0.6rem;color:#2a5a6a;">Sit still · breathe normally</div></div>', unsafe_allow_html=True)
        st.progress(float(prog))
    else:
        label = "CALM" if (score or 50) < 35 else ("ELEVATED" if (score or 50) < 65 else "HIGH")
        st.markdown(f'<div class="arousal-card"><div class="arousal-label">Arousal Index</div><div class="arousal-value" style="color:{color}">{score_txt}</div><div class="arousal-sub">/ 100 &nbsp;·&nbsp; {label}</div></div>', unsafe_allow_html=True)
        if len(snap["arousal_hist"]) > 2:
            st.line_chart(pd.DataFrame({"Arousal": snap["arousal_hist"]}), height=100, use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:1rem;">Signals</div>', unsafe_allow_html=True)
    st.markdown(metric_card("hr", "Heart Rate", fmt(snap["hr"]), "BPM", "#00e5ff"), unsafe_allow_html=True)
    st.markdown(metric_card("br", "Breathing", fmt(snap["br"]), "Breaths / min", "#00ff88"), unsafe_allow_html=True)
    hrv_display = min(snap["hrv"], 150) if snap["hrv"] is not None else None
    st.markdown(metric_card("hrv", "HRV · RMSSD", fmt(hrv_display), "ms", "#ff6b6b"), unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-header">Behavioral</div>', unsafe_allow_html=True)
    st.markdown(metric_card("blink", "Blink Rate", fmt(snap["blink"]), "Blinks / min", "#ffd93d"), unsafe_allow_html=True)
    st.markdown(metric_card("motion", "Motion", fmt(snap["motion"], 2), "0 – 1 scale", "#c77dff"), unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:1rem;">HR Trend</div>', unsafe_allow_html=True)
    if len(snap["hr_hist"]) > 2:
        st.line_chart(pd.DataFrame({"HR (bpm)": snap["hr_hist"]}), height=100, use_container_width=True)
    else:
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.6rem;color:#2a4a5a;padding:1rem 0;text-align:center;">Warming up...</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:0.8rem;">Breathing Trend</div>', unsafe_allow_html=True)
    if len(snap["br_hist"]) > 2:
        st.line_chart(pd.DataFrame({"Breathing (/min)": snap["br_hist"]}), height=100, use_container_width=True)
    else:
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.6rem;color:#2a4a5a;padding:1rem 0;text-align:center;">Warming up...</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1.2rem;padding:0.8rem;border:1px solid #1a2a3a;border-radius:4px;"><div style="font-family:\'Share Tech Mono\',monospace;font-size:0.55rem;letter-spacing:0.2em;color:#2a4a5a;text-transform:uppercase;margin-bottom:0.5rem;">How it works</div><div style="font-size:0.75rem;color:#4a6a7a;line-height:1.5;">5 signals z-scored against your 90s calm baseline, weighted into a single arousal index.<br><br>No wearables. No contact. Just your webcam.</div></div>', unsafe_allow_html=True)

    # ── PDF Report ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header" style="margin-top:1.2rem;">Session Report</div>', unsafe_allow_html=True)

    def generate_report():
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        hr_avg = f"{sum(snap['hr_hist'])/len(snap['hr_hist']):.1f} BPM" if snap["hr_hist"] else "No data"
        br_avg = f"{sum(snap['br_hist'])/len(snap['br_hist']):.1f} /min" if snap["br_hist"] else "No data"
        arousal_avg = f"{sum(snap['arousal_hist'])/len(snap['arousal_hist']):.1f} / 100" if snap["arousal_hist"] else "No data"
        arousal_peak = f"{max(snap['arousal_hist']):.1f} / 100" if snap["arousal_hist"] else "No data"
        hrv_val = f"{min(snap['hrv'], 150):.1f} ms" if snap["hrv"] else "No data"
        blink_val = f"{snap['blink']:.1f} /min" if snap["blink"] else "No data"
        motion_val = f"{snap['motion']:.2f}" if snap["motion"] else "No data"
        phase = "Live" if snap["phase"] == "live" else "Calibrating"

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #080c10; color: #c8d8e8; padding: 40px; margin: 0; }}
  h1 {{ font-family: monospace; color: #00e5ff; letter-spacing: 0.2em; text-transform: uppercase; font-size: 1.4rem; border-bottom: 1px solid #1a2a3a; padding-bottom: 12px; }}
  h2 {{ font-family: monospace; color: #4a7a8a; font-size: 0.65rem; letter-spacing: 0.25em; text-transform: uppercase; margin-top: 28px; margin-bottom: 10px; }}
  .meta {{ font-size: 0.75rem; color: #4a6a7a; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #1a2a3a; font-size: 0.9rem; }}
  td:first-child {{ color: #4a6a7a; font-family: monospace; font-size: 0.75rem; letter-spacing: 0.1em; width: 40%; }}
  td:last-child {{ color: #c8d8e8; font-weight: 600; }}
  .arousal-big {{ font-family: monospace; font-size: 3rem; color: #00e5ff; text-align: center; padding: 20px; border: 1px solid #1a3a4a; border-radius: 4px; margin: 12px 0; background: #0d1f2d; }}
  .disclaimer {{ margin-top: 32px; padding: 12px; border: 1px solid #1a2a3a; border-radius: 4px; font-size: 0.75rem; color: #4a6a7a; line-height: 1.6; }}
</style>
</head>
<body>
<h1>⬡ Physiological Arousal Monitor</h1>
<div class="meta">Session Report &nbsp;·&nbsp; Generated {now} &nbsp;·&nbsp; Status: {phase}</div>

<h2>Arousal Score</h2>
<div class="arousal-big">{arousal_avg}</div>

<h2>Session Averages</h2>
<table>
<tr><td>Heart Rate</td><td>{hr_avg}</td></tr>
<tr><td>Breathing Rate</td><td>{br_avg}</td></tr>
<tr><td>HRV · RMSSD</td><td>{hrv_val}</td></tr>
<tr><td>Blink Rate</td><td>{blink_val}</td></tr>
<tr><td>Motion Score</td><td>{motion_val}</td></tr>
</table>

<h2>Arousal Summary</h2>
<table>
<tr><td>Average Arousal</td><td>{arousal_avg}</td></tr>
<tr><td>Peak Arousal</td><td>{arousal_peak}</td></tr>
</table>

<div class="disclaimer">
This report is generated from a single webcam session. Values represent averages over the live session window.
Arousal score is relative to your personal 90-second calm baseline — not an absolute measure.
This is a research-grade tool, not a medical device.
<br><br>phys-arousal-monitor.streamlit.app
</div>
</body>
</html>"""
        return html

    if snap["phase"] == "live" and snap["arousal_hist"]:
        report_html = generate_report()
        st.download_button(
            label="⬇  Download Session Report",
            data=report_html.encode("utf-8"),
            file_name="arousal_report.html",
            mime="text/html",
            use_container_width=True,
        )
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.55rem;color:#2a4a5a;text-align:center;margin-top:0.3rem;">Opens in browser · Print to PDF with Ctrl+P</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.6rem;color:#2a4a5a;text-align:center;padding:0.5rem 0;">Available after calibration completes</div>', unsafe_allow_html=True)
