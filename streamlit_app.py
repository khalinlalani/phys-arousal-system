"""
streamlit_app.py
----------------
Physiological Arousal Monitor — Streamlit web app.
Uses streamlit-webrtc for live browser webcam access.

Run locally:  streamlit run streamlit_app.py
Deploy:       push to GitHub → connect streamlit.io/cloud
"""

import av
import cv2
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

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Arousal Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    background-color: #080c10 !important;
    color: #c8d8e8 !important;
    font-family: 'Rajdhani', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 1rem 2rem !important; max-width: 1400px; }

.title-block {
    text-align: center;
    padding: 0.5rem 0 1.5rem 0;
    border-bottom: 1px solid #1a2a3a;
    margin-bottom: 1.5rem;
}
.title-block h1 {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.6rem !important;
    letter-spacing: 0.25em;
    color: #00e5ff !important;
    margin: 0 !important;
    text-transform: uppercase;
}
.title-block p {
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    color: #4a6a7a;
    margin: 0.2rem 0 0 0;
    text-transform: uppercase;
}
.arousal-card {
    background: linear-gradient(135deg, #0d1f2d 0%, #0a1520 100%);
    border: 1px solid #1a3a4a;
    border-radius: 4px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.arousal-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00e5ff, transparent);
}
.arousal-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    color: #4a7a8a;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.arousal-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 4.5rem;
    line-height: 1;
    font-weight: 400;
    margin: 0;
}
.arousal-sub {
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #4a6a7a;
    margin-top: 0.3rem;
    text-transform: uppercase;
}
.metric-card {
    background: #0a1520;
    border: 1px solid #1a2a3a;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    position: relative;
}
.metric-card::after {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 2px;
    border-radius: 4px 0 0 4px;
}
.metric-card.hr::after    { background: #00e5ff; }
.metric-card.br::after    { background: #00ff88; }
.metric-card.hrv::after   { background: #ff6b6b; }
.metric-card.blink::after { background: #ffd93d; }
.metric-card.motion::after{ background: #c77dff; }
.metric-name {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    color: #4a6a7a;
    text-transform: uppercase;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.8rem;
    line-height: 1.1;
    margin: 0.1rem 0;
}
.metric-unit {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: #4a6a7a;
    text-transform: uppercase;
}
.cal-container {
    background: #0a1520;
    border: 1px solid #1a2a3a;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.cal-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    color: #00e5ff;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.status-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: #4a6a7a;
    text-transform: uppercase;
    margin-top: 0.6rem;
}
.dot { width:6px; height:6px; border-radius:50%; display:inline-block; }
.dot.green  { background:#00ff88; box-shadow:0 0 6px #00ff88; }
.dot.red    { background:#ff4444; box-shadow:0 0 6px #ff4444; }
.dot.yellow { background:#ffd93d; box-shadow:0 0 6px #ffd93d; }
.section-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    color: #2a4a5a;
    text-transform: uppercase;
    border-bottom: 1px solid #1a2a3a;
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
}
.stButton > button {
    background: transparent !important;
    border: 1px solid #1a3a4a !important;
    color: #4a8a9a !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 0.4rem 1.2rem !important;
    border-radius: 2px !important;
    width: 100% !important;
}
.stButton > button:hover {
    border-color: #00e5ff !important;
    color: #00e5ff !important;
    background: rgba(0,229,255,0.05) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CALIBRATION_SECONDS = 90
HISTORY_LEN = 150

RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ── Result container (stored on processor instance, read via ctx.video_processor) ──

HISTORY_LEN = 150

# ── Video processor ────────────────────────────────────────────────────────────
class ArousalProcessor(VideoProcessorBase):
    def __init__(self):
        import os
        os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
        self.tracker = FaceTracker(enable_pose=False)
        self.rppg = RPPGEngine()
        self.breathing = BreathingEngine()
        self.blink = BlinkEngine()
        self.motion = MotionEngine()
        self.hrv = HRVEngine()
        self.fusion = ArousalFusion(calibration_seconds=CALIBRATION_SECONDS)
        self._last_update = 0.0
        self._lock = threading.Lock()
        # Results stored here, read by main thread via ctx.video_processor
        self.result = {
            "hr": None, "br": None, "hrv": None,
            "blink": None, "motion": None,
            "arousal": None, "phase": "calibrating",
            "cal_progress": 0.0, "face_found": False,
            "hr_q": 0.0, "br_q": 0.0,
            "hr_hist": [], "br_hist": [], "arousal_hist": [],
        }
        self._hr_hist = deque(maxlen=HISTORY_LEN)
        self._br_hist = deque(maxlen=HISTORY_LEN)
        self._arousal_hist = deque(maxlen=HISTORY_LEN)

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
            "hr_bpm": rppg_out.get("bpm"),
            "hr_quality": rppg_out.get("signal_quality"),
            "brpm": br_out.get("brpm"),
            "br_quality": br_out.get("signal_quality"),
            "rmssd_ms": hrv_out.get("rmssd_ms"),
            "motion_score": motion_out.get("motion_score"),
            "blink_rate_bpm": blink_out.get("blink_rate_bpm"),
        })

        now = time.time()
        if now - self._last_update > 0.33:
            self._last_update = now
            hr = rppg_out.get("bpm")
            br = br_out.get("brpm")
            arousal = fusion_state.get("arousal_score")
            if hr is not None:
                self._hr_hist.append(hr)
            if br is not None:
                self._br_hist.append(br)
            if arousal is not None:
                self._arousal_hist.append(arousal)
            with self._lock:
                self.result = {
                    "hr": hr,
                    "br": br,
                    "hrv": hrv_out.get("rmssd_ms"),
                    "blink": blink_out.get("blink_rate_bpm"),
                    "motion": motion_out.get("motion_score"),
                    "arousal": arousal,
                    "phase": fusion_state["phase"],
                    "cal_progress": fusion_state.get("calibration_progress", 0.0),
                    "face_found": tdata["face_found"],
                    "hr_q": rppg_out.get("signal_quality") or 0.0,
                    "br_q": br_out.get("signal_quality") or 0.0,
                    "hr_hist": list(self._hr_hist),
                    "br_hist": list(self._br_hist),
                    "arousal_hist": list(self._arousal_hist),
                }

        # Draw ROI overlays
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

# ── Helpers ────────────────────────────────────────────────────────────────────
def fmt(v, decimals=1):
    if v is None:
        return "---"
    return f"{v:.{decimals}f}"

def arousal_color(score):
    if score is None: return "#4a6a7a"
    if score < 35:    return "#00ff88"
    if score < 55:    return "#00e5ff"
    if score < 70:    return "#ffd93d"
    return "#ff6b6b"

def metric_card(css_class, label, value, unit, color):
    return f"""
    <div class="metric-card {css_class}">
        <div class="metric-name">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-unit">{unit}</div>
    </div>"""

# ── Layout ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>⬡ Physiological Arousal Monitor</h1>
    <p>Contactless biometric sensing via webcam</p>
</div>
""", unsafe_allow_html=True)

col_feed, col_mid, col_right = st.columns([2.2, 1.4, 1.4])

# Get latest results from video processor
_default_snap = {
    "hr": None, "br": None, "hrv": None, "blink": None, "motion": None,
    "arousal": None, "phase": "calibrating", "cal_progress": 0.0,
    "face_found": False, "hr_q": 0.0, "br_q": 0.0,
    "hr_hist": [], "br_hist": [], "arousal_hist": [],
}
if ctx.video_processor is not None:
    with ctx.video_processor._lock:
        snap = dict(ctx.video_processor.result)
else:
    snap = _default_snap

# ── Left: webcam ───────────────────────────────────────────────────────────────
with col_feed:
    st.markdown('<div class="section-header">Live Feed</div>', unsafe_allow_html=True)

    camera_options = {"Default camera": 0, "Camera 1": 1, "Camera 2": 2}
    selected = st.selectbox("Select camera (stop stream first to switch)", list(camera_options.keys()), index=0)
    device_id = camera_options[selected]

    ctx = webrtc_streamer(
        key=f"arousal-{device_id}",
        video_processor_factory=ArousalProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": {"deviceId": {"ideal": str(device_id)}}, "audio": False},
        async_processing=True,
    )

    face_dot = "green" if snap["face_found"] else "red"
    face_txt = "Face detected" if snap["face_found"] else "No face detected"
    hr_q_pct = int((snap["hr_q"] or 0) * 100)
    br_q_pct = int((snap["br_q"] or 0) * 100)

    st.markdown(f"""
    <div class="status-row"><span class="dot {face_dot}"></span>{face_txt}</div>
    <div class="status-row"><span class="dot {'green' if hr_q_pct > 40 else 'yellow'}"></span>rPPG signal {hr_q_pct}%</div>
    <div class="status-row"><span class="dot {'green' if br_q_pct > 40 else 'yellow'}"></span>Breathing signal {br_q_pct}%</div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.8rem;'>", unsafe_allow_html=True)
    if st.button("↺  Reset Baseline"):
        if ctx.video_processor is not None:
            ctx.video_processor.fusion.reset()
    st.markdown("</div>", unsafe_allow_html=True)

# ── Middle: arousal score + top 3 metrics ─────────────────────────────────────
with col_mid:
    st.markdown('<div class="section-header">Arousal Score</div>', unsafe_allow_html=True)

    score = snap["arousal"]
    color = arousal_color(score)
    score_txt = fmt(score, 1)

    if snap["phase"] == "calibrating":
        prog = snap["cal_progress"]
        remaining = int(CALIBRATION_SECONDS * (1 - prog))
        st.markdown(f"""
        <div class="cal-container">
            <div class="cal-label">▶ Calibrating — {remaining}s remaining</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;
                        color:#2a5a6a;margin-bottom:0.4rem;">
                Sit still · breathe normally
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(prog))
    else:
        label = "CALM" if (score or 50) < 35 else ("ELEVATED" if (score or 50) < 65 else "HIGH")
        st.markdown(f"""
        <div class="arousal-card">
            <div class="arousal-label">Arousal Index</div>
            <div class="arousal-value" style="color:{color}">{score_txt}</div>
            <div class="arousal-sub">/ 100 &nbsp;·&nbsp; {label}</div>
        </div>
        """, unsafe_allow_html=True)

        if len(snap["arousal_hist"]) > 2:
            st.markdown("<div style='margin-top:0.6rem;'>", unsafe_allow_html=True)
            df = pd.DataFrame({"Arousal": snap["arousal_hist"]})
            st.line_chart(df, height=100, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:1rem;">Signals</div>',
                unsafe_allow_html=True)
    st.markdown(metric_card("hr",  "Heart Rate",   fmt(snap["hr"]),  "BPM",           "#00e5ff"), unsafe_allow_html=True)
    st.markdown(metric_card("br",  "Breathing",    fmt(snap["br"]),  "Breaths / min", "#00ff88"), unsafe_allow_html=True)
    st.markdown(metric_card("hrv", "HRV · RMSSD",  fmt(snap["hrv"]), "ms",            "#ff6b6b"), unsafe_allow_html=True)

# ── Right: behavioral + charts ────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-header">Behavioral</div>', unsafe_allow_html=True)
    st.markdown(metric_card("blink",  "Blink Rate", fmt(snap["blink"]),       "Blinks / min", "#ffd93d"), unsafe_allow_html=True)
    st.markdown(metric_card("motion", "Motion",     fmt(snap["motion"], 2),   "0 – 1 scale",  "#c77dff"), unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:1rem;">HR Trend</div>',
                unsafe_allow_html=True)
    if len(snap["hr_hist"]) > 2:
        st.line_chart(pd.DataFrame({"HR (bpm)": snap["hr_hist"]}), height=100, use_container_width=True)
    else:
        st.markdown("<div style='font-family:Share Tech Mono,monospace;font-size:0.6rem;color:#2a4a5a;padding:1rem 0;text-align:center;'>Warming up...</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:0.8rem;">Breathing Trend</div>',
                unsafe_allow_html=True)
    if len(snap["br_hist"]) > 2:
        st.line_chart(pd.DataFrame({"Breathing (/min)": snap["br_hist"]}), height=100, use_container_width=True)
    else:
        st.markdown("<div style='font-family:Share Tech Mono,monospace;font-size:0.6rem;color:#2a4a5a;padding:1rem 0;text-align:center;'>Warming up...</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.2rem;padding:0.8rem;border:1px solid #1a2a3a;border-radius:4px;">
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;
                    letter-spacing:0.2em;color:#2a4a5a;text-transform:uppercase;
                    margin-bottom:0.5rem;">How it works</div>
        <div style="font-size:0.75rem;color:#4a6a7a;line-height:1.5;">
            5 signals z-scored against your 90s calm baseline,
            weighted into a single arousal index.<br><br>
            No wearables. No contact. Just your webcam.
        </div>
    </div>
    """, unsafe_allow_html=True)
