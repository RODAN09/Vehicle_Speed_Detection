import streamlit as st
import cv2
import tempfile
from detector import VehicleDetector

st.set_page_config(page_title="Vehicle Speed Detection", layout="wide")

st.title("🚗 Vehicle Speed Detection Dashboard")

# ---------------- SESSION STATE ----------------
if "run" not in st.session_state:
    st.session_state.run = False

# ---------------- LOAD MODEL ----------------
detector = VehicleDetector("weights/yolov8n.pt")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")

source = st.sidebar.radio("Input Source", ["Upload Video", "Live Camera"])

if st.sidebar.button("▶ Start"):
    st.session_state.run = True

if st.sidebar.button("⏹ Stop"):
    st.session_state.run = False

frame_placeholder = st.empty()

# ---------------- VIDEO UPLOAD ----------------
if source == "Upload Video":
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

    if video_file and st.session_state.run:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        while cap.isOpened() and st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                break

            frame = detector.process_frame(frame)
            frame_placeholder.image(frame, channels="BGR")

        cap.release()

# ---------------- LIVE CAMERA ----------------
if source == "Live Camera" and st.session_state.run:
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.process_frame(frame)
        frame_placeholder.image(frame, channels="BGR")

    cap.release()
