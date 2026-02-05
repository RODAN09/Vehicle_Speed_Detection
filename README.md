🚗 Vehicle Speed Detection System (YOLOv8 + Deep SORT + Streamlit)
📌 Project Overview

This project is an end-to-end vehicle speed detection system built using YOLOv8 for object detection, Deep SORT (real-time version) for multi-object tracking, and Streamlit for an interactive dashboard.
The system supports both uploaded video files and live webcam input, detects vehicles in real time, tracks them across frames, and estimates their approximate speed based on pixel displacement over time.
This project focuses on real-world ML engineering challenges, including model integration, tracker compatibility with modern PyTorch versions, real-time performance handling, and UI–backend coordination.

🎯 Key Features

🚘 Real-time vehicle detection using YOLOv8
🔁 Multi-object tracking with Deep SORT (deep-sort-realtime)
⚡ Live speed estimation for each tracked vehicle
📹 Supports video upload and live webcam stream
▶️ Start / ⏹ Stop controls via Streamlit dashboard
🖥️ Clean, interactive web-based UI
🧠 Modern, PyTorch-compatible tracking pipeline

🛠️ Tech Stack

Programming Language: Python
Object Detection: YOLOv8 (Ultralytics)
Object Tracking: Deep SORT (deep-sort-realtime)
Frontend / Dashboard: Streamlit
Computer Vision: OpenCV
Deep Learning: PyTorch

⚙️ System Workflow

Input is provided via uploaded video or live webcam
YOLOv8 detects vehicles frame-by-frame
Deep SORT assigns persistent IDs and tracks vehicles across frames
Speed is estimated using pixel displacement between frames
Results are visualized in real time on the Streamlit dashboard

📂 Project Structure
vehicle_speed_detection/
│
├── app.py                # Streamlit dashboard (UI)
├── detector.py           # YOLOv8 + Deep SORT backend logic
├── requirements.txt      # Project dependencies
│
└── weights/
    └── yolov8n.pt        # YOLOv8 model weights


📊 Speed Estimation Method

The vehicle speed is estimated using Euclidean distance between consecutive tracked positions:
Pixel displacement is converted to meters using an approximate pixels-per-meter (PPM) factor
Speed is calculated based on movement across frames

🚧 Limitations

Speed estimation is approximate, not legally accurate
Accuracy depends on camera position, frame rate, and object distance
Streamlit is not optimized for high-FPS real-time video processing

🚀 Future Improvements

Line-crossing based speed estimation for higher accuracy
Vehicle counting and analytics dashboard
FPS optimization and GPU acceleration
Export results (CSV / video output)
Deployment on cloud platforms (Streamlit Cloud / AWS)

🧠 Learning Outcomes

End-to-end ML system integration
Real-time object detection and tracking
Handling library incompatibilities and API changes
Building interactive ML dashboards
Debugging production-level issues in computer vision pipelines

## Model Weights
Download YOLOv8 weights manually:

https://github.com/ultralytics/ultralytics

Place `yolov8n.pt` inside the `weights/` directory.
