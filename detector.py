import cv2
import math
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------- SPEED ESTIMATION ----------------
def estimate_speed(p1, p2):
    d_pixel = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    ppm = 8  # pixels per meter (approx calibration)
    d_meters = d_pixel / ppm
    return int(d_meters * 15 * 3.6)

# ---------------- DETECTOR CLASS ----------------
class VehicleDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True
        )

        self.track_history = {}

    def process_frame(self, frame):
        # Resize for stable FPS
        frame = cv2.resize(frame, (960, 540))

        results = self.model(frame, conf=0.4, iou=0.5)[0]

        detections = []

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(float, box)
                detections.append(([x1, y1, x2, y2], float(conf)))

        tracks = self.tracker.update_tracks(
            detections,
            frame=frame
        )

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=5)

            self.track_history[track_id].appendleft((cx, cy))

            speed = 0
            if len(self.track_history[track_id]) >= 2:
                speed = estimate_speed(
                    self.track_history[track_id][-1],
                    self.track_history[track_id][0]
                )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id} | {speed} km/h",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        return frame
