# Developer: tubakhxn

import os
import sys
import time
import queue
import threading
from collections import deque, defaultdict

import cv2
import numpy as np
import requests
from tqdm import tqdm

from zones import ZoneManager

MODEL_DIR = "models"
MODEL_NAME = "yolov8n.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"

DEMO_DIR = "demo_videos"
DEMO_VIDEO_NAME = "demo_traffic_intersection.mp4"
DEMO_VIDEO_PATH = os.path.join(DEMO_DIR, DEMO_VIDEO_NAME)
DEMO_VIDEO_URL = "https://storage.googleapis.com/coverr-main/mp4/Mt_Baker.mp4"

MAX_FRAME_WIDTH = 1280
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
PEDESTRIAN_CLASSES = {"person"}
TRACK_HISTORY_LEN = 15
SPEED_STOPPED_THRESHOLD = 1.6
FPS_SMOOTHING_WINDOW = 30

BOX_COLOR_VEHICLE = (255, 200, 40)
BOX_COLOR_PEDESTRIAN = (60, 220, 255)
BOX_COLOR_WARNING = (0, 0, 255)
PANEL_BG = (25, 25, 25)
TEXT_WHITE = (245, 245, 245)


def download_with_progress(url, destination, label):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [!] Could not reach '{url}' ({exc}).")
        return False

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 8192

    print(f"  Downloading {label} ...")
    try:
        with open(destination, "wb") as f, tqdm(
            total=total_size if total_size > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"  {label}",
            colour="cyan",
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    except (IOError, OSError) as exc:
        print(f"  [!] Failed writing '{destination}' ({exc}).")
        return False

    print(f"  Saved to {destination}\n")
    return True


def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        print(f"[OK] Model already present: {MODEL_PATH}")
        return MODEL_PATH

    print(f"[i] Model not found locally. Fetching '{MODEL_NAME}' ...")
    ok = download_with_progress(MODEL_URL, MODEL_PATH, MODEL_NAME)
    if not ok or not os.path.isfile(MODEL_PATH):
        print("[!] Automatic model download failed.")
        print(f"    Please manually place a YOLOv8 '.pt' file at: {MODEL_PATH}")
        sys.exit(1)
    return MODEL_PATH


def ensure_demo_video():
    os.makedirs(DEMO_DIR, exist_ok=True)
    if os.path.isfile(DEMO_VIDEO_PATH) and os.path.getsize(DEMO_VIDEO_PATH) > 0:
        print(f"[OK] Demo video already present: {DEMO_VIDEO_PATH}")
        return DEMO_VIDEO_PATH

    print("[i] No demo video found locally. Downloading a fallback clip ...")
    ok = download_with_progress(DEMO_VIDEO_URL, DEMO_VIDEO_PATH, DEMO_VIDEO_NAME)
    if not ok or not os.path.isfile(DEMO_VIDEO_PATH):
        print("[!] Automatic demo video download failed.")
        print("    Please download a traffic clip yourself from one of these:")
        print("      https://www.pexels.com/search/videos/traffic%20intersection/")
        print("      https://pixabay.com/videos/search/traffic/")
        print("      https://www.pexels.com/search/videos/crosswalk/")
        print("      https://pixabay.com/videos/search/road/")
        print(f"    ... then save it as: {DEMO_VIDEO_PATH}")
        sys.exit(1)
    return DEMO_VIDEO_PATH


class VideoStream:
    def __init__(self, path, queue_size=64):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        while not self.stopped.is_set():
            if not self.q.full():
                ok, frame = self.cap.read()
                if not ok:
                    self.stopped.set()
                    break
                self.q.put(frame)
            else:
                time.sleep(0.001)
        self.cap.release()

    def read(self):
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return None

    def more(self):
        return not self.stopped.is_set() or not self.q.empty()

    def stop(self):
        self.stopped.set()


def draw_rounded_box(frame, x1, y1, x2, y2, color, radius=8, thickness=-1, alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_label(frame, text, x, y, color, font_scale=0.55, thickness=1, bg=(15, 15, 15)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y + 2), bg, -1)
    cv2.putText(frame, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color, thickness, cv2.LINE_AA)


def draw_stats_panel(frame, stats, fps):
    x1, y1, x2, y2 = 12, 12, 300, 190
    draw_rounded_box(frame, x1, y1, x2, y2, PANEL_BG, radius=14, alpha=0.6)

    cv2.putText(frame, "SMART CROSSWALK SAFETY", (x1 + 14, y1 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

    rows = [
        (f"FPS: {fps:5.1f}", TEXT_WHITE),
        (f"Vehicles: {stats['vehicles']}", BOX_COLOR_VEHICLE),
        (f"Pedestrians: {stats['pedestrians']}", BOX_COLOR_PEDESTRIAN),
        (f"Waiting: {stats['waiting']}", (235, 160, 50)),
        (f"Crossing: {stats['crossing']}", (50, 50, 235)),
        (f"Yielding vehicles: {stats['yielding']}", (60, 200, 60)),
    ]
    y = y1 + 52
    for text, color in rows:
        cv2.putText(frame, text, (x1 + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
        y += 22


def draw_collision_banner(frame, w):
    flash_on = int(time.time() * 4) % 2 == 0
    if not flash_on:
        return
    text = "COLLISION WARNING - VEHICLE NOT YIELDING"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x = max(10, (w - tw) // 2)
    draw_rounded_box(frame, x - 16, 8, x + tw + 16, 8 + th + 22, (0, 0, 200), radius=10, alpha=0.85)
    cv2.putText(frame, text, (x, 8 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)


def pick_video_source():
    print("\nEnter a path to a traffic video (or press Enter to use the demo clip):")
    user_path = input("  Video path > ").strip().strip('"')

    if user_path:
        if os.path.isfile(user_path):
            return user_path
        print(f"  [!] File not found: '{user_path}'. Falling back to demo video.\n")

    return ensure_demo_video()


def resize_if_needed(frame):
    h, w = frame.shape[:2]
    if w > MAX_FRAME_WIDTH:
        scale = MAX_FRAME_WIDTH / w
        frame = cv2.resize(frame, (MAX_FRAME_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def run():
    print("=" * 60)
    print(" SMART CROSSWALK SAFETY SYSTEM")
    print(" Developer: tubakhxn")
    print("=" * 60)

    model_path = ensure_model()

    from ultralytics import YOLO
    import supervision as sv

    print("[i] Loading YOLO model ...")
    model = YOLO(model_path)
    class_names = model.names

    video_path = pick_video_source()
    print(f"[i] Opening video stream: {video_path}")
    stream = VideoStream(video_path).start()

    tracker = sv.ByteTrack()
    zones = None

    track_history = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LEN))
    fps_deque = deque(maxlen=FPS_SMOOTHING_WINDOW)

    window_name = "Smart Crosswalk Safety System - tubakhxn"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[i] Running. Press 'Q' in the video window to quit.\n")

    try:
        while stream.more():
            loop_start = time.time()
            frame = stream.read()
            if frame is None:
                break

            frame = resize_if_needed(frame)
            h, w = frame.shape[:2]

            if zones is None:
                zones = ZoneManager(w, h)

            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            zones.draw_all(frame)

            stats = {"vehicles": 0, "pedestrians": 0, "waiting": 0,
                     "crossing": 0, "yielding": 0}
            pedestrian_crossing_active = False
            unsafe_vehicle_detected = False

            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                class_id = int(detections.class_id[i]) if detections.class_id is not None else -1
                confidence = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
                class_name = class_names.get(class_id, "object")

                if class_name not in VEHICLE_CLASSES and class_name not in PEDESTRIAN_CLASSES:
                    continue

                foot_point = ((x1 + x2) / 2, y2)
                track_history[track_id].append(foot_point)

                status_label = None
                box_color = TEXT_WHITE

                if class_name in PEDESTRIAN_CLASSES:
                    stats["pedestrians"] += 1
                    box_color = BOX_COLOR_PEDESTRIAN
                    if zones.is_in_crosswalk(foot_point):
                        status_label = "CROSSING"
                        stats["crossing"] += 1
                        pedestrian_crossing_active = True
                        box_color = (50, 50, 235)
                    elif zones.is_in_waiting_zone(foot_point):
                        status_label = "WAITING"
                        stats["waiting"] += 1
                        box_color = (235, 160, 50)

                elif class_name in VEHICLE_CLASSES:
                    stats["vehicles"] += 1
                    box_color = BOX_COLOR_VEHICLE

                    history = track_history[track_id]
                    if len(history) >= 2:
                        dx = history[-1][0] - history[0][0]
                        dy = history[-1][1] - history[0][1]
                        avg_speed = (dx ** 2 + dy ** 2) ** 0.5 / max(1, len(history) - 1)
                    else:
                        avg_speed = 0.0

                    is_stopped = avg_speed < SPEED_STOPPED_THRESHOLD
                    near_or_in_crosswalk = zones.is_in_stop_zone(foot_point) or \
                        zones.is_in_near_crosswalk(foot_point) or zones.is_in_crosswalk(foot_point)

                    if near_or_in_crosswalk and is_stopped:
                        status_label = "STOPPED"
                        box_color = (60, 200, 60)
                        if pedestrian_crossing_active:
                            stats["yielding"] += 1
                    elif near_or_in_crosswalk and not is_stopped:
                        status_label = "MOVING"
                        box_color = (0, 165, 255)
                        if pedestrian_crossing_active:
                            status_label = "NOT YIELDING"
                            box_color = BOX_COLOR_WARNING
                            unsafe_vehicle_detected = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
                id_txt = f"#{track_id} " if track_id >= 0 else ""
                label = f"{id_txt}{class_name} {confidence:.2f}"
                draw_label(frame, label, x1, y1 - 6, box_color)
                if status_label:
                    draw_label(frame, status_label, x1, y2 + 20, box_color)

            if unsafe_vehicle_detected:
                draw_collision_banner(frame, w)

            frame_time = time.time() - loop_start
            fps_deque.append(1.0 / frame_time if frame_time > 0 else 0.0)
            avg_fps = sum(fps_deque) / len(fps_deque)

            draw_stats_panel(frame, stats, avg_fps)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                print("[i] 'Q' pressed. Exiting ...")
                break

    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print("[OK] Session ended cleanly.")


if __name__ == "__main__":
    run()
