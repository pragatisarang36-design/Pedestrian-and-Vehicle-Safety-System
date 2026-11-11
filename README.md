# Pedestrian & Vehicle Safety System

A video-based computer vision prototype for observing pedestrian and vehicle activity around a crosswalk. It detects selected object classes, assigns tracking IDs, checks object foot points against predefined image zones, and displays a live annotated video with per-frame counts and a heuristic warning when a vehicle is moving near the crossing while a pedestrian is detected in it.

The application processes a video file. It does not control traffic signals, communicate with vehicles, or provide a certified collision prediction system.

## Problem and objectives

Pedestrians and vehicles can occupy the same crossing area, and a vehicle that continues moving as a pedestrian crosses may present a safety concern. This project demonstrates how object detection, multi-object tracking, and simple zone checks can be combined to make activity visible in video.

Its objectives are to:

- Detect people and common road vehicles in each video frame.
- Track detections across frames and show their tracker IDs.
- Classify pedestrian foot points as waiting or crossing when they fall inside the corresponding predefined zones.
- Estimate short-term vehicle motion from tracked image positions and label vehicles near the crossing as stopped or moving.
- Display counts and a visual warning for the implemented non-yielding heuristic.

## Technologies

- **Python** — application logic.
- **Ultralytics YOLOv8** — object detection using the `yolov8n.pt` model.
- **Supervision ByteTrack** — multi-object tracking and conversion of Ultralytics results to detections.
- **OpenCV** — video capture and display, image processing, polygon checks, and drawing overlays.
- **NumPy** — point and coordinate arrays.
- **Requests** and **tqdm** — downloading model/video files and showing download progress.
- Python standard library modules (`threading`, `queue`, `collections`, `time`, `os`, and `sys`) — video reader thread, track history, and runtime utilities.

## How it works

1. **Prepare the model.** On startup, the program checks for `models/yolov8n.pt`. If it is missing, it attempts to download the YOLOv8n weights from the URL configured in `app.py`.
2. **Select a video.** The program prompts for a video file path. Press Enter to use the fallback clip, which is downloaded to `demo_videos/demo_traffic_intersection.mp4` if it is not already present. An invalid supplied path also falls back to that clip.
3. **Read frames.** OpenCV reads the video on a background thread into a bounded queue. Frames wider than 1280 pixels are resized proportionally.
4. **Detect and track.** YOLO inference runs on each frame. Supervision's ByteTrack updates the detections with tracker IDs.
5. **Evaluate zones and motion.** The bottom-center (foot) point of each detection is tested against the zone polygons. Recent tracked foot points are also used to estimate a vehicle's average pixel displacement per frame.
6. **Render results.** OpenCV draws the zones, boxes, labels, statuses, a statistics panel, and—when the heuristic condition is met—a flashing warning banner. Press **Q** in the video window to exit.

## Implemented features

### Detection and tracking

The application uses the COCO class names returned by the loaded YOLO model and processes only `person`, `car`, `truck`, `bus`, and `motorcycle` detections. Other classes are ignored. Bounding boxes show class name and confidence; when available, the ByteTrack tracking ID is included.

### Zones

`zones.py` creates five polygons from percentages of the first processed frame's width and height:

- One **crosswalk** zone.
- Two **waiting** zones, one on each side.
- One **near-crosswalk** approach zone.
- One **vehicle stop** zone along the bottom of the frame.

The polygons are drawn as translucent overlays and their locations are fixed by the default layout in `zones.py`. The program does not provide a calibration or zone-editing interface.

Pedestrians are labeled **CROSSING** when their foot point is inside the crosswalk and **WAITING** when it is inside either waiting zone. Vehicles are assessed against the stop, near-crosswalk, and crosswalk zones.

### Motion, safety indication, and statistics

For each tracked object, the program keeps up to 15 recent foot-point positions. A vehicle is treated as stopped when the distance between the oldest and newest retained points, divided by the number of frame intervals, is below `1.6` pixels per frame. This is an image-space heuristic, not a real-world speed measurement.

A vehicle in or near the crossing zones is labeled **STOPPED** or **MOVING** by this heuristic. If a crossing pedestrian has already been encountered in that frame's detection loop, a moving vehicle in those zones is labeled **NOT YIELDING**, and the video displays a flashing **COLLISION WARNING - VEHICLE NOT YIELDING** banner. A stopped vehicle is counted as yielding under the same crossing-state condition.

The on-screen panel reports the current frame's vehicle, pedestrian, waiting, crossing, and yielding detection counts, along with a smoothed processing FPS estimate. These are not unique-object totals or historical analytics. There is no audio, external notification, event log, or persistent statistics output.

## Requirements and setup

Use Python 3.9 or newer. The dependencies and minimum versions are listed in `requirements.txt`:

- `ultralytics>=8.2.0`
- `opencv-python>=4.9.0`
- `supervision>=0.21.0`
- `numpy>=1.26.0`
- `tqdm>=4.66.0`
- `requests>=2.31.0`

A network connection is needed on first use if the model weights or fallback video are not already present. The application uses an OpenCV display window, so run it in a desktop session with GUI support.

## Installation

Clone the repository and change into its directory. Then create and activate a virtual environment and install the requirements.

### Windows PowerShell

```powershell
git clone <repository-url>
cd pedestrian-vehicle-safety-cv-main
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation due to its execution policy, use Command Prompt and activate with:

```bat
.venv\Scripts\activate.bat
```

### macOS or Linux

```bash
git clone <repository-url>
cd pedestrian-vehicle-safety-cv-main
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Replace `<repository-url>` with the URL of the repository you cloned. If the local folder has a different name, use that folder name in the `cd` command.

## Run on Windows

From PowerShell, with the environment activated and the current directory set to the project root:

```powershell
python app.py
```

At the prompt, enter a video file path (quoted paths are accepted), or press Enter to fetch and use the fallback clip. If the model download fails, place a compatible YOLOv8 `.pt` weights file at `models\yolov8n.pt`. The program opens a resizable video window; focus that window and press **Q** to stop processing.

## Limitations

- Zone coordinates are hard-coded as proportions of the frame. They may not correspond to the actual road layout, camera perspective, or crosswalk in a supplied video.
- The system does not calibrate perspective, estimate physical distances, or convert image motion to vehicle speed. The stopped/moving threshold is expressed in pixels per frame and depends on video resolution and viewpoint.
- The crossing warning is a simple geometric and motion heuristic. It does not estimate trajectories, time to collision, right of way, or actual impact risk. Its crossing flag is set while detections are processed, so the result can also depend on detection order within a frame.
- Detection and tracking can fail or change IDs because of occlusion, image quality, camera motion, frame rate, or model limitations.
- Counts are frame-level detections, not deduplicated people or vehicles. No event history is saved.
- Processing speed depends on the computer, video, and available hardware. The displayed FPS is a smoothed processing estimate, not a performance guarantee.
- The fallback URL is a general video asset and is not guaranteed to depict a traffic intersection or crosswalk.
- This is a video-file prototype; it does not connect to a live camera, traffic controller, alarm device, or external service beyond initial file downloads.

## Disclaimer

This project is provided for educational and demonstration purposes. Its detections, zone labels, counts, and warnings may be inaccurate and must not be used as the sole basis for traffic control, operational safety decisions, law enforcement, or emergency response. A qualified review and location-specific validation are required before any real-world use.

## Developer

**Pragati Sarang**
