# AI-Based Road Sign & Vehicle Detection Using YOLO

A simple real-time computer vision mini-project using YOLO and OpenCV.

## Features

- Detects cars, motorcycles, buses, trucks and bicycles
- Detects pedestrians
- Detects traffic lights
- Shows bounding boxes and confidence scores
- Counts detected vehicles
- Displays a HIGH TRAFFIC alert when 8+ vehicles are visible
- Supports road video and webcam input

## Requirements

- Python 3.9+
- Windows/macOS/Linux
- Internet connection on first run so Ultralytics can download `yolo11n.pt`

## Installation

### Windows

Open a terminal in this folder:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Put your road video in this folder and name it:

`road.mp4`

Then run:

```bash
python main.py
```

Or double-click `run_video.bat`.

## Webcam

Activate the environment and run:

```bash
python webcam.py
```

Or double-click `run_webcam.bat`.

Press `Q` to quit the detection window.

## Important note about traffic signs

The included pretrained YOLO model is intended for a quick working demo. It can detect road-related objects such as traffic lights and vehicles, but it does NOT reliably recognize individual signs such as STOP, SPEED LIMIT, NO ENTRY, LEFT TURN, etc.

For those individual signs, train or use a YOLO model trained on a traffic-sign dataset and replace:

`yolo11n.pt`

with the trained model, for example:

`best.pt`

## Project flow

Road video/webcam
       ↓
OpenCV
       ↓
YOLO object detection
       ↓
Vehicles / people / traffic lights
       ↓
Bounding boxes + confidence
       ↓
Object counting + traffic alert

## Demo explanation

"This project uses YOLO for real-time object detection and OpenCV for video processing. The system analyzes each video frame and detects road objects such as cars, motorcycles, buses, trucks, bicycles, pedestrians and traffic lights. It draws bounding boxes, displays confidence scores and counts vehicles. The project can be extended with a traffic-sign-specific YOLO model to recognize STOP, speed-limit and other road signs."

## Troubleshooting

If `python` is not recognized, install Python and enable "Add Python to PATH".

If the video does not open, make sure `road.mp4` is in the same folder as `main.py`.

If the webcam does not open, close other applications using the camera.
