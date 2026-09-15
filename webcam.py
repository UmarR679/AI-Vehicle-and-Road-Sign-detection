from ultralytics import YOLO
import cv2
from collections import Counter

MODEL_PATH = "yolo11n.pt"

ROAD_CLASSES = {
    "person", "bicycle", "car", "motorcycle",
    "bus", "truck", "traffic light"
}
VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame, verbose=False)
    result = results[0]
    counts = Counter()

    for box in result.boxes:
        class_id = int(box.cls[0])
        name = model.names[class_id]
        conf = float(box.conf[0])

        if name not in ROAD_CLASSES:
            continue

        counts[name] += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {conf:.0%}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    vehicles = sum(counts[c] for c in VEHICLE_CLASSES)

    cv2.rectangle(frame, (10, 10), (300, 125), (0, 0, 0), -1)
    cv2.putText(frame, "LIVE ROAD DETECTOR", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Vehicles: {vehicles}", (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"People: {counts['person']}", (20, 91),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Traffic lights: {counts['traffic light']}", (20, 114),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("YOLO Road Detector - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
