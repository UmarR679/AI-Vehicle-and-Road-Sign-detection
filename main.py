from ultralytics import YOLO
import cv2
import os
from collections import Counter

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "yolo11n.pt"

# ==========================================
# SUPPORTED FILE TYPES
# ==========================================

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]

VIDEO_EXTENSIONS = [
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv"
]

# ==========================================
# LOAD YOLO
# ==========================================

print("================================")
print(" AI ROAD DETECTION SYSTEM")
print("================================")

print("\nLoading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully!")


# ==========================================
# FIND IMAGE OR VIDEO AUTOMATICALLY
# ==========================================

files = os.listdir(".")

input_file = None

for file in files:

    extension = os.path.splitext(file)[1].lower()

    if extension in IMAGE_EXTENSIONS or extension in VIDEO_EXTENSIONS:

        # Ignore generated output
        if file.lower() != "output.jpg":

            input_file = file
            break


if input_file is None:

    print("\nERROR: No image or video found!")

    print("\nPut an image or video in this folder.")

    print("Supported images:")
    print("JPG, JPEG, PNG, BMP, WEBP")

    print("\nSupported videos:")
    print("MP4, AVI, MOV, MKV, WMV")

    input("\nPress ENTER to exit...")

    exit()


print("\nInput file found:")
print(input_file)


# ==========================================
# OBJECT CLASSES
# ==========================================

ROAD_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light"
}

VEHICLE_CLASSES = {
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck"
}


# ==========================================
# DETECTION FUNCTION
# ==========================================

def detect(frame):

    results = model(
        frame,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    counts = Counter()

    for box in result.boxes:

        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        confidence = float(box.conf[0])

        if class_name not in ROAD_CLASSES:
            continue

        counts[class_name] += 1

        # Bounding box
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # Draw box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = f"{class_name} {confidence:.0%}"

        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # ======================================
    # VEHICLE COUNT
    # ======================================

    vehicle_count = sum(
        counts[x]
        for x in VEHICLE_CLASSES
    )

    # ======================================
    # DASHBOARD
    # ======================================

    cv2.rectangle(
        frame,
        (10, 10),
        (350, 190),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "AI ROAD DETECTOR",
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (25, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Cars: {counts['car']}",
        (25, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Motorcycles: {counts['motorcycle']}",
        (25, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Buses: {counts['bus']}",
        (25, 162),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"People: {counts['person']}",
        (25, 187),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    # Traffic warning
    if vehicle_count >= 8:

        cv2.putText(
            frame,
            "HIGH TRAFFIC!",
            (25, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    return frame, counts


# ==========================================
# DETERMINE INPUT TYPE
# ==========================================

extension = os.path.splitext(
    input_file
)[1].lower()


# ==========================================
# IMAGE
# ==========================================

if extension in IMAGE_EXTENSIONS:

    print("\nInput type: IMAGE")

    image = cv2.imread(input_file)

    if image is None:

        print("ERROR: Could not read image.")

        input("\nPress ENTER to exit...")

        exit()

    print("Running YOLO detection...")

    output, counts = detect(image)

    # Save result
    cv2.imwrite(
        "output.jpg",
        output
    )

    print("\n================================")
    print("DETECTION COMPLETE")
    print("================================")

    print("\nObjects detected:")

    for name, count in counts.items():

        print(
            f"{name}: {count}"
        )

    print(
        "\nOutput saved as: output.jpg"
    )

    # Resize for display
    height, width = output.shape[:2]

    max_width = 1200
    max_height = 800

    scale = min(
        max_width / width,
        max_height / height,
        1
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    display = cv2.resize(
        output,
        (new_width, new_height)
    )

    cv2.imshow(
        "YOLO Road Detection - Image",
        display
    )

    print("\nPress ANY KEY on the image window to close.")

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ==========================================
# VIDEO
# ==========================================

elif extension in VIDEO_EXTENSIONS:

    print("\nInput type: VIDEO")

    cap = cv2.VideoCapture(
        input_file
    )

    if not cap.isOpened():

        print("ERROR: Could not open video.")

        input("\nPress ENTER to exit...")

        exit()

    print("Running YOLO video detection...")
    print("Press Q to stop.")

    while True:

        success, frame = cap.read()

        if not success:

            break

        output, counts = detect(frame)

        # Resize large videos
        height, width = output.shape[:2]

        max_width = 1200

        if width > max_width:

            scale = max_width / width

            output = cv2.resize(
                output,
                (
                    int(width * scale),
                    int(height * scale)
                )
            )

        cv2.imshow(
            "YOLO Road Detection - Video",
            output
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    cap.release()

    cv2.destroyAllWindows()

    print("\nVideo detection finished.")


print("\nProgram finished.")