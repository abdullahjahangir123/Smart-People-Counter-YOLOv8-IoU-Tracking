# people_counter.py
import sys
from ultralytics import YOLO
import cv2
import os

# Accept input/output via argv (so frontend can call it)
# usage: python people_counter.py input_path output_path
if len(sys.argv) >= 2:
    video_path = sys.argv[1]
else:
    video_path = "uploaded_video.mp4"

if len(sys.argv) >= 3:
    output_path = sys.argv[2]
else:
    output_path = "output_with_tracking.mp4"

if not os.path.exists(video_path):
    print(f"ERROR: input video file not found: {video_path}")
    sys.exit(1)

# load model (will download if missing)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Processing video: {video_path} -> {output_path}")

unique_ids = 0
tracks = {}
iou_threshold = 0.3

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
    area2 = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processing frame {frame_count}...")

    # predict people
    results = model(frame, classes=0, verbose=False)[0]

    current_boxes = []
    for box in results.boxes:
        if float(box.conf[0]) > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_boxes.append([x1, y1, x2, y2])

    new_tracks = {}
    for box in current_boxes:
        best_iou = 0
        best_id = None
        for tid, old_box in tracks.items():
            iou = calculate_iou(old_box, box)
            if iou > best_iou:
                best_iou = iou
                best_id = tid

        if best_iou > iou_threshold and best_id is not None:
            new_tracks[best_id] = box
        else:
            unique_ids += 1
            new_tracks[unique_ids] = box

    tracks = new_tracks

    for tid, box in tracks.items():
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)

    cv2.putText(frame, f"Total Unique People: {unique_ids}",
                (20, 70), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print("="*60)
print(f"Video saved: {output_path}")
print(f"Total Unique People Detected: {unique_ids}")
print("="*60)
