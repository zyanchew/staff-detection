from ultralytics import YOLO
import cv2
import os
import csv

# ---------------- CONFIG ----------------
PERSON_MODEL = "yolov8l.pt"  # pretrained COCO
TAG_MODEL = "runs/detect/tag_model/weights/best.pt"  # your tag model
VIDEO_IN = "sample.mp4"
VIDEO_OUT = "output_debug_tag_boxes.mp4"
CROP_DIR = "staff_crops"
CSV_OUT = "detections.csv"

TAG_CONF_THRES = 0.15  # lower threshold for debugging
DEBUG_MODE = True      # True = draw all persons in red
BOX_EXPAND_RATIO = 0.05 # expand person box slightly for tag matching

os.makedirs(CROP_DIR, exist_ok=True)
# ----------------------------------------

# Load models
person_model = YOLO(PERSON_MODEL)
tag_model = YOLO(TAG_MODEL)

# Video setup
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video {VIDEO_IN}")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) 
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (width, height))

# CSV setup
csv_file = open(CSV_OUT, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "person_id", "x1", "y1", "x2", "y2", "tag_conf"])

frame_id = 0
print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    # 1. Detect persons
    p_results = person_model(frame, imgsz=640, conf=0.1, verbose=False)
    person_boxes = []
    if len(p_results) > 0 and hasattr(p_results[0], "boxes"):
        for box, cls in zip(p_results[0].boxes.xyxy.cpu().numpy(),
                            p_results[0].boxes.cls.cpu().numpy()):
            if int(cls) == 0:  # person class
                person_boxes.append(box)  # [x1,y1,x2,y2]

    # 2. Draw all persons if DEBUG_MODE
    for pid, pbox in enumerate(person_boxes):
        px1, py1, px2, py2 = map(int, pbox)
        if DEBUG_MODE:
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0,0,255), 2)  # red for all persons

    # 3. Detect tags
    t_results = tag_model(frame, imgsz=640, conf=0.1, verbose=False)
    tag_boxes = []
    tag_confs = []
    if len(t_results) > 0 and hasattr(t_results[0], "boxes"):
        for box, conf in zip(t_results[0].boxes.xyxy.cpu().numpy(),
                             t_results[0].boxes.conf.cpu().numpy()):
            if conf >= TAG_CONF_THRES:
                tag_boxes.append(box)
                tag_confs.append(float(conf))
                # Draw small tag box on frame
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,165,255), 2)  # orange for tag
                cv2.putText(frame, f"{conf:.2f}", (x1, max(15, y1-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

    # 4. Match tags to persons
    for tbox, tconf in zip(tag_boxes, tag_confs):
        tx1, ty1, tx2, ty2 = map(int, tbox)
        t_cx = (tx1 + tx2) // 2
        t_cy = (ty1 + ty2) // 2
        for pid, pbox in enumerate(person_boxes):
            px1, py1, px2, py2 = map(int, pbox)

            # Expand person box slightly
            pw = px2 - px1
            ph = py2 - py1
            px1_exp = max(0, px1 - int(pw * BOX_EXPAND_RATIO))
            px2_exp = min(width, px2 + int(pw * BOX_EXPAND_RATIO))
            py1_exp = max(0, py1 - int(ph * BOX_EXPAND_RATIO))
            py2_exp = min(height, py2 + int(ph * BOX_EXPAND_RATIO))

            if px1_exp <= t_cx <= px2_exp and py1_exp <= t_cy <= py2_exp:
                # Draw bounding box around full person (green)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0,220,0), 3)
                # Label and coordinates
                cv2.putText(frame, f"STAFF ({tconf:.2f})", (px1, max(15, py1-20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,0), 2)
                cv2.putText(frame, f"{px1},{py1},{px2},{py2}", (px1, max(35, py1-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 1)

                # Save full-person crop
                person_crop = frame[py1:py2, px1:px2]
                crop_name = os.path.join(CROP_DIR, f"frame{frame_id:05d}_person{pid}.jpg")
                cv2.imwrite(crop_name, person_crop)

                # Log to CSV
                csv_writer.writerow([frame_id, pid, px1, py1, px2, py2, f"{tconf:.3f}"])
                break  # one tag → one person

    # Write annotated frame to output video
    writer.write(frame)

# Cleanup
csv_file.close()
cap.release()
writer.release()
print(f"Done. Output video: {VIDEO_OUT}, crops: {CROP_DIR}, csv: {CSV_OUT}")