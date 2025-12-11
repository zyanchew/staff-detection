import cv2
import os

# -----------------------------
# CONFIG
video_path = "sample.mp4"       # your input video
output_dir = "frames"           # folder to save frames
interval_sec = 0.2                # save a frame every 0.4 seconds
# -----------------------------

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)   # frames per second of video
frame_interval = int(fps * interval_sec)  # convert seconds to frame count

frame_id = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save frame every `frame_interval`
    if frame_id % frame_interval == 0:
        filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1
        print(f"Saved {filename}")

    frame_id += 1

cap.release()
print(f"Done! Total frames saved: {saved_count}")
