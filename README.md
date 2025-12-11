# Staff Detection Using YOLO (Person+Tag Recognition)
This assignment implements a two-stage YOLO-based detection pipeline that identifies staff members in a video by detecting persons and verifying whether they are wearing a name tag.

## Overview
This system combines two YOLO models:
1. Person Detection - [Pretrained YOLOv8 Model](https://docs.ultralytics.com/models/yolov8/)
2. Tag Detection - Custom [YOLO tag model](runs/detect/tag_model/weights/best.pt)
## Final Output
[Output video](output_debug_tag_boxes.mp4)
- Staff with tag -> green bounding box
- Person without tag -> red bounding box
- Detected tag -> orange bounding box
- The value indicates confidence scores.
The coordinates of staff is recorded in [detections.csv](detections.csv).

## Algorithm Workflow
1. Person Detection
- Uses YOLOv8 pretrained on the COCO dataset (yolov8l.pt).
- Outputs bounding boxes + confidence scores
- Confidence = probability the detected object is a human
- Example: 0.89 means the model is 89% sure it detected a person
2. Tag Detection 
- Uses a custom [YOLO tag model](scripts/yolo.ipynb) trained on a [frame dataset](frames) extracted from sample.mp4.
- The object is labeled using [labelImg](https://pypi.org/project/labelImg/).
- The dataset was splited into [train set](data/images/train) and [val set](data/images/val).
- Only detects the staff name tag.
- Confidence = probability that the box contains a tag
3. Staff Identification
- A person is considered staff when the tag bounding box lies inside the person’s bounding box.

## Accuracy and Limitations
While the system is functional, several factors currently limit its detection accuracy:
1. Video Quality Constraints
- The sample video used for testing has low resolution, motion blur, and a small, low-visibility staff tag.
- These factors significantly reduce the model’s ability to correctly detect both persons and tags.
- Improving video quality (higher resolution, stable camera, better lighting) will directly improve detection performance.
2. Pretrained YOLO Person Detection Limitations
- The system relies on a pretrained YOLOv8 person model, which may fail under: Unusual body poses, low light, crowded scenes.
- When the person detector misses a person or produces inaccurate bounding boxes, the tag detector cannot function correctly.
Potential Improvement:
Fine-tune the person model on specific environment (e.g., shopping mall, office, event space). Even a small custom dataset can significantly increase performance.
3. Tag Detector Model Performance
Fine-tuned tag model performs comparatively well, but accuracy still depends on:
- How clearly the tag appears
- Variations in tag design, size, orientation
- Distance from camera
Expanding the training data to include more angles, distances, and lighting conditions will make the model more robust.
