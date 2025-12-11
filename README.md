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
- Uses YOLOv8 pretrained on the COCO dataset (yolov8l.pt)
- Outputs bounding boxes + confidence scores
- Confidence = probability the detected object is a human
- Example: 0.89 means the model is 89% sure it detected a person
2. Tag Detection (yolo.ipynb)
- Uses a custom YOLO tag model trained on a [frame dataset](frames) extracted from sample.mp4.
- The object is labeled using [labelImg](https://pypi.org/project/labelImg/).
- The dataset was splited into [train set](data/images/train) and [val set](data/images/val).
- Only detects the staff name tag.
- Confidence = probability that the box contains a tag
3. Staff Identification
- A person is considered staff when the tag bounding box lies inside the person’s bounding box.

## Accuracy and Limitations
Current issues:
- The resolution of the sample video is low, motion blur, small tag; all these may contribute to the inaccuracy of detection
- The pretrained YOLO person model does not perfectly perform; thus, recognition is not accurate.
- To improve accuracy, improve accuracy/performance of person model, tag model perform relatively better, however, it could also be improve by increasing training set.
