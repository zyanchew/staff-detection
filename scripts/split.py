import os
import shutil
import random

# Paths
frames_dir = "frames"
labels_dir = "data/labels/train"
train_img_dir = "data/images/train"
val_img_dir = "data/images/val"

# List all images
images = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]

# Identify labeled images
labeled_images = []
for label_file in os.listdir(labels_dir):
    if label_file.endswith(".txt"):
        img_name = label_file.replace(".txt", ".jpg")
        labeled_images.append(img_name)

# Convert to set for speed
labeled_images = set(labeled_images)

# Split unlabeled images
unlabeled_images = [img for img in images if img not in labeled_images]

# 80/20 split for unlabeled
val_count = int(len(images) * 0.2)  # total 20%
# But train already has labeled, so sample only from unlabeled
val_candidates = min(len(unlabeled_images), val_count)

val_split = set(random.sample(unlabeled_images, val_candidates))
train_split = set(images) - val_split  # Everything else in train

# --- Move files ---
def move_images(img_list, dest_dir):
    for img in img_list:
        shutil.copy(os.path.join(frames_dir, img),
                    os.path.join(dest_dir, img))

# Move according to split
move_images(train_split, train_img_dir)
move_images(val_split, val_img_dir)

print("Done!")
print(f"Labeled images forced to train: {len(labeled_images)}")
print(f"Train images total: {len(train_split)}")
print(f"Val images total: {len(val_split)}")