## Importing all at once
import os
import xml.etree.ElementTree as ET
import shutil
import random
import yaml
import glob
import cv2
from pathlib import Path
from ultralytics import YOLO

## Directories 
# Originally your Config.py (change these to your local paths) 
XML_DIR    = r"D:\Github\jungminnking-econ8310_semester_project\Annotations"
VIDEO_DIR  = r"D:\Github\jungminnking-econ8310_semester_project\Raw Videos"
OUTPUT_DIR = r"D:\Github\jungminnking-econ8310_semester_project\Output"

## Settings
MODEL_SIZE  = "yolov8n.pt"  
EPOCHS      = 10            #Ephoch 150 -> 10 (could be reduced or increased depending on your pc's performance)
IMG_SIZE    = 640           
VAL_SPLIT   = 0.2           # 20% of frames held out for validation
RANDOM_SEED = 42            

## Pairing Each XML with Videos By Filename
# We're going to match each xml and video
# You'll see some of them are not matched just because someone hasn't finished annotations or mistyped filenames...
# By far, I'm able to identify 59 pairs of XML and Videos out of total 78
all_pairs = []
for fname in os.listdir(XML_DIR):
    if fname.endswith(".xml"):
        name       = os.path.splitext(fname)[0]          # common stem
        xml_path   = os.path.join(XML_DIR, fname)
        video_path = os.path.join(VIDEO_DIR, name + ".mov")
        if os.path.exists(video_path):
            all_pairs.append((xml_path, video_path))
        else:
            print(f"WARNING: no video found for {fname}")

print(f"Found {len(all_pairs)} matched XML/video pairs (all pairs used)")


### BEFOR THIS: "raw_frames" should be on OUTPUT_DIR, for example in my setting, r"D:\Github\jungminnking-econ8310_semester_project\Output\raw_frames"
## Parse XML annotations and match to extracted frames in raw_frames/
# It reads each XML, collects bounding boxes, then matches them to the JPEGs already sitting in raw_frames/.
# Output: annotations = list of (image_path, [boxes]) for every annotated frame
def parse_annotations(all_pairs, output_dir):
    raw_dir = Path(output_dir) / "raw_frames" # "raw_frames" should be on OUTPUT_DIR = r"D:\Github\jungminnking-econ8310_semester_project\Output\raw_frames"
    final_annotations = []
    total_boxes     = 0
    skipped_outside = 0
    for xml_path, video_path in all_pairs:
        stem = Path(xml_path).stem
        print(f"  Processing: {stem}")
        if not Path(xml_path).exists():
            print(f"    WARNING: XML not found, skipping: {stem}")
            continue
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as e:
            print(f"    WARNING: could not parse {stem}: {e}")
            continue

        # Get image dimensions from the first available frame for this stem
        sample_frames = sorted(raw_dir.glob(f"{stem}_frame*.jpg"))
        if not sample_frames:
            print(f"    WARNING: no frames found in raw_frames for {stem}")
            continue
        sample = cv2.imread(str(sample_frames[0]))
        img_h, img_w = sample.shape[:2]

        # Collect all boxes grouped by frame index
        frame_boxes = {}
        for track in root.findall("track"):
            for box in track.findall("box"):
                if int(box.attrib.get("outside", 0)) == 1:
                    skipped_outside += 1
                    continue
                frame_idx = int(box.attrib["frame"])
                try:
                    xtl = float(box.attrib["xtl"])
                    ytl = float(box.attrib["ytl"])
                    xbr = float(box.attrib["xbr"])
                    ybr = float(box.attrib["ybr"])
                except (KeyError, ValueError):
                    continue
                if xbr <= xtl or ybr <= ytl:
                    continue
                moving_attr = box.find("attribute[@name='moving']")
                is_moving = (
                    moving_attr is not None
                    and moving_attr.text is not None
                    and moving_attr.text.strip().lower() == "true"
                )
                frame_boxes.setdefault(frame_idx, []).append(
                    (xtl, ytl, xbr, ybr, is_moving)
                )
        if not frame_boxes:
            print(f"    WARNING: no valid boxes found in {stem}")
            continue

        # Match each JPEG on disk to its bounding boxes from the XML
        for img_path in sample_frames:
            frame_idx = int(img_path.stem.rsplit("_frame", 1)[1])
            if frame_idx not in frame_boxes:
                continue
            boxes = [
                (xtl, ytl, xbr, ybr, img_w, img_h, is_moving)
                for (xtl, ytl, xbr, ybr, is_moving) in frame_boxes[frame_idx]
            ]
            final_annotations.append((str(img_path), boxes))
            total_boxes += len(boxes)

    print(f"\nAnnotated frames : {len(final_annotations)}")
    print(f"Total boxes      : {total_boxes}")
    print(f"Skipped (outside=1) : {skipped_outside}")
    return final_annotations
annotations = parse_annotations(all_pairs, OUTPUT_DIR)

## Convert pixel coords to YOLO normalised format 
# YOLO expects coordinates as fractions of image width/height (0.0 to 1.0),
# Also assigns class id:
# class 0 = baseball_stationary
# class 1 = baseball_moving
def cvat_box_to_yolo(xtl, ytl, xbr, ybr, img_w, img_h, is_moving):
    cx       = max(0.0, min(1.0, ((xtl + xbr) / 2) / img_w))  # center x
    cy       = max(0.0, min(1.0, ((ytl + ybr) / 2) / img_h))  # center y
    w        = max(0.0, min(1.0, (xbr - xtl) / img_w))         # box width
    h        = max(0.0, min(1.0, (ybr - ytl) / img_h))         # box height
    class_id = 1 if is_moving else 0
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


## Deduplicate first in case any frame appears more than once
seen, unique = set(), []
for img_path, boxes in annotations:
    if img_path not in seen:
        seen.add(img_path)
        unique.append((img_path, boxes))

if len(unique) < len(annotations):
    print(f"Removed {len(annotations) - len(unique)} duplicate frames")

# Shuffle with fixed seed so results are reproducible
random.seed(RANDOM_SEED)
random.shuffle(unique)

## Shuffle and split 80% train / 20% val
# 80% train, 20% test
n_tst   = max(1, int(len(unique) * VAL_SPLIT))
tst_set = unique[:n_tst]
trn_set = unique[n_tst:]

# Count moving vs stationary in each split — useful to spot class imbalance
def count_labels(dataset):
    moving, stationary = 0, 0
    for _, boxes in dataset:
        for box in boxes:
            if box[6]:
                moving += 1
            else:
                stationary += 1
    return moving, stationary
trn_mov, trn_stat = count_labels(trn_set)
tst_mov, tst_stat = count_labels(tst_set)

print(f"Label breakdown:")
print(f"  Train → moving: {trn_mov}  stationary: {trn_stat}  frames: {len(trn_set)}")
print(f"  Test  → moving: {tst_mov}  stationary: {tst_stat}  frames: {len(tst_set)}")

## Write images and label .txt files into YOLO folder structure 
#   images/train/  
#   images/test/
#   labels/train/ 
#   labels/test/
# Each .txt file contains one line per ball in that frame:
#   class_id  center_x  center_y  width  height  (all normalised 0-1)
for split, data in [("train", trn_set), ("test", tst_set)]:
    img_dir = Path(OUTPUT_DIR) / "images" / split
    lbl_dir = Path(OUTPUT_DIR) / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path, boxes in data:
        img_path = Path(img_path)
        stem     = img_path.stem
        shutil.copy2(img_path, img_dir / img_path.name)
        with open(lbl_dir / f"{stem}.txt", "w") as f:
            for (xtl, ytl, xbr, ybr, iw, ih, is_moving) in boxes:
                f.write(cvat_box_to_yolo(xtl, ytl, xbr, ybr, iw, ih, is_moving) + "\n")
print(f"Dataset ready: {len(trn_set)} train / {len(tst_set)} test frames")

## Write dataset.yaml
# YOLO reads this file to find the images and know how many classes there are.
# nc = number of classes (was 1 in script 1, now 2)
# names = class labels in order — index 0 = stationary, index 1 = moving
yaml_path = Path(OUTPUT_DIR) / "dataset.yaml"
config = {
    "path"  : str(OUTPUT_DIR).replace("\\", "/"),
    "train" : "images/train",
    "val"   : "images/test", # YOLO internally calls this key "val"
    "nc"    : 2,
    "names" : ["baseball_stationary", "baseball_moving"]
}
with open(yaml_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)
print(f"YAML written: {yaml_path}")

## Train 
# Start from yolov8n.pt — a model pre-trained on millions of general images.
# Fine-tune it on our baseball data so it learns to find and classify our ball.
# Augmentations (hsv, fliplr, mosaic, etc.) artificially increase training variety
# by randomly transforming images each epoch — helps the model generalise better.
print(f"Loading pre-trained model: {MODEL_SIZE}")
model = YOLO(MODEL_SIZE)

model.train(
    data      = str(yaml_path),
    epochs    = EPOCHS,
    imgsz     = IMG_SIZE,
    batch     = 32,           # 2 is too slow, making it too much slow 
    device    = "cpu",        # Apple Silicon GPU; change to "cuda" for Windows GPU or "cpu"
    project   = str(Path(OUTPUT_DIR) / "runs"),
    name      = "baseball_detect",
    patience  = 10,           # stop early if val loss doesn't improve for 10 epochs
    hsv_h     = 0.015,        # random hue shift
    hsv_s     = 0.7,          # random saturation shift
    hsv_v     = 0.4,          # random brightness shift
    fliplr    = 0.5,          # 50% chance of horizontal flip per image
    mosaic    = 1.0,          # combine 4 images into one training sample
    degrees   = 10.0,         # random rotation up to ±10°
    translate = 0.1,          # random translation up to 10% of image size
    scale     = 0.5,          # random zoom in/out up to 50%
    save      = True,
    plots     = True,
    verbose   = True,
)

## Evaluate — overall + per-class + label accuracy
# Load the best checkpoint saved during training (lowest val loss)
runs = sorted(glob.glob(str(Path(OUTPUT_DIR) / "runs" / "baseball_detect*")))
best = Path(runs[-1]) / "weights" / "best.pt"
print(f"Evaluating: {best}")

model   = YOLO(str(best))
metrics = model.val(
    data   = str(yaml_path),
    imgsz  = IMG_SIZE,
    device = "cpu",   # Apple Silicon GPU; change to "cuda" for Windows GPU or "cpu"
)

# OVERALL — averaged across both classes
print("\n========== OVERALL ==========")
print(f"mAP@0.5:      {metrics.box.map50:.4f}")
# mAP@0.5: for each prediction, checks if the predicted box overlaps the true box
# by at least 50% (IoU). Averages precision across all confidence thresholds.
# e.g. 0.85 = model finds the ball in roughly the right spot 85% of the time

print(f"mAP@0.5-0.95: {metrics.box.map:.4f}")
# Stricter version: averages across IoU thresholds 50%→95%
# A box that's slightly off still gets penalized — harder to score well on

print(f"Precision:    {metrics.box.mp:.4f}")
# Of all boxes the model predicted, what fraction were real balls?
# High precision = model doesn't produce many false detections

print(f"Recall:       {metrics.box.mr:.4f}")
# Of all real balls in the images, what fraction did the model find?
# High recall = model doesn't miss many balls

# PER-CLASS — same metrics split by stationary vs moving
# Tells you if the model struggles more with one class than the other
# e.g. high AP on stationary but low AP on moving → need more moving training data
print("\n========== PER-CLASS ==========")
for i, name in enumerate(["baseball_stationary", "baseball_moving"]):
    try:
        print(f"  [{i}] {name}")
        print(f"       Precision : {metrics.box.p[i]:.4f}")
        print(f"       Recall    : {metrics.box.r[i]:.4f}")
        print(f"       AP@0.5    : {metrics.box.ap50[i]:.4f}")
    except IndexError:
        print(f"  [{i}] {name} — not available")

# LABEL ACCURACY — simpler check that ignores box location entirely
# Only asks: did the model predict the right class (moving vs stationary)?
# Even if the bounding box is slightly off, as long as the class label is correct
# it counts as correct here. Complement to mAP — if label accuracy is low,
# the model is fundamentally confused about the two classes, not just imprecise.
print("\n========== LABEL ACCURACY ==========")
tst_img_dir = Path(OUTPUT_DIR) / "images" / "test"
tst_lbl_dir = Path(OUTPUT_DIR) / "labels" / "test"

correct_class, wrong_class, no_pred = 0, 0, 0

for img_path in sorted(tst_img_dir.glob("*.jpg")):
    lbl_path = tst_lbl_dir / (img_path.stem + ".txt")
    if not lbl_path.exists():
        continue

    # Ground truth classes from the .txt label file
    gt_classes = set()
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                gt_classes.add(int(parts[0]))

    # Predicted classes from the model
    results      = model(str(img_path), imgsz=IMG_SIZE, conf=0.25, verbose=False)
    pred_classes = {int(box.cls.item()) for box in results[0].boxes}

    if not pred_classes:
        no_pred += 1                            # model detected nothing
    elif pred_classes.issubset(gt_classes):
        correct_class += 1                      # all predicted classes are correct
    else:
        wrong_class += 1                        # at least one wrong class predicted

total     = correct_class + wrong_class + no_pred
label_acc = correct_class / (correct_class + wrong_class) if (correct_class + wrong_class) > 0 else 0
print(f"Frames checked  : {total}")
print(f"Correct class   : {correct_class}")
print(f"Wrong class     : {wrong_class}")
print(f"No prediction   : {no_pred}")
print(f"Label accuracy  : {label_acc:.2%}")
print("=====================================")
print("mAP@0.5        = did the model find the ball at the right location?")
print("Label accuracy = did the model predict the right class (moving vs stationary)?")

## Save sample predictions 
# Runs the model on 10 val images and saves the annotated output as JPEGs
# so you can visually inspect what the model is detecting
out_dir = Path(OUTPUT_DIR) / "sample_predictions"
out_dir.mkdir(exist_ok=True)

for img_path in list(tst_img_dir.glob("*.jpg"))[:10]:
    results   = model(str(img_path), imgsz=IMG_SIZE, conf=0.25)
    annotated = results[0].plot()   # draws boxes + class labels on the image
    cv2.imwrite(str(out_dir / img_path.name), annotated)

print(f"Sample predictions saved to: {out_dir}")
print(f"\nAll done! Results saved to:\n  {OUTPUT_DIR}")
