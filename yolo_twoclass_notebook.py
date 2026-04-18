## ── Cell 1: Imports ──────────────────────────────────────────────────────────
import os
import xml.etree.ElementTree as ET
import shutil
import random
import yaml
import glob
import cv2
from pathlib import Path


## ── Cell 2: Directories (change these to your local paths) ───────────────────
XML_DIR    = r"\\JUNGMINN\Users\jungm\Documents\GitHub\jungminnking-econ8310_semester_project\Annotations"
VIDEO_DIR  = r"\\JUNGMINN\Users\jungm\Documents\GitHub\jungminnking-econ8310_semester_project\Raw Videos"
OUTPUT_DIR = r"\\JUNGMINN\Users\jungm\Documents\GitHub\jungminnking-econ8310_semester_project\Output"

MODEL_SIZE  = "yolov8n.pt"
EPOCHS      = 50
IMG_SIZE    = 640
VAL_SPLIT   = 0.2
RANDOM_SEED = 42


## ── Cell 3: Pair each XML with its .mov video ────────────────────────────────
# Same logic as script 2's all_pairs loop
# Warns for any XML with no matching video, uses ALL pairs (no 20% cap)
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


## ── Cell 4: Load all frames from a video (same as script 2) ──────────────────
def load_all_frames(video_path):
    cap    = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)   # keep BGR — cv2.imwrite expects BGR
    cap.release()
    return frames


## ── Cell 5: Extract annotated frames from videos and save as JPEGs ───────────
# Bridge between script 2's video loader and YOLO's disk format:
# Instead of returning tensors, we write each annotated frame to disk.
def parse_and_extract(all_pairs, output_dir):
    raw_dir = Path(output_dir) / "raw_frames"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_annotations = []
    total_boxes     = 0
    skipped_outside = 0
    skipped_oob     = 0

    for xml_path, video_path in all_pairs:
        print(f"  Processing: {Path(xml_path).stem}")

        frames = load_all_frames(video_path)
        if not frames:
            print(f"    WARNING: could not read frames from {Path(video_path).name}")
            continue

        total_f      = len(frames)
        img_h, img_w = frames[0].shape[:2]

        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as e:
            print(f"    WARNING: could not parse {Path(xml_path).name}: {e}")
            continue

        # Collect boxes per frame across all tracks (script 2 iterates tracks)
        frame_boxes = {}
        for track in root.findall("track"):
            for box in track.findall("box"):

                if int(box.attrib.get("outside", 0)) == 1:   # script 2: skip invisible
                    skipped_outside += 1
                    continue

                frame_idx = int(box.attrib["frame"])
                if frame_idx >= total_f:                      # script 2: skip out-of-range
                    skipped_oob += 1
                    continue

                try:
                    xtl = float(box.attrib["xtl"])
                    ytl = float(box.attrib["ytl"])
                    xbr = float(box.attrib["xbr"])
                    ybr = float(box.attrib["ybr"])
                except (KeyError, ValueError):
                    continue

                if xbr <= xtl or ybr <= ytl:
                    continue

                # Read moving attribute (script 2 logic verbatim)
                moving_attr = box.find("attribute[@name='moving']")
                is_moving   = (
                    moving_attr is not None
                    and moving_attr.text is not None
                    and moving_attr.text.strip().lower() == "true"
                )

                frame_boxes.setdefault(frame_idx, []).append(
                    (xtl, ytl, xbr, ybr, is_moving)
                )

        # Write each annotated frame to disk as JPEG
        for frame_idx, boxes in frame_boxes.items():
            img_name = f"{Path(xml_path).stem}_frame{frame_idx:06d}.jpg"
            img_path = raw_dir / img_name
            cv2.imwrite(str(img_path), frames[frame_idx])

            yolo_boxes = [
                (xtl, ytl, xbr, ybr, img_w, img_h, is_moving)
                for (xtl, ytl, xbr, ybr, is_moving) in boxes
            ]
            all_annotations.append((str(img_path), yolo_boxes))
            total_boxes += len(yolo_boxes)

    print(f"\nExtracted   : {len(all_annotations)} annotated frames")
    print(f"Total boxes : {total_boxes}")
    print(f"Skipped (outside=1)    : {skipped_outside}")
    print(f"Skipped (beyond video) : {skipped_oob}")
    return all_annotations

annotations = parse_and_extract(all_pairs, OUTPUT_DIR)


## ── Cell 6: Convert pixel coords to YOLO format ──────────────────────────────
# class 0 = baseball_stationary
# class 1 = baseball_moving
def cvat_box_to_yolo(xtl, ytl, xbr, ybr, img_w, img_h, is_moving):
    cx       = max(0.0, min(1.0, ((xtl + xbr) / 2) / img_w))
    cy       = max(0.0, min(1.0, ((ytl + ybr) / 2) / img_h))
    w        = max(0.0, min(1.0, (xbr - xtl) / img_w))
    h        = max(0.0, min(1.0, (ybr - ytl) / img_h))
    class_id = 1 if is_moving else 0
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


## ── Cell 7: Shuffle and split 80% train / 20% val ────────────────────────────
# Deduplicate first
seen, unique = set(), []
for img_path, boxes in annotations:
    if img_path not in seen:
        seen.add(img_path)
        unique.append((img_path, boxes))

if len(unique) < len(annotations):
    print(f"Removed {len(annotations) - len(unique)} duplicate frames")

random.seed(RANDOM_SEED)
random.shuffle(unique)

n_val   = max(1, int(len(unique) * VAL_SPLIT))
val_set = unique[:n_val]
trn_set = unique[n_val:]

# Label breakdown
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
val_mov, val_stat = count_labels(val_set)
print(f"Label breakdown:")
print(f"  Train → moving: {trn_mov}  stationary: {trn_stat}  frames: {len(trn_set)}")
print(f"  Val   → moving: {val_mov}  stationary: {val_stat}  frames: {len(val_set)}")


## ── Cell 8: Write images and label .txt files to YOLO folder structure ────────
for split, data in [("train", trn_set), ("val", val_set)]:
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

print(f"Dataset ready: {len(trn_set)} train / {len(val_set)} val frames")


## ── Cell 9: Write dataset.yaml (2 classes) ───────────────────────────────────
yaml_path = Path(OUTPUT_DIR) / "dataset.yaml"
config = {
    "path"  : str(OUTPUT_DIR).replace("\\", "/"),
    "train" : "images/train",
    "val"   : "images/val",
    "nc"    : 2,
    "names" : ["baseball_stationary", "baseball_moving"]
}
with open(yaml_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)
print(f"YAML written: {yaml_path}")


## ── Cell 10: Train ───────────────────────────────────────────────────────────
from ultralytics import YOLO

print(f"Loading pre-trained model: {MODEL_SIZE}")
model = YOLO(MODEL_SIZE)

model.train(
    data      = str(yaml_path),
    epochs    = EPOCHS,
    imgsz     = IMG_SIZE,
    batch     = 2,
    device    = "mps",
    project   = str(Path(OUTPUT_DIR) / "runs"),
    name      = "baseball_detect",
    patience  = 50,
    hsv_h     = 0.015,
    hsv_s     = 0.7,
    hsv_v     = 0.4,
    fliplr    = 0.5,
    mosaic    = 1.0,
    degrees   = 10.0,
    translate = 0.1,
    scale     = 0.5,
    save      = True,
    plots     = True,
    verbose   = True,
)


## ── Cell 11: Evaluate — overall + per-class + label accuracy ─────────────────
runs = sorted(glob.glob(str(Path(OUTPUT_DIR) / "runs" / "baseball_detect*")))
best = Path(runs[-1]) / "weights" / "best.pt"
print(f"Evaluating: {best}")

model   = YOLO(str(best))
metrics = model.val(
    data   = str(yaml_path),
    imgsz  = IMG_SIZE,
    device = "mps",
)

print("\n========== OVERALL ==========")
print(f"mAP@0.5:      {metrics.box.map50:.4f}")
print(f"mAP@0.5-0.95: {metrics.box.map:.4f}")
print(f"Precision:    {metrics.box.mp:.4f}")
print(f"Recall:       {metrics.box.mr:.4f}")

print("\n========== PER-CLASS ==========")
for i, name in enumerate(["baseball_stationary", "baseball_moving"]):
    try:
        print(f"  [{i}] {name}")
        print(f"       Precision : {metrics.box.p[i]:.4f}")
        print(f"       Recall    : {metrics.box.r[i]:.4f}")
        print(f"       AP@0.5    : {metrics.box.ap50[i]:.4f}")
    except IndexError:
        print(f"  [{i}] {name} — not available")

# Label accuracy: does predicted class match ground-truth class?
print("\n========== LABEL ACCURACY ==========")
val_img_dir = Path(OUTPUT_DIR) / "images" / "val"
val_lbl_dir = Path(OUTPUT_DIR) / "labels" / "val"

correct_class, wrong_class, no_pred = 0, 0, 0

for img_path in sorted(val_img_dir.glob("*.jpg")):
    lbl_path = val_lbl_dir / (img_path.stem + ".txt")
    if not lbl_path.exists():
        continue

    gt_classes = set()
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                gt_classes.add(int(parts[0]))

    results      = model(str(img_path), imgsz=IMG_SIZE, conf=0.25, verbose=False)
    pred_classes = {int(box.cls.item()) for box in results[0].boxes}

    if not pred_classes:
        no_pred += 1
    elif pred_classes.issubset(gt_classes):
        correct_class += 1
    else:
        wrong_class += 1

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


## ── Cell 12: Save sample predictions ─────────────────────────────────────────
out_dir = Path(OUTPUT_DIR) / "sample_predictions"
out_dir.mkdir(exist_ok=True)

for img_path in list(val_img_dir.glob("*.jpg"))[:10]:
    results   = model(str(img_path), imgsz=IMG_SIZE, conf=0.25)
    annotated = results[0].plot()
    cv2.imwrite(str(out_dir / img_path.name), annotated)

print(f"Sample predictions saved to: {out_dir}")
print(f"\nAll done! Results saved to:\n  {OUTPUT_DIR}")
