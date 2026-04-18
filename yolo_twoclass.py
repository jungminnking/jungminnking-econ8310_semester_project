import os
import xml.etree.ElementTree as ET
import shutil
import random
import yaml
import glob
import cv2
from pathlib import Path

# Load user-specific paths and settings from config.py
try:
    from config import (
        XML_DIR, VIDEO_DIR, OUTPUT_DIR,
        MODEL_SIZE, EPOCHS, IMG_SIZE, VAL_SPLIT, RANDOM_SEED
    )
except ImportError:
    print("ERROR: config.py not found.")
    print("Copy config.py, fill in your local paths, and try again.")
    exit(1)

## Setting-up Directories
# I've downloaded the whole annotations & raw videos from our class Onedrive, then put them into my local directory
# What we're going to do is, therefore, pulling all videos together to train our models.
# Change these to your local paths  
XML_DIR   = r"\\JUNGMINN\Users\jungm\Documents\GitHub\jungminnking-econ8310_semester_project\Annotations"
VIDEO_DIR = r"\\JUNGMINN\Users\jungm\Documents\GitHub\jungminnking-econ8310_semester_project\Raw Videos"

## Pairing Each XML with Videos By Filename
# We're going to match each xml and video
# You'll see some of them are not matched just because someone hasn't finished annotations or mistyped filenames...
# By far, I'm able to identify 59 pairs of XML and Videos out of total 78
def get_all_pairs(xml_dir, video_dir):
    xml_dir   = Path(xml_dir)
    video_dir = Path(video_dir)
    all_pairs = []

    for xml_path in sorted(xml_dir.glob("*.xml")):
        name       = xml_path.stem    # Common Name                 
        video_path = video_dir / (name + ".mov")
        if video_path.exists():
            all_pairs.append((xml_path, video_path))
        else:
            print(f"WARNING: no video found for {xml_path.name}")

    print(f"Found {len(all_pairs)} matched XML/video pairs (all pairs used)")
    return all_pairs

print(f"Found {len(all_pairs)} matched XML/video pairs")
# ── STEP 2: Extract annotated frames from videos and save as JPEGs ────────────
# This is the bridge between script 2's video-based loader and YOLO's disk format.
# Script 2 loaded all frames into RAM as tensors.
# Here we load them the same way but write annotated frames to disk as JPEGs
# so YOLO can read them normally.

def load_all_frames(video_path):
    """
    Read every frame from a .mov file.
    Same as script 2's BaseballDataset._load_all_frames(),
    except we keep BGR (no cvtColor) since cv2.imwrite expects BGR.
    """
    cap    = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def parse_and_extract(all_pairs, output_dir):
    """
    For every XML/video pair:
      1. Load all video frames into memory  (script 2 style)
      2. Parse the XML for track-level box annotations with moving attribute
      3. Write each annotated frame as a JPEG to output_dir/raw_frames/
      4. Return list of (img_path, boxes) where each box is
         (xtl, ytl, xbr, ybr, img_w, img_h, is_moving)

    Follows script 2 rules:
      - outside=1 boxes are skipped
      - frames beyond video length are skipped
      - moving label read from attribute[@name='moving']
    """
    raw_dir = Path(output_dir) / "raw_frames"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_annotations = []
    total_boxes     = 0
    skipped_outside = 0
    skipped_oob     = 0

    for xml_path, video_path in all_pairs:
        print(f"  Processing: {xml_path.stem}")

        frames = load_all_frames(video_path)
        if not frames:
            print(f"    WARNING: could not read frames from {video_path.name}")
            continue

        total_f      = len(frames)
        img_h, img_w = frames[0].shape[:2]

        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as e:
            print(f"    WARNING: could not parse {xml_path.name}: {e}")
            continue

        # Collect all boxes per frame across every track in this XML
        # { frame_idx: [(xtl, ytl, xbr, ybr, is_moving), ...] }
        frame_boxes = {}

        for track in root.findall("track"):       # same loop structure as script 2
            for box in track.findall("box"):

                # Skip invisible balls (script 2: outside == 1 → continue)
                if int(box.attrib.get("outside", 0)) == 1:
                    skipped_outside += 1
                    continue

                frame_idx = int(box.attrib["frame"])

                # Skip if beyond actual video length (script 2 check)
                if frame_idx >= total_f:
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
            img_name = f"{xml_path.stem}_frame{frame_idx:06d}.jpg"
            img_path = raw_dir / img_name
            cv2.imwrite(str(img_path), frames[frame_idx])

            yolo_boxes = [
                (xtl, ytl, xbr, ybr, img_w, img_h, is_moving)
                for (xtl, ytl, xbr, ybr, is_moving) in boxes
            ]
            all_annotations.append((str(img_path), yolo_boxes))
            total_boxes += len(yolo_boxes)

    print(f"\nExtracted {len(all_annotations)} annotated frames to disk")
    print(f"Total bounding boxes   : {total_boxes}")
    print(f"Skipped (outside=1)    : {skipped_outside}")
    print(f"Skipped (beyond video) : {skipped_oob}")
    return all_annotations


# ── STEP 3: YOLO label conversion ─────────────────────────────────────────────

def cvat_box_to_yolo(xtl, ytl, xbr, ybr, img_w, img_h, is_moving):
    """
    Pixel coords → YOLO normalised format.
    class 0 = baseball_stationary
    class 1 = baseball_moving
    """
    cx       = max(0.0, min(1.0, ((xtl + xbr) / 2) / img_w))
    cy       = max(0.0, min(1.0, ((ytl + ybr) / 2) / img_h))
    w        = max(0.0, min(1.0, (xbr - xtl) / img_w))
    h        = max(0.0, min(1.0, (ybr - ytl) / img_h))
    class_id = 1 if is_moving else 0
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# ── STEP 4: 80/20 split → YOLO folder structure ───────────────────────────────

def build_yolo_dataset(annotations):
    """
    Shuffle all frames, split 80% train / 20% val,
    copy JPEGs and write .txt label files into YOLO structure.
    Prints moving/stationary counts for both splits.
    """
    # Deduplicate
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
    print(f"\nLabel breakdown:")
    print(f"  Train → moving: {trn_mov}  stationary: {trn_stat}  frames: {len(trn_set)}")
    print(f"  Val   → moving: {val_mov}  stationary: {val_stat}  frames: {len(val_set)}")

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
                    f.write(
                        cvat_box_to_yolo(xtl, ytl, xbr, ybr, iw, ih, is_moving) + "\n"
                    )

    print(f"Dataset ready: {len(trn_set)} train / {len(val_set)} val frames")


# ── STEP 5: YAML ──────────────────────────────────────────────────────────────

def write_yaml():
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
    return str(yaml_path)


# ── STEP 6: Train (unchanged from script 1) ───────────────────────────────────

def train_model(yaml_path):
    from ultralytics import YOLO

    print(f"\nLoading pre-trained model: {MODEL_SIZE}")
    model = YOLO(MODEL_SIZE)
    print(f"Training for {EPOCHS} epochs...\n")

    model.train(
        data      = yaml_path,
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


# ── STEP 7: Evaluate with per-class + label accuracy ─────────────────────────

def evaluate_model():
    from ultralytics import YOLO

    runs = sorted(glob.glob(str(Path(OUTPUT_DIR) / "runs" / "baseball_detect*")))
    if not runs:
        print("No trained model found.")
        return

    best = Path(runs[-1]) / "weights" / "best.pt"
    if not best.exists():
        print("No best.pt found.")
        return

    print(f"\nEvaluating: {best}")
    model   = YOLO(str(best))
    metrics = model.val(
        data   = str(Path(OUTPUT_DIR) / "dataset.yaml"),
        imgsz  = IMG_SIZE,
        device = "mps",
    )

    print("\n========== OVERALL RESULTS ==========")
    print(f"mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"mAP@0.5-0.95:  {metrics.box.map:.4f}")
    print(f"Precision:     {metrics.box.mp:.4f}")
    print(f"Recall:        {metrics.box.mr:.4f}")

    print("\n========== PER-CLASS RESULTS ==========")
    for i, name in enumerate(["baseball_stationary", "baseball_moving"]):
        try:
            print(f"  [{i}] {name}")
            print(f"       Precision : {metrics.box.p[i]:.4f}")
            print(f"       Recall    : {metrics.box.r[i]:.4f}")
            print(f"       AP@0.5    : {metrics.box.ap50[i]:.4f}")
        except IndexError:
            print(f"  [{i}] {name} — metrics not available")

    # Label accuracy: does the predicted class match ground-truth class?
    print("\n========== LABEL ACCURACY (class correctness) ==========")
    val_img_dir = Path(OUTPUT_DIR) / "images" / "val"
    val_lbl_dir = Path(OUTPUT_DIR) / "labels" / "val"

    correct_class = 0
    wrong_class   = 0
    no_pred       = 0

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

    total = correct_class + wrong_class + no_pred
    if total > 0:
        label_acc = correct_class / (correct_class + wrong_class) if (correct_class + wrong_class) > 0 else 0
        print(f"Frames checked  : {total}")
        print(f"Correct class   : {correct_class}")
        print(f"Wrong class     : {wrong_class}")
        print(f"No prediction   : {no_pred}")
        print(f"Label accuracy  : {label_acc:.2%}")
    print("=========================================")
    print("mAP@0.5        = did the model find the ball at the right location?")
    print("Label accuracy = did the model predict the right class (moving vs stationary)?")


def save_sample_predictions():
    from ultralytics import YOLO

    best = Path(OUTPUT_DIR) / "runs" / "baseball_detect" / "weights" / "best.pt"
    if not best.exists():
        return

    model   = YOLO(str(best))
    val_dir = Path(OUTPUT_DIR) / "images" / "val"
    out_dir = Path(OUTPUT_DIR) / "sample_predictions"
    out_dir.mkdir(exist_ok=True)

    for img_path in list(val_dir.glob("*.jpg"))[:10]:
        results   = model(str(img_path), imgsz=IMG_SIZE, conf=0.25)
        annotated = results[0].plot()
        cv2.imwrite(str(out_dir / img_path.name), annotated)

    print(f"Sample predictions saved to: {out_dir}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Baseball Detection - Two-Class Pipeline (moving/stationary)")
    print("=" * 55)
    print(f"Model:        {MODEL_SIZE}")
    print(f"Epochs:       {EPOCHS}")
    print(f"Image size:   {IMG_SIZE}")
    print(f"XML folder:   {XML_DIR}")
    print(f"Video folder: {VIDEO_DIR}")
    print(f"Output:       {OUTPUT_DIR}")

    print("\n[1/6] Pairing XMLs with videos...")
    all_pairs = get_all_pairs(XML_DIR, VIDEO_DIR)
    if not all_pairs:
        print("\nERROR: No XML/video pairs found. Check XML_DIR and VIDEO_DIR.")
        exit(1)

    print("\n[2/6] Extracting annotated frames from videos...")
    annotations = parse_and_extract(all_pairs, OUTPUT_DIR)
    if not annotations:
        print("\nERROR: No annotated frames extracted.")
        exit(1)

    print("\n[3/6] Building YOLO dataset (80/20 split, 2 classes)...")
    build_yolo_dataset(annotations)

    print("\n[4/6] Writing dataset config...")
    yaml_path = write_yaml()

    print("\n[5/6] Training model...")
    train_model(yaml_path)

    print("\n[6/6] Evaluating model...")
    evaluate_model()

    print("\n[Bonus] Saving sample predictions...")
    save_sample_predictions()

    print(f"\nAll done! Results saved to:\n  {OUTPUT_DIR}")
