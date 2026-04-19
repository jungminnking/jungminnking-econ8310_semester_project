## learning_curve_experiment.py
## Standalone script for the learning curve experiment.
## Contains the full setup pipeline (annotation parsing → split → file writing),
## then sweeps over TRAIN_FRACTIONS instead of running a single baseline train.
## Run this independently — does not depend on the main script having run first.

## Importing all at once
import os
import xml.etree.ElementTree as ET
import shutil
import random
import yaml
import glob
import cv2
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe on headless/Windows machines
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

## Directories 
# Originally your Config.py (change these to your local paths) 
XML_DIR    = r"D:\Github\jungminnking-econ8310_semester_project\Annotations"
VIDEO_DIR  = r"D:\Github\jungminnking-econ8310_semester_project\Raw Videos"
OUTPUT_DIR = r"D:\Github\jungminnking-econ8310_semester_project\Output"

## Settings
MODEL_SIZE  = "yolov8n.pt"  
EPOCHS      = 10            # could be reduced or increased depending on your pc's performance
IMG_SIZE    = 640           
VAL_SPLIT   = 0.2           # 20% of frames held out for validation
RANDOM_SEED = 42            

## 7-point log-ish spacing chosen based on dataset size (2,523 total frames):
##   5%  → ~101 frames  : floor / severe underfit baseline
##   10% → ~202 frames  : marginal, informative
##   20% → ~404 frames  : first point where real learning should kick in
##   40% → ~808 frames  : solid mid-range
##   60% → ~1,211 frames: good upper-mid
##   80% → ~1,615 frames: near-full
##   100%→  2,019 frames: full training pool baseline
## Note: class imbalance is ~17:1 (stationary:moving). At low fractions the
## moving-ball metrics will be noisy — per-class curves are plotted separately.
TRAIN_FRACTIONS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]

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

## ═══════════════════════════════════════════════════════════════════════════
## LEARNING CURVE EXPERIMENT
## Trains a fresh model for each fraction in TRAIN_FRACTIONS, always evaluating
## on the same fixed test set (tst_set) so results are directly comparable.
##
## Folder structure created under OUTPUT_DIR:
##   lc_experiment/
##     pct_005/images/train   lc_experiment/pct_005/images/test
##     pct_005/labels/train   lc_experiment/pct_005/labels/test
##     pct_005/dataset.yaml
##     pct_010/ ...
##     ...
##   lc_runs/                     ← YOLO training outputs (weights, plots, etc.)
##   learning_curve_results.csv   ← one row per fraction
##   learning_curve.png           ← the actual plot
##
## Why a separate folder instead of reusing images/train?
##   Each fraction needs its own dataset.yaml pointing to its own image subset.
##   Mixing them with the main run would overwrite files and corrupt the baseline.
## ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("LEARNING CURVE EXPERIMENT")
print("="*60)

lc_dir  = Path(OUTPUT_DIR) / "lc_experiment"   # per-fraction datasets live here
lc_runs = Path(OUTPUT_DIR) / "lc_runs"          # YOLO training outputs live here

results_log = []   # accumulates one dict per fraction

for fraction in TRAIN_FRACTIONS:

    # ── how many frames this run gets ────────────────────────
    n_train = max(1, int(len(trn_set) * fraction))
    pct_tag = f"pct_{int(fraction * 100):03d}"   # e.g. "pct_005", "pct_100"

    # ── sample reproducibly ──────────────────────────────────
    # Different seed per fraction so subsets are not all identical prefixes,
    # but the same fraction always picks the same frames across re-runs.
    rng    = random.Random(RANDOM_SEED + int(fraction * 1000))
    subset = rng.sample(trn_set, n_train)

    # Count class balance in this subset — useful to flag at low fractions
    sub_mov, sub_stat = count_labels(subset)
    print(f"\n[{pct_tag}] {n_train} train frames  "
          f"(moving: {sub_mov}  stationary: {sub_stat})")

    # ── write images + labels for this fraction ──────────────
    frac_base = lc_dir / pct_tag
    for split, data in [("train", subset), ("test", tst_set)]:
        img_out = frac_base / "images" / split
        lbl_out = frac_base / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path, boxes in data:
            img_path = Path(img_path)
            shutil.copy2(img_path, img_out / img_path.name)
            with open(lbl_out / f"{img_path.stem}.txt", "w") as f:
                for (xtl, ytl, xbr, ybr, iw, ih, is_moving) in boxes:
                    f.write(cvat_box_to_yolo(xtl, ytl, xbr, ybr, iw, ih, is_moving) + "\n")

    # ── write dataset.yaml for this fraction ─────────────────
    frac_yaml = frac_base / "dataset.yaml"
    frac_config = {
        "path"  : str(frac_base).replace("\\", "/"),
        "train" : "images/train",
        "val"   : "images/test",   # YOLO calls this key "val" internally
        "nc"    : 2,
        "names" : ["baseball_stationary", "baseball_moving"]
    }
    with open(frac_yaml, "w") as f:
        yaml.dump(frac_config, f, default_flow_style=False)

    # ── train a fresh model from the same pretrained weights ─
    # Always start from yolov8n.pt so the only variable is training set size.
    # All other hyperparameters are identical to the main script's baseline run.
    lc_model = YOLO(MODEL_SIZE)
    lc_model.train(
        data      = str(frac_yaml),
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = 32,
        device    = "cpu",        # change to "cuda" for Windows GPU
        project   = str(lc_runs),
        name      = pct_tag,
        patience  = 10,
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
        verbose   = False,        # quieter output during the sweep
    )

    # ── evaluate on the fixed test set ───────────────────────
    # Find the best checkpoint for this run
    run_dirs  = sorted(glob.glob(str(lc_runs / f"{pct_tag}*")))
    best_ckpt = Path(run_dirs[-1]) / "weights" / "best.pt"
    print(f"  Evaluating: {best_ckpt}")

    lc_model = YOLO(str(best_ckpt))
    m        = lc_model.val(
        data   = str(frac_yaml),
        imgsz  = IMG_SIZE,
        device = "cpu",
    )

    # ── collect overall + per-class metrics ──────────────────
    row = {
        "fraction"        : fraction,
        "n_train_frames"  : n_train,
        "map50"           : round(m.box.map50, 4),
        "map50_95"        : round(m.box.map,   4),
        "precision"       : round(m.box.mp,    4),
        "recall"          : round(m.box.mr,    4),
        # per-class AP@0.5 — guarded for the imbalanced case where the moving
        # class may not appear at all in tiny subsets
        "ap50_stationary" : round(m.box.ap50[0], 4) if len(m.box.ap50) > 0 else None,
        "ap50_moving"     : round(m.box.ap50[1], 4) if len(m.box.ap50) > 1 else None,
        "moving_in_train" : sub_mov,
        "stat_in_train"   : sub_stat,
    }
    results_log.append(row)

    print(f"  mAP@0.5: {row['map50']:.4f}  "
          f"Prec: {row['precision']:.4f}  "
          f"Recall: {row['recall']:.4f}  "
          f"AP50_stat: {row['ap50_stationary']}  "
          f"AP50_mov: {row['ap50_moving']}")

# ── save results table ────────────────────────────────────────
csv_path = Path(OUTPUT_DIR) / "learning_curve_results.csv"
df = pd.DataFrame(results_log)
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")
print(df.to_string(index=False))

# ── plot learning curves ──────────────────────────────────────
# Three subplots so overall and per-class curves don't overlap.
# The x-axis uses actual frame counts (not percentages) so the spacing
# reflects real data volume rather than equal-looking percentage steps.
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Learning curve — YOLOv8n baseball detection", fontsize=13)

x = df["n_train_frames"]

# subplot 1: overall mAP
axes[0].plot(x, df["map50"],    marker="o", label="mAP@0.5",      color="#3266ad")
axes[0].plot(x, df["map50_95"], marker="s", label="mAP@0.5-0.95", color="#73726c",
             linestyle="--")
axes[0].set_title("Overall mAP")
axes[0].set_xlabel("Training frames")
axes[0].set_ylabel("Score")
axes[0].set_ylim(0, 1)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
# annotate percentage labels above each point so the plot is self-explanatory
for _, r in df.iterrows():
    axes[0].annotate(f"{int(r.fraction*100)}%",
                     (r.n_train_frames, r.map50),
                     textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center")

# subplot 2: precision & recall
axes[1].plot(x, df["precision"], marker="o", label="Precision", color="#1D9E75")
axes[1].plot(x, df["recall"],    marker="s", label="Recall",    color="#D85A30",
             linestyle="--")
axes[1].set_title("Precision & Recall")
axes[1].set_xlabel("Training frames")
axes[1].set_ylabel("Score")
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# subplot 3: per-class AP@0.5
# Moving-ball curve will be noisy at low fractions (~17:1 class imbalance means
# tiny subsets may contain only a handful of moving-ball examples).
# None values (class absent in a run) are dropped automatically by matplotlib.
axes[2].plot(x, df["ap50_stationary"], marker="o", label="Stationary", color="#534AB7")
axes[2].plot(x, df["ap50_moving"],     marker="s", label="Moving",     color="#BA7517",
             linestyle="--")
axes[2].set_title("Per-class AP@0.5\n(moving noisy at low fractions — 17:1 imbalance)")
axes[2].set_xlabel("Training frames")
axes[2].set_ylabel("AP@0.5")
axes[2].set_ylim(0, 1)
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = Path(OUTPUT_DIR) / "learning_curve.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Learning curve plot saved to: {plot_path}")

print("\n" + "="*60)
print("LEARNING CURVE EXPERIMENT COMPLETE")
print(f"  Results table : {csv_path}")
print(f"  Plot          : {plot_path}")
print(f"  Model weights : {lc_runs}/<fraction>/weights/best.pt")
print("="*60)
