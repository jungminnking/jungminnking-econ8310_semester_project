import os
import xml.etree.ElementTree as ET
import shutil
import random
import yaml
import glob
from pathlib import Path
 
# Load user-specific paths and settings from config.py
try:
    from config import (
        XML_DIR, FRAMES_DIR, OUTPUT_DIR,
        MODEL_SIZE, EPOCHS, IMG_SIZE, VAL_SPLIT, RANDOM_SEED
    )
except ImportError:
    print("ERROR: config.py not found.")
    print("Copy config.py, fill in your local paths, and try again.")
    exit(1)
 
 
# VIDEO NAME MAPPING

NAME_MAP = {
    "IMG_8226_jared": "jared1",
    "IMG_8241_jared": "jared2",
    "IMG_8242_jared": "jared3",
    "IMG_8243_jared": "jared4",
    "IMG_8252_zach":  "zach1",
    "IMG_8255_zach":  "zach2",
    "IMG_8256_zach":  "zach3",
}
 
 
def find_image(img_name, frames_dir):
    
    # Try to find the image file locally using multiple strategies.
    frames_dir = Path(frames_dir)
 
    # Exact relative path as stored in XML
    candidate = frames_dir / img_name.replace("/", os.sep)
    if candidate.exists():
        return str(candidate)
 
    # Search all subdirectories by filename
    filename = Path(img_name).name
    for match in frames_dir.rglob(filename):
        return str(match)
 
    # Use NAME_MAP to convert CVAT names to local names
    # e.g. IMG_8226_jared/frame_0001.jpg -> jared1_0001.jpg
    parts = img_name.replace("\\", "/").split("/")
    if len(parts) == 2:
        folder, fname = parts
        prefix = NAME_MAP.get(folder)
        if prefix:
            number = Path(fname).stem.replace("frame_", "")
            mapped = f"{prefix}_{number}.jpg"
            candidate = frames_dir / mapped
            if candidate.exists():
                return str(candidate)
 
    # Stem-only match (ignore extension)
    stem = Path(img_name).stem
    for match in frames_dir.rglob(f"{stem}.*"):
        return str(match)
 
    return None  # not found locally
 
 
def parse_all_xmls(xml_dir, frames_dir):
    """
    Parse every XML file in xml_dir.
    Returns list of (img_name, img_full_path, boxes)
    only for frames we can find locally.
    """
    xml_files = list(Path(xml_dir).glob("*.xml"))
    print(f"Found {len(xml_files)} XML files")
 
    all_annotations = []
    total_boxes    = 0
    skipped_no_img = 0
    skipped_no_box = 0
 
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"  WARNING: Could not parse {xml_path.name}: {e}")
            continue
 
        for image_elem in root.findall("image"):
            img_name = image_elem.get("name", "")
            img_w    = int(image_elem.get("width",  2160))
            img_h    = int(image_elem.get("height", 3840))
 
            # Collect bounding boxes
            boxes = []
            for box in image_elem.findall("box"):
                try:
                    xtl = float(box.get("xtl"))
                    ytl = float(box.get("ytl"))
                    xbr = float(box.get("xbr"))
                    ybr = float(box.get("ybr"))
                    if xbr > xtl and ybr > ytl:
                        boxes.append((xtl, ytl, xbr, ybr, img_w, img_h))
                except (TypeError, ValueError):
                    continue
 
            if not boxes:
                skipped_no_box += 1
                continue
 
            # Try to find the image locally
            img_full = find_image(img_name, frames_dir)
            if img_full is None:
                skipped_no_img += 1
                continue
 
            all_annotations.append((img_name, img_full, boxes))
            total_boxes += len(boxes)
 
    print(f"Matched {len(all_annotations)} annotated frames with local images")
    print(f"Total bounding boxes: {total_boxes}")
    print(f"Skipped (no local image): {skipped_no_img}")
    print(f"Skipped (no bounding boxes): {skipped_no_box}")
    return all_annotations
 
 
def cvat_box_to_yolo(xtl, ytl, xbr, ybr, img_w, img_h):

    cx = max(0.0, min(1.0, ((xtl + xbr) / 2) / img_w))
    cy = max(0.0, min(1.0, ((ytl + ybr) / 2) / img_h))
    w  = max(0.0, min(1.0, (xbr - xtl) / img_w))
    h  = max(0.0, min(1.0, (ybr - ytl) / img_h))
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
 
 
def build_yolo_dataset(annotations):
  
    # Remove duplicates (same image annotated by multiple teams)
    seen   = set()
    unique = []
    for ann in annotations:
        if ann[1] not in seen:
            seen.add(ann[1])
            unique.append(ann)
 
    if len(unique) < len(annotations):
        print(f"Removed {len(annotations) - len(unique)} duplicate frames")
 
    random.seed(RANDOM_SEED)
    random.shuffle(unique)
 
    n_val   = max(1, int(len(unique) * VAL_SPLIT))
    val_set = unique[:n_val]
    trn_set = unique[n_val:]
 
    for split, data in [("train", trn_set), ("val", val_set)]:
        img_dir = Path(OUTPUT_DIR) / "images" / split
        lbl_dir = Path(OUTPUT_DIR) / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
 
        for img_name, img_full, boxes in data:
            safe_name = img_name.replace("/", "_").replace("\\", "_")
            if not safe_name.lower().endswith(".jpg"):
                safe_name += ".jpg"
            stem = Path(safe_name).stem
 
            shutil.copy2(img_full, img_dir / safe_name)
 
            with open(lbl_dir / f"{stem}.txt", "w") as f:
                for (xtl, ytl, xbr, ybr, iw, ih) in boxes:
                    f.write(cvat_box_to_yolo(xtl, ytl, xbr, ybr, iw, ih) + "\n")
 
    print(f"Dataset ready: {len(trn_set)} train, {len(val_set)} val frames")
    return len(trn_set), len(val_set)
 
 
def write_yaml():
    
    yaml_path = Path(OUTPUT_DIR) / "dataset.yaml"
    config = {
        "path"  : str(OUTPUT_DIR).replace("\\", "/"),
        "train" : "images/train",
        "val"   : "images/val",
        "nc"    : 1,
        "names" : ["baseball"]
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"YAML written: {yaml_path}")
    return str(yaml_path)
 
 
def train_model(yaml_path):

    from ultralytics import YOLO
 
    print(f"\nLoading pre-trained model: {MODEL_SIZE}")
    model = YOLO(MODEL_SIZE)
    print(f"Training for {EPOCHS} epochs on MPS (Apple Silicon)...\n")
 
    model.train(
        data      = yaml_path,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = 2,
        device    = "mps",       # Apple Silicon GPU
        project   = str(Path(OUTPUT_DIR) / "runs"),
        name      = "baseball_detect",
        patience  = 50,          # early stopping patience
 
       
        hsv_h     = 0.015,       # hue variation
        hsv_s     = 0.7,         # saturation variation
        hsv_v     = 0.4,         # brightness variation
        fliplr    = 0.5,         # random horizontal flip
        mosaic    = 1.0,         # mosaic (combines 4 frames)
        degrees   = 10.0,        # random rotation ±10°
        translate = 0.1,         # random translation
        scale     = 0.5,         # random zoom in/out
        
 
        save      = True,
        plots     = True,
        verbose   = True,
    )
 
 
def evaluate_model():
    #Evaluate the best saved model and print metrics
    from ultralytics import YOLO
    import glob

    runs = sorted(glob.glob(str(Path(OUTPUT_DIR) / "runs" / "baseball_detect*")))
    if not runs:
        print("No trained model found.")
        return

    best = Path(runs[-1]) / "weights" / "best.pt"
    if not best.exists():
        print("No trained model found.")
        return

    print(f"\nEvaluating: {best}")
    model   = YOLO(str(best))
    metrics = model.val(
        data   = str(Path(OUTPUT_DIR) / "dataset.yaml"),
        imgsz  = IMG_SIZE,
        device = "mps",
    )

    print("\n========== RESULTS ==========")
    print(f"mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"mAP@0.5-0.95:  {metrics.box.map:.4f}")
    print(f"Precision:     {metrics.box.mp:.4f}")
    print(f"Recall:        {metrics.box.mr:.4f}")
    print("=============================")
    print("\nmAP@0.5 = detection accuracy at 50% IoU threshold (higher = better)")
 
def save_sample_predictions():

    from ultralytics import YOLO
    import cv2
 
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
 
 

# MAIN

if __name__ == "__main__":
    print("=" * 50)
    print("Baseball Detection - Training Pipeline (Improved)")
    print("=" * 50)
    print(f"Model:         {MODEL_SIZE}")
    print(f"Epochs:        {EPOCHS}")
    print(f"Image size:    {IMG_SIZE}")
    print(f"XML folder:    {XML_DIR}")
    print(f"Frames folder: {FRAMES_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
 
    print("\n[1/5] Parsing all XML annotations...")
    annotations = parse_all_xmls(XML_DIR, FRAMES_DIR)
 
    if len(annotations) == 0:
        print("\nERROR: No matching frames found.")
        print("Check that FRAMES_DIR contains your JPEG frame files.")
        exit(1)
 
    print("\n[2/5] Building YOLO dataset...")
    build_yolo_dataset(annotations)
 
    print("\n[3/5] Writing dataset config...")
    yaml_path = write_yaml()
 
    print("\n[4/5] Training model...")
    train_model(yaml_path)
 
    print("\n[5/5] Evaluating model...")
    evaluate_model()
 
    print("\n[Bonus] Saving sample predictions...")
    save_sample_predictions()
 
    print(f"\nAll done! Results saved to:\n  {OUTPUT_DIR}")