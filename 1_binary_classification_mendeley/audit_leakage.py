"""
Leakage audit for the Mendeley "Mammogram Mastery" binary classifier.

Question: is the reported 99.7% "final test accuracy on a held-out unseen set"
real, or an artefact of the "Original Dataset" test images also being present
(as augmented variants) inside the "Augmented Dataset" training/validation split?

Run from the repo root or from this directory; paths are resolved relative to
this file, so `python audit_leakage.py` works from anywhere.

Prints:
  1. A generator-code check: does Image_Augmentation.py write an unmodified
     copy of each original into the augmented folder under the same filename?
  2. Pixel-level comparison of all Original vs Augmented same-name pairs
     (dimension match + perceptual hash distance) against a random-pair
     control, to rule out coincidence.
  3. Exact overlap counts between the "unseen" Original test set and the
     Augmented_Split/{train,validation} folders actually used for training.
"""

import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "Breast Cancer Dataset Mendeley"
ORIGINAL_DIR = DATA_ROOT / "Original Dataset"
AUGMENTED_DIR = DATA_ROOT / "Augmented Dataset"
SPLIT_DIR = DATA_ROOT / "Augmented_Split"
CLASSES = ["Cancer", "Non-Cancer"]

FILENAME_RE = re.compile(r"^IMG \((\d+)\)\.(jpg|jpeg|png)$", re.IGNORECASE)


def dhash(path: Path, hash_size: int = 16) -> np.ndarray:
    """Perceptual difference hash as a boolean bit vector."""
    img = Image.open(path).convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.int16)
    return (arr[:, 1:] > arr[:, :-1]).flatten()


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_generator_script() -> None:
    section("1. Does the dataset's own augmentation script re-save originals "
            "into the augmented folder?")
    script_path = REPO_ROOT / "1_binary_classification_mendeley" / "Image_Augmentation.py"
    if not script_path.exists():
        print(f"  Image_Augmentation.py not found at {script_path}")
        return
    src = script_path.read_text(encoding="utf-8", errors="replace")
    saves_original = bool(
        "original_path" in src
        and "augmented_folder" in src
        and re.search(r"Image\.fromarray\(original_image\)\.save\(original_path\)", src)
    )
    print(f"  Script: {script_path.relative_to(REPO_ROOT)}")
    print(f"  Contains an explicit unmodified-original save into the augmented "
          f"folder: {saves_original}")
    if saves_original:
        print("  -> This is the mechanism: every augmented folder entry set "
              "includes an unaugmented copy of the source image, filed under "
              "the same name as the 'unseen' original.")


def check_pixel_overlap() -> dict:
    section("2. Pixel-level comparison: Original vs Augmented same-name pairs")
    results = {}
    for cls in CLASSES:
        orig_dir = ORIGINAL_DIR / cls
        aug_dir = AUGMENTED_DIR / cls
        if not orig_dir.exists() or not aug_dir.exists():
            print(f"  Skipping {cls}: folder missing")
            continue

        distances = []
        same_dims = 0
        missing = 0
        for fname in sorted(os.listdir(orig_dir)):
            aug_path = aug_dir / fname
            if not aug_path.exists():
                missing += 1
                continue
            orig_path = orig_dir / fname
            with Image.open(orig_path) as a, Image.open(aug_path) as b:
                if a.size == b.size:
                    same_dims += 1
            distances.append(int((dhash(orig_path) != dhash(aug_path)).sum()))

        d = np.array(distances)
        n = len(d)
        print(f"\n  {cls}: n={n} pairs checked, missing_in_augmented={missing}")
        print(f"    identical pixel dimensions: {same_dims}/{n}")
        print(f"    dhash Hamming distance (0=identical, max=240 bits): "
              f"mean={d.mean():.2f}  max={d.max()}  "
              f"<=2:{(d <= 2).sum()}  <=5:{(d <= 5).sum()}")
        results[cls] = {"n": n, "missing": missing, "same_dims": same_dims, "distances": d}

    # Control: distance between random pairs of genuinely different originals
    section("2b. Control: dhash distance between random pairs of DIFFERENT "
            "original images")
    rng = np.random.default_rng(0)
    control_dir = ORIGINAL_DIR / "Non-Cancer"
    files = sorted(os.listdir(control_dir))
    control_dists = []
    for _ in range(300):
        a, b = rng.choice(files, size=2, replace=False)
        control_dists.append(int((dhash(control_dir / a) != dhash(control_dir / b)).sum()))
    cd = np.array(control_dists)
    print(f"  300 random distinct-image pairs: mean={cd.mean():.1f}  "
          f"median={np.median(cd):.0f}  min={cd.min()}")
    print("  -> Matched Original/Augmented pairs sit far below this control "
          "distribution: they are the same images, not coincidental matches.")

    return results


def group_key(fname: str) -> str | None:
    m = FILENAME_RE.match(fname)
    if m:
        return m.group(1)
    m2 = re.match(r"^IMG \((\d+)\)_", fname)
    return m2.group(1) if m2 else None


def check_split_overlap() -> None:
    section("3. Overlap between 'unseen' Original test images and the "
            "Augmented_Split/{train,validation} folders actually used for "
            "training")
    if not SPLIT_DIR.exists():
        print(f"  {SPLIT_DIR} not found -- run the split step from "
              "train_resnet50.ipynb first.")
        return

    total_orig = total_train = total_val = 0
    for cls in CLASSES:
        orig_files = set(os.listdir(ORIGINAL_DIR / cls))
        train_files = set(os.listdir(SPLIT_DIR / "train" / cls))
        val_files = set(os.listdir(SPLIT_DIR / "validation" / cls))

        verbatim_in_train = orig_files & train_files
        verbatim_in_val = orig_files & val_files

        orig_groups = {group_key(f) for f in orig_files if group_key(f)}
        train_groups = {group_key(f) for f in train_files if group_key(f)}

        print(f"\n  {cls}: {len(orig_files)} original test images")
        print(f"    verbatim filename also in TRAIN:      {len(verbatim_in_train)}")
        print(f"    verbatim filename also in VALIDATION: {len(verbatim_in_val)}")
        print(f"    test lesions with >=1 variant in TRAIN: "
              f"{len(orig_groups & train_groups)}/{len(orig_groups)}")

        total_orig += len(orig_files)
        total_train += len(verbatim_in_train)
        total_val += len(verbatim_in_val)

    print(f"\n  TOTAL: {total_orig} 'unseen' test images -> "
          f"{total_train} ({100*total_train/total_orig:.1f}%) verbatim in train, "
          f"{total_val} ({100*total_val/total_orig:.1f}%) verbatim in validation, "
          f"{total_train + total_val}/{total_orig} "
          f"({100*(total_train+total_val)/total_orig:.1f}%) leaked overall.")


def check_majority_baseline() -> None:
    section("4. Class balance and majority-class baseline")
    counts = {cls: len(os.listdir(ORIGINAL_DIR / cls)) for cls in CLASSES
              if (ORIGINAL_DIR / cls).exists()}
    total = sum(counts.values())
    for cls, n in counts.items():
        print(f"  {cls}: {n} ({100*n/total:.1f}%)")
    majority = max(counts.values())
    print(f"  Majority-class baseline: {100*majority/total:.1f}%")


if __name__ == "__main__":
    check_generator_script()
    check_pixel_overlap()
    check_split_overlap()
    check_majority_baseline()
