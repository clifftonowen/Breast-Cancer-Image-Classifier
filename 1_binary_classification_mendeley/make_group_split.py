"""
Build a leakage-free train/val/test split for the Mendeley "Mammogram Mastery"
dataset, at the image (source lesion) level rather than the file level.

Why: the Augmented Dataset contains, for every original image `IMG (N).jpg`,
an unmodified copy plus 12 augmented variants (`IMG (N)_Crop.jpg`,
`IMG (N)_Dropout.jpg`, ...). The original notebook's 80/20 split of the
augmented set put variants of the same lesion in both train and validation,
and the "Original Dataset" held out as the test set is exactly the source
material those variants were derived from -- see audit_leakage.py.

This script instead:
  1. Groups every file in the Augmented Dataset by (class, lesion index), so
     an original and all 12 of its variants are one indivisible group.
  2. Splits GROUPS (not files) into train/val/test, stratified by class,
     so no lesion's variant ever crosses a split boundary.
  3. Writes the result as CSV index files (path, label, group) rather than
     copying image files -- the dataset is Git LFS tracked (*.jpg), and a
     copied ImageFolder tree under data/ would stage thousands of new LFS
     objects.

Output: 1_binary_classification_mendeley/splits/{train,val,test}.csv
"""

import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "Breast Cancer Dataset Mendeley"
AUGMENTED_DIR = DATA_ROOT / "Augmented Dataset"
SPLITS_DIR = Path(__file__).resolve().parent / "splits"
CLASSES = ["Cancer", "Non-Cancer"]

RANDOM_SEED = 42
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10  # matches the ~80/20-ish scale of the original notebook's
                       # train/val split, but now with a genuine held-out test set

# "IMG (12).jpg" -> group "12"; "IMG (12)_Crop.jpg" -> group "12" as well.
GROUP_RE = re.compile(r"^IMG \((\d+)\)(?:_.*)?\.(jpg|jpeg|png)$", re.IGNORECASE)


def group_key(fname: str) -> str | None:
    m = GROUP_RE.match(fname)
    return m.group(1) if m else None


def collect_groups(cls: str) -> dict[str, list[str]]:
    """Map lesion index -> list of relative file paths (original + variants)."""
    cls_dir = AUGMENTED_DIR / cls
    groups: dict[str, list[str]] = defaultdict(list)
    unparsed = []
    for fname in sorted(os.listdir(cls_dir)):
        key = group_key(fname)
        if key is None:
            unparsed.append(fname)
            continue
        groups[key].append(str((cls_dir / fname).relative_to(REPO_ROOT)))
    if unparsed:
        raise ValueError(
            f"{len(unparsed)} files in {cls_dir} did not match the expected "
            f"'IMG (N)[_Variant].ext' naming and were not grouped: "
            f"{unparsed[:5]}{'...' if len(unparsed) > 5 else ''}"
        )
    return groups


def split_groups(group_ids: list[str], rng: np.random.Generator) -> tuple[list[str], list[str], list[str]]:
    ids = list(group_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, round(n * TEST_FRACTION))
    n_val = max(1, round(n * VAL_FRACTION))
    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test + n_val]
    train_ids = ids[n_test + n_val:]
    return train_ids, val_ids, test_ids


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    SPLITS_DIR.mkdir(exist_ok=True)

    rows = {"train": [], "val": [], "test": []}
    group_assignment = {"train": defaultdict(set), "val": defaultdict(set), "test": defaultdict(set)}

    for cls in CLASSES:
        groups = collect_groups(cls)
        train_ids, val_ids, test_ids = split_groups(list(groups.keys()), rng)

        for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            for gid in ids:
                group_assignment[split_name][cls].add(gid)
                for path in groups[gid]:
                    rows[split_name].append((path, cls, f"{cls}:{gid}"))

    # --- Hard invariant: zero group overlap across splits ---
    for cls in CLASSES:
        train_g = group_assignment["train"][cls]
        val_g = group_assignment["val"][cls]
        test_g = group_assignment["test"][cls]
        overlap_tv = train_g & val_g
        overlap_tt = train_g & test_g
        overlap_vt = val_g & test_g
        assert not overlap_tv, f"{cls}: groups leaked between train/val: {overlap_tv}"
        assert not overlap_tt, f"{cls}: groups leaked between train/test: {overlap_tt}"
        assert not overlap_vt, f"{cls}: groups leaked between val/test: {overlap_vt}"

    for split_name, split_rows in rows.items():
        out_path = SPLITS_DIR / f"{split_name}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label", "group"])
            writer.writerows(split_rows)

    print("Group-level split written to", SPLITS_DIR)
    for split_name in ["train", "val", "test"]:
        by_class = defaultdict(int)
        groups_by_class = defaultdict(int)
        for _, cls, _ in rows[split_name]:
            by_class[cls] += 1
        for cls in CLASSES:
            groups_by_class[cls] = len(group_assignment[split_name][cls])
        total = sum(by_class.values())
        print(f"  {split_name}: {total} files "
              f"({', '.join(f'{cls}={by_class[cls]} files / {groups_by_class[cls]} lesions' for cls in CLASSES)})")

    test_cancer = len(group_assignment["test"]["Cancer"])
    test_noncancer = len(group_assignment["test"]["Non-Cancer"])
    test_total = test_cancer + test_noncancer
    if test_total:
        print(f"\n  Test-set majority baseline: "
              f"{100 * max(test_cancer, test_noncancer) / test_total:.1f}% "
              f"({test_cancer} Cancer / {test_noncancer} Non-Cancer lesions)")

    print("\nAssertions passed: no lesion's images cross a train/val/test boundary.")


if __name__ == "__main__":
    main()
