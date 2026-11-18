"""按 9:1 划分数据集：data/<类别> -> data1/train/<类别> + data1/test/<类别>

用法（在项目根目录执行）:
    python split_dataset.py                        # 默认 data -> data1，0.9/0.1，种子 42
    python split_dataset.py --ratio 0.9 --seed 42
    python split_dataset.py --move                 # 移动文件而不是复制（节省空间）
    python split_dataset.py --dst data1 --overwrite  # 覆盖已有的 data1/train、data1/test
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
SKIP_DIRS = {"train", "test"}


def list_classes(src: Path):
    return sorted(
        d for d in src.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith(".")
    )


def split_class(class_dir: Path, train_dir: Path, test_dir: Path,
                ratio: float, rng: random.Random, move: bool):
    files = sorted(f for f in class_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    if not files:
        return 0, 0
    rng.shuffle(files)

    n_train = round(len(files) * ratio)
    train_files = files[:n_train]
    test_files = files[n_train:]

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    action = shutil.move if move else shutil.copy2
    for f in train_files:
        action(str(f), str(train_dir / f.name))
    for f in test_files:
        action(str(f), str(test_dir / f.name))

    return len(train_files), len(test_files)


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/test by ratio")
    parser.add_argument("--src", default="data", help="source folder containing class folders")
    parser.add_argument("--dst", default="data1", help="output folder, gets train/ and test/")
    parser.add_argument("--ratio", type=float, default=0.9, help="train ratio, default 0.9")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--move", action="store_true", help="move files instead of copying")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete existing <dst>/train and <dst>/test first")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")

    dst = Path(args.dst).resolve()
    if dst == src:
        raise SystemExit("--dst must be different from --src")

    train_root = dst / "train"
    test_root = dst / "test"

    for out in (train_root, test_root):
        if out.exists():
            if not args.overwrite:
                raise SystemExit(f"Output folder already exists: {out}\n"
                                 f"Use --overwrite to delete it, or choose another --dst")
            shutil.rmtree(out)

    classes = list_classes(src)
    if not classes:
        raise SystemExit(f"No class folders found in {src}")

    rng = random.Random(args.seed)
    total_train = total_test = 0
    skipped = []

    for i, class_dir in enumerate(classes, 1):
        n_train, n_test = split_class(class_dir, train_root / class_dir.name,
                                      test_root / class_dir.name, args.ratio, rng, args.move)
        total_train += n_train
        total_test += n_test
        status = "SKIP (empty)" if n_train + n_test == 0 else f"train={n_train:5d}  test={n_test:4d}"
        if n_train + n_test == 0:
            skipped.append(class_dir.name)
        print(f"[{i:2d}/{len(classes)}] {class_dir.name:<10s} {status}")

    print("-" * 50)
    print(f"Done. train={total_train}, test={total_test}, "
          f"split ratio={total_train / max(1, total_train + total_test):.3f}")
    print(f"Output: {train_root}  |  {test_root}")
    if skipped:
        print(f"Warning: empty classes skipped (no images): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
