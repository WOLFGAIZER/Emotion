import os
import shutil
import random

SRC = "datasets/jaffe"
DEST = "datasets/jaffe_split"

random.seed(42)

EMOTIONS = ["angry", "happy", "neutral", "sad"]

# create structure
for mode in ["train", "val"]:
    for emo in EMOTIONS:
        os.makedirs(os.path.join(DEST, mode, emo), exist_ok=True)

for emo in EMOTIONS:
    files = os.listdir(os.path.join(SRC, emo))
    files = [f for f in files if f.lower().endswith(".tiff")]
    random.shuffle(files)

    split_idx = int(0.8 * len(files))
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    for f in train_files:
        shutil.copy(os.path.join(SRC, emo, f),
                    os.path.join(DEST, "train", emo, f))

    for f in val_files:
        shutil.copy(os.path.join(SRC, emo, f),
                    os.path.join(DEST, "val", emo, f))

print("JAFFE successfully split into train/val")
