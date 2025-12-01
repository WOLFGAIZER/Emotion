import os
import shutil

# Path where your raw JAFFE images currently are
SRC_DIR = "datasets/jaffe_raw"

# Target folder where sorted images will be stored
DEST_DIR = "datasets/jaffe"

# Map JAFFE codes → your 4 emotion classes
LABEL_MAP = {
    "AN": "angry",
    "HA": "happy",
    "NE": "neutral",
    "SA": "sad"
}

# Create target folder structure
for label in LABEL_MAP.values():
    os.makedirs(os.path.join(DEST_DIR, label), exist_ok=True)

files = os.listdir(SRC_DIR)
count = 0

for file in files:
    if not file.lower().endswith(".tiff"):
        continue

    parts = file.split(".")
    if len(parts) < 2:
        continue

    # Extract the two-letter emotion code
    emotion_code = parts[1][:2]

    if emotion_code in LABEL_MAP:
        dest_folder = LABEL_MAP[emotion_code]

        src_path = os.path.join(SRC_DIR, file)
        dest_path = os.path.join(DEST_DIR, dest_folder, file)

        shutil.copy(src_path, dest_path)
        print(f"[OK] {file} -> {dest_folder}")
        count += 1
    else:
        print(f"[SKIP] {file} (unsupported code: {emotion_code})")

print(f"\nDONE! Sorted {count} JAFFE images into 4 folders.")
