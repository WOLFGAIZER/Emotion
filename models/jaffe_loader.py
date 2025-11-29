import os
import cv2
import numpy as np

def jaffe_loader(dataset_path, target_size=(48, 48)):
    images, labels = [], []
    label_map = {
        "AN": "angry",
        "HA": "happy",
        "SA": "sad",
        "SU": "surprise",
        "NE": "neutral"
    }

    for file in os.listdir(dataset_path):
        if file.endswith(".tiff") or file.endswith(".jpg"):
            img = cv2.imread(os.path.join(dataset_path, file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, target_size)
            images.append(img)
            # extract label code from filename like KA.AN1.39.tiff
            code = file.split(".")[1][:2]
            labels.append(label_map.get(code, "neutral"))

    images = np.array(images, dtype="float32") / 255.0
    images = np.expand_dims(images, axis=-1)
    return images, labels
