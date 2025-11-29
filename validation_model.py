"""
validation_model.py

Evaluates trained 4-class model (angry, happy, sad, neutral)
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.shufflenet_model import build_shufflenetv2, load_emotion_weights

# ========== CONFIG ==========
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64
NUM_CLASSES = 4
MODEL_WEIGHTS = "emotion_weights.weights.h5"

EMOTION_LABELS = ['angry', 'happy', 'sad', 'neutral']
FER_DIR = "datasets/fer2013"

# ========== LOAD MODEL ==========
model = build_shufflenetv2(input_shape=(48, 48, 1), num_classes=NUM_CLASSES)

if not load_emotion_weights(model, MODEL_WEIGHTS):
    raise FileNotFoundError(f"Could not load weights from {MODEL_WEIGHTS}")

# MUST COMPILE BEFORE EVALUATE
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

print("[INFO] Model compiled successfully.")

# ========== LOAD FER2013 VALIDATION ==========
print("[INFO] Preparing FER2013 validation set...")
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

fer_val_gen = datagen.flow_from_directory(
    os.path.join(FER_DIR, 'train'),
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset='validation',
    shuffle=False,
    classes=EMOTION_LABELS
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)


# ========== EVALUATE ==========
print("\n[VALIDATION] Evaluating on FER2013...")
fer_loss, fer_acc = model.evaluate(fer_val_gen)
print(f"FER2013 Accuracy: {fer_acc * 100:.2f}%")

# ========== CONFUSION MATRIX ==========
print("\n[INFO] Generating confusion matrix...")
fer_preds = model.predict(fer_val_gen, verbose=1)
fer_y_true = fer_val_gen.classes
fer_y_pred = np.argmax(fer_preds, axis=1)

print("\n===== FER2013 Classification Report =====")
print(classification_report(fer_y_true, fer_y_pred, target_names=EMOTION_LABELS))

cm = confusion_matrix(fer_y_true, fer_y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTION_LABELS,
            yticklabels=EMOTION_LABELS)
plt.title("FER2013 Confusion Matrix (4-Class)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

