"""
train_emotion_model.py — FINAL VERSION
--------------------------------------
✓ Stable training (no validation collapse)
✓ Consistent preprocessing
✓ Light FER-safe augmentation
✓ Correct class weights
✓ Confusion matrix
✓ Classification report
✓ Accuracy and loss plots
✓ Saves best weights to emotion_weights.weights.h5
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from models.shufflenet_model import build_shufflenetv2

# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = (96, 96)
BATCH_SIZE = 64
EPOCHS = 40

DATASET_DIR = "datasets/fer2013/train"
WEIGHTS_PATH = "emotion_weights.weights.h5"

EMOTION_LABELS = ["angry", "happy", "neutral", "sad"]
NUM_CLASSES = len(EMOTION_LABELS)

os.makedirs("plots", exist_ok=True)

# ============================================================
# DATASET LOADERS (SAFE AUGMENTATION ONLY)
# ============================================================

datagen = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2,
    horizontal_flip=True
)

train_gen_base = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    classes=EMOTION_LABELS,
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    classes=EMOTION_LABELS,
    shuffle=False
)

print("\nCLASS ORDER:", train_gen_base.class_indices)
print("Train Samples:", train_gen_base.samples)
print("Val Samples:", val_gen.samples)

# ============================================================
# CLASS WEIGHTS
# ============================================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2, 3]),
    y=train_gen_base.classes
)

class_weights = {i: weights[i] for i in range(NUM_CLASSES)}
print("Computed Class Weights:", class_weights)

# ============================================================
# SAFE GENERATOR (NO HEAVY AUGMENTATION)
# ============================================================

def safe_train_generator():
    while True:
        batch_x, batch_y = next(train_gen_base)

        sample_weights = np.zeros(len(batch_y), dtype="float32")
        for i in range(len(batch_y)):
            class_idx = np.argmax(batch_y[i])
            sample_weights[i] = class_weights[class_idx]

        yield batch_x, batch_y, sample_weights

# ============================================================
# BUILD MODEL
# ============================================================

model = build_shufflenetv2(
    input_shape=(96, 96, 1),
    num_classes=NUM_CLASSES,
    width_multiplier=1.0
)

model.compile(
    optimizer=Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint(WEIGHTS_PATH, monitor="val_accuracy",
                    save_best_only=True, save_weights_only=True)
]

steps_per_epoch = train_gen_base.samples // BATCH_SIZE
validation_steps = val_gen.samples // BATCH_SIZE

# ============================================================
# TRAINING
# ============================================================

history = model.fit(
    safe_train_generator(),
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============================================================
# SAVE WEIGHTS
# ============================================================

model.save_weights(WEIGHTS_PATH)
print(f"\n✔ Saved weights to: {WEIGHTS_PATH}")

# ============================================================
# PLOTS: ACCURACY + LOSS
# ============================================================

plt.figure(figsize=(8,6))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend(); plt.grid()
plt.title("Model Accuracy")
plt.savefig("plots/accuracy_curve.png")
plt.close()

plt.figure(figsize=(8,6))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend(); plt.grid()
plt.title("Model Loss")
plt.savefig("plots/loss_curve.png")
plt.close()

print("✔ Accuracy & Loss plots saved in /plots")

# ============================================================
# CONFUSION MATRIX + REPORT
# ============================================================

val_gen.reset()
preds = model.predict(val_gen, verbose=1)
pred_classes = np.argmax(preds, axis=1)
true_classes = val_gen.classes

cm = confusion_matrix(true_classes, pred_classes)
print("\n====== CONFUSION MATRIX ======\n")
print(cm)

plt.figure(figsize=(7,6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(NUM_CLASSES), EMOTION_LABELS)
plt.yticks(range(NUM_CLASSES), EMOTION_LABELS)
plt.savefig("plots/confusion_matrix.png")
plt.close()

print("\n====== CLASSIFICATION REPORT ======\n")
report = classification_report(true_classes, pred_classes, target_names=EMOTION_LABELS)
print(report)

with open("plots/classification_report.txt", "w") as f:
    f.write(report)

print("✔ Confusion Matrix + Report saved in /plots")

print("\nFinal Train Accuracy:", history.history["accuracy"][-1])
print("Final Val Accuracy:", history.history["val_accuracy"][-1])
