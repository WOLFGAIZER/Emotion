"""
fer2013.py
Aligned completely with train_emotion_model.py configurations.
- Uses validation_split (no separate 'val' folder needed)
- Safe augmentation (flip only)
- Class weights
- Correct callbacks
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from models.shufflenet_model import build_shufflenetv2

# ================================
# CONFIGURATIONS
# ================================
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 64
EPOCHS = 40
NUM_CLASSES = 4  # angry, happy, sad, neutral
EMOTION_LABELS = ["angry", "happy", "neutral", "sad"]

DATASET_DIR = "datasets/fer2013/train"
WEIGHTS_PATH = "emotion_weights.weights.h5"

# ================================
# DATA GENERATORS (Aligned with train_emotion_model.py)
# ================================
# Using validation_split=0.2 instead of looking for a separate 'val' folder
# Using only horizontal_flip for augmentation

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True
)

print("[INFO] Loading Training Data...")
train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    classes=EMOTION_LABELS,
    shuffle=True
)

print("[INFO] Loading Validation Data...")
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

# ================================
# CLASS WEIGHTS
# ================================
# Calculate weights to balance the training
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2, 3]),
    y=train_gen.classes
)
class_weights = {i: weights[i] for i in range(NUM_CLASSES)}
print(f"[INFO] Computed Class Weights: {class_weights}")

# ================================
# BUILD MODEL
# ================================
model = build_shufflenetv2(
    input_shape=(96, 96, 1),
    num_classes=NUM_CLASSES
)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ================================
# CALLBACKS
# ================================
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(WEIGHTS_PATH, monitor="val_accuracy", save_best_only=True, save_weights_only=True, verbose=1)
]

# ================================
# TRAINING
# ================================
print(f"\n[INFO] Starting training for {EPOCHS} epochs...")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# ================================
# SAVE FINAL WEIGHTS
# ================================
# Redundant if ModelCheckpoint is used, but good for safety
model.save_weights(WEIGHTS_PATH)
print(f"\n✅ Training complete. Weights saved to {WEIGHTS_PATH}")