import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from models.shufflenet_model import build_shufflenetv2


# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
IMG_SIZE = (96, 96)
BATCH_SIZE = 4
EPOCHS = 15
LR = 1e-5
NUM_CLASSES = 4

JAFFE_TRAIN = "datasets/jaffe_split/train"
JAFFE_VAL = "datasets/jaffe_split/val"

BASE_WEIGHTS = "emotion_weights.weights.h5"
SAVE_WEIGHTS = "emotion_weights_finetuned.weights.h5"


# ------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------
model = build_shufflenetv2(input_shape=(96, 96, 1), num_classes=NUM_CLASSES)
model.load_weights(BASE_WEIGHTS)

# Freeze 90% of layers
freeze_until = int(len(model.layers) * 0.9)
for i, layer in enumerate(model.layers):
    layer.trainable = (i >= freeze_until)

model.compile(
    optimizer=Adam(LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------
# GENERATORS
# ------------------------------------------------------
train_aug = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05
)

val_aug = ImageDataGenerator(rescale=1/255.)

train_gen = train_aug.flow_from_directory(
    JAFFE_TRAIN,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode="categorical"
)

val_gen = val_aug.flow_from_directory(
    JAFFE_VAL,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode="categorical"
)


# ------------------------------------------------------
# TRAIN
# ------------------------------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=1
)

model.save_weights(SAVE_WEIGHTS)
print("[OK] Saved:", SAVE_WEIGHTS)


# ------------------------------------------------------
# EVALUATION
# ------------------------------------------------------
val_gen.reset()
pred = model.predict(val_gen)
y_pred = np.argmax(pred, axis=1)
y_true = val_gen.classes

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred,
                            target_names=val_gen.class_indices.keys()))
