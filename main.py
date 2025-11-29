"""
main.py — Final Driver Monitoring System

Features included:
- Mediapipe FaceMesh
- Dynamic EAR Calibration (for different eye shapes)
- Blink Detection + Blink Rate + Blink Fatigue
- MAR-based Yawn Detection
- Drowsiness Score (0–100)
- Anger Alert
- Temporal Emotion Smoothing
- Face Alignment (eye-level)
- Alarm System
"""

import cv2
import numpy as np
import mediapipe as mp
import pygame
import math
import os
from collections import deque

# -------------------------
# IMPORT MODEL + UTILITIES
# -------------------------
from models.shufflenet_model import build_shufflenetv2, load_emotion_weights
from models.ear_calculator import (
    eye_aspect_ratio,
    mouth_aspect_ratio,
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    MOUTH_OUTER_IDX
)

# -------------------------
# CONSTANTS
# -------------------------
YAWN_THRESHOLD = 0.62
YAWN_FRAMES_REQUIRED = 12
BLINK_FRAMES_REQUIRED = 3

DROWSINESS_INCREASE = 1.5
DROWSINESS_DECAY = 0.08

ANGER_CONFIDENCE = 0.55

EMOTION_LABELS = ["angry", "happy", "neutral", "sad"]
NUM_CLASSES = len(EMOTION_LABELS)

WEIGHTS_PATH = "emotion_weights.weights.h5"
ALARM_SOUND = "alarm.mp3"

# -------------------------
# DYNAMIC EAR CALIBRATION
# -------------------------
CALIBRATION_FRAMES = 60
calibration_ears = []
EAR_THRESHOLD_DYNAMIC = None

# -------------------------
# SESSION STATE
# -------------------------
total_blinks = 0
blink_state = "open"
yawn_counter = 0
total_yawns = 0
drowsiness_score = 0
max_drowsiness_seen = 0
start_time = cv2.getTickCount()
total_frames = 0

# -------------------------
# FACE ALIGNMENT
# -------------------------
def align_face(frame, landmarks):
    LEFT_EYE = [33, 133]
    RIGHT_EYE = [362, 263]

    left_center = np.mean([landmarks[i] for i in LEFT_EYE], axis=0).astype(int)
    right_center = np.mean([landmarks[i] for i in RIGHT_EYE], axis=0).astype(int)
    
    left_center = (int(left_center[0]), int(left_center[1]))
    right_center = (int(right_center[0]), int(right_center[1]))

    dx = right_center[0] - left_center[0]
    dy = right_center[1] - left_center[1]
    angle = math.degrees(math.atan2(dy, dx))

    center_x = int((left_center[0] + right_center[0]) / 2)
    center_y = int((left_center[1] + right_center[1])/ 2)
    center = (center_x, center_y)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    return rotated

# -------------------------
# EMOTION SMOOTHING
# -------------------------
SMOOTHING_WINDOW = 4
emotion_history = deque(maxlen=SMOOTHING_WINDOW)

def smooth_emotion(preds):
    emotion_history.append(preds)
    avg = np.mean(np.array(emotion_history), axis=0)
    idx = np.argmax(avg)
    return EMOTION_LABELS[idx], float(avg[idx])

# -------------------------
# FACE PREPROCESSING
# -------------------------
clahe = cv2.createCLAHE(clipLimit=2.0)

def preprocess_face(gray):
    gray = clahe.apply(gray)
    face = cv2.resize(gray, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, -1)
    return np.expand_dims(face, 0)

# -------------------------
# LOAD EMOTION MODEL
# -------------------------
print("[INFO] Loading emotion model...")

emotion_model = build_shufflenetv2(
    input_shape=(48, 48, 1),
    num_classes=NUM_CLASSES,
    width_multiplier=1.2,
)

load_emotion_weights(emotion_model, WEIGHTS_PATH)

# -------------------------
# MEDIAPIPE INIT
# -------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# -------------------------
# ALARM SYSTEM
# -------------------------
pygame.mixer.init()
ALARM_ON = False

if os.path.exists(ALARM_SOUND):
    pygame.mixer.music.load(ALARM_SOUND)

def trigger_alarm():
    global ALARM_ON
    if not ALARM_ON:
        pygame.mixer.music.play(-1)
        ALARM_ON = True

def stop_alarm():
    global ALARM_ON
    if ALARM_ON:
        pygame.mixer.music.stop()
        ALARM_ON = False

# -------------------------
# VIDEO STREAM
# -------------------------
cap = cv2.VideoCapture(0)
print("[INFO] Starting video stream...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            cv2.imshow("DMS", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        landmarks = [(int(l.x * w), int(l.y * h)) for l in result.multi_face_landmarks[0].landmark]

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
        EAR = (left_ear + right_ear) / 2

        # -------------------------
        # DYNAMIC EAR CALIBRATION
        # -------------------------
        if EAR_THRESHOLD_DYNAMIC is None:
            calibration_ears.append(EAR)
            cv2.putText(frame, f"Calibrating EAR... {len(calibration_ears)}/{CALIBRATION_FRAMES}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            if len(calibration_ears) >= CALIBRATION_FRAMES:
                EAR_THRESHOLD_DYNAMIC = float(np.mean(calibration_ears) * 0.75)
                print("[INFO] EAR Threshold set to:", EAR_THRESHOLD_DYNAMIC)

            cv2.imshow("DMS", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # -------------------------
        # BLINK DETECTION
        # -------------------------
        if EAR < EAR_THRESHOLD_DYNAMIC:
            if blink_state == "open":
                blink_state = "closed"
        else:
            if blink_state == "closed":
                total_blinks += 1
                blink_state = "open"

        # -------------------------
        # YAWN DETECTION (MAR)
        # -------------------------
        mar = mouth_aspect_ratio(landmarks, MOUTH_OUTER_IDX)

        if mar > YAWN_THRESHOLD:
            yawn_counter += 1
        else:
            if yawn_counter >= YAWN_FRAMES_REQUIRED:
                drowsiness_score += 8
                total_yawns += 1
            yawn_counter = 0

        # -------------------------
        # UPDATE DROWSINESS SCORE
        # -------------------------
        if EAR < EAR_THRESHOLD_DYNAMIC:
            drowsiness_score += DROWSINESS_INCREASE

        drowsiness_score = max(0, drowsiness_score - DROWSINESS_DECAY)
        drowsiness_score = min(100, drowsiness_score)

        max_drowsiness_seen = max(max_drowsiness_seen, drowsiness_score)

        # -------------------------
        # FACE ALIGNMENT
        # -------------------------
        aligned_frame = align_face(frame, landmarks)

        # -------------------------
        # EMOTION DETECTION
        # -------------------------
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]

        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(w - 1, max(xs)), min(h - 1, max(ys))

        emotion_label = "unknown"
        emotion_conf = 0.0

        if (x2 - x1) >= 40 and (y2 - y1) >= 40:
           
           pad_x = 5
           pad_y = 25
           rx1 = max(0, x1 - pad_x)
           ry1 = max(0, y1 - pad_y)
           rx2 = min(w - 1, x2 + pad_x)
           ry2 = min(h - 1, y2 + pad_y)

           roi = aligned_frame[ry1:ry2, rx1:rx2]

           if roi is None or roi.size == 0:
               pass
           elif roi.shape [0] < 20 or roi.shape[1] < 20:
               pass
           else: 
               
               gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

               gray = cv2.equalizeHist(gray) #or simply gray = gray

               face_input = preprocess_face(gray)

               preds = emotion_model.predict(face_input, verbose = 0) [0]

               emotion_label, emotion_conf = smooth_emotion(preds)

        # -------------------------
        # ALERTS
        # -------------------------
        if drowsiness_score > 60:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)
            trigger_alarm()
        else:
            if emotion_label != "angry":
                stop_alarm()

        if emotion_label == "angry" and emotion_conf >= ANGER_CONFIDENCE:
            cv2.putText(frame, "ANGER ALERT!", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)
            trigger_alarm()

        # -------------------------
        # DISPLAY OVERLAY
        # -------------------------
        cv2.putText(frame, f"EAR Thr: {EAR_THRESHOLD_DYNAMIC:.3f}", (10, h-170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(frame, f"EAR: {EAR:.2f}", (10, h-140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {total_blinks}", (10, h-115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,0), 2)
        cv2.putText(frame, f"Yawns: {total_yawns}", (10, h-90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,180,80), 2)
        cv2.putText(frame, f"Drowsiness: {drowsiness_score:.1f}",
                    (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"{emotion_label} ({emotion_conf:.2f})",
                    (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if EAR_THRESHOLD_DYNAMIC is not None:
            cv2.putText(frame,
                f"EAR Thr: {EAR_THRESHOLD_DYNAMIC:.3f}",
                (10, h-170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,0),
                2
            )

        cv2.imshow("DMS", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    stop_alarm()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
