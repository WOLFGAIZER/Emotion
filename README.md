🧠 Driver Emotion & Fatigue Detection System

A lightweight, intelligent driver monitoring system powered by Mediapipe and ShuffleNetV2.

🚗 Overview

This project detects driver fatigue, drowsiness, anger, and other emotional states in real time using a live video feed from your webcam.
It integrates Mediapipe FaceMesh, OpenCV, and a custom-trained ShuffleNetV2 CNN to provide accurate, fast, and resource-efficient analysis — making it ideal for embedded or low-power systems.

🎯 Key Features

✅ Real-time Emotion Recognition — Detects emotions like angry, fatigue, drowsy, neutral, happy, sad, surprise.
✅ Drowsiness Detection — Based on both Eye Aspect Ratio (EAR) and emotion probability.
✅ Fatigue Monitoring — Identifies droopy eyes, yawning, and prolonged fatigue states.
✅ Augmentation Support — Simulates multiple face variants (glasses, beard, hairstyle, lighting, tilt).
✅ Alarm System — Plays an alert sound when the driver shows signs of fatigue or anger.
✅ Hybrid Dataset Training — Combines FER2013 (large-scale) and JAFFE (fine-tuning) for higher accuracy.
✅ Optimized ShuffleNetV2 Backbone — Lightweight CNN ideal for real-time inference on CPUs.

conda create -n WOLF python=3.11
conda activate WOLF

🔹 Eye Aspect Ratio (EAR)

Used to detect blinking and eye closure:


If EAR < 0.22 for 20 consecutive frames → Drowsiness alert!

🔹 Emotion Classification

The ShuffleNetV2 CNN classifies faces into multiple emotional states based on training data.

🔹 Data Augmentation

augmentor.py generates realistic facial variations for improved model generalization:

With/without glasses

Beard

Different hairstyles

Varying light and angles