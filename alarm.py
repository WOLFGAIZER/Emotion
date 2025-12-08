import pygame
import time
import os

# Update this if your file is in a different folder
ALARM_PATH = "utils/alarm.mp3" 

if not os.path.exists(ALARM_PATH):
    print(f"❌ Error: Could not find {ALARM_PATH}")
else:
    print("✅ File found! Preparing to blast ears...")
    pygame.mixer.init()
    pygame.mixer.music.load(ALARM_PATH)
    pygame.mixer.music.play()
    
    # Keep script running while sound plays
    while pygame.mixer.music.get_busy():
        time.sleep(1)