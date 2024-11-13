from enum import Enum
import mediapipe as mp
import cv2
import argparse
import os
from pathlib import Path
from pprint import pprint as _print

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class HandLandmark(Enum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

def process_image(hands, inputfile, save):
    frame = cv2.imread(inputfile)
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    for hand_landmarks in result.multi_hand_landmarks:
        if save is True:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
    
    if save is True:
        # save the update frame
        filename = Path(inputfile).name.split(".")
        filename.remove("jpg")
        filename = ".".join(filename)
        filename = f"./output_images/{filename}-output.jpg"
        
        cv2.imwrite(filename, frame)