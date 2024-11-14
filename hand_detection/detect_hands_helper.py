from enum import Enum
import json
import pickle
from google.protobuf.json_format import MessageToDict
import mediapipe as mp
import cv2
import argparse
import os
import base64
from pathlib import Path
from pprint import pprint as _print
from typing import List, Dict, Any
from dataclasses import dataclass, asdict, field

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

@dataclass
class Landmark:
    x: float
    y: float
    z: float
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

@dataclass
class HandLandmarks:
    landmarks: List[Landmark] = field(default_factory=list)
    
    @classmethod
    def from_hand_landmarks(self, hand_landmarks):
        instance = self()
        hand_landmark_indexes = [landmark.value for landmark in HandLandmark]
        for index in hand_landmark_indexes:
            landmark = hand_landmarks.landmark[index]
            instance.landmarks.append(Landmark(landmark.x,landmark.y,landmark.z))
        return instance
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @staticmethod
    def from_json(json_str: str) -> 'HandLandmarks':
        data = json.loads(json_str)
        data['landmarks'] = [Landmark(**landmark) for landmark in data.get('landmarks', [])]
        return HandLandmarks(**data)

def process_image(hands, inputfile, save):
    frame = cv2.imread(inputfile)
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    processed_hands = []
    
    if result.multi_handedness:
        for handedness in result.multi_handedness:
            processed_hands.append(handedness.classification[0].label)
    
    index = 0
    for hand_landmarks in result.multi_hand_landmarks:
        if save is True:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
            
            # encoding hand landmarks
            custom_hand_landmarks = HandLandmarks.from_hand_landmarks(hand_landmarks)
            encoded_hand_landmarks = base64.b64encode(custom_hand_landmarks.to_json().encode("utf-8")).decode("utf-8")
            
            # decoding hand landmarks
            decoded_string = base64.b64decode(encoded_hand_landmarks).decode("utf-8")
            decoded_hand_landmarks = HandLandmarks.from_json(decoded_string)
            _print(decoded_hand_landmarks.landmarks)
            index = index + 1
    
    if save is True:
        # save the update frame
        filename = Path(inputfile).name.split(".")
        filename.remove("jpg")
        filename = ".".join(filename)
        filename = f"./output_images/{filename}-output.jpg"
        
        cv2.imwrite(filename, frame)