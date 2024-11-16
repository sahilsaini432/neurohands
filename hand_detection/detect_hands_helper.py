from enum import Enum
import json
from math import pi
import pickle
from google.protobuf.json_format import MessageToDict
import mediapipe as mp
import cv2
import argparse
import os
import base64
import numpy as np
from pathlib import Path
from pprint import pprint as _print
from typing import List, Dict, Any
from dataclasses import dataclass, asdict, field
from PIL import Image, PngImagePlugin
from PIL.ExifTags import TAGS

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

def get_exif_data(inputfile, key):
    image = Image.open(inputfile)
    encodedData = image.info[key]
    dataImageLandmarks = decode_hand_landmarks(encodedData)
    return dataImageLandmarks

def encode_hand_landmarks(landmarks):
    custom_hand_landmarks = HandLandmarks.from_hand_landmarks(landmarks)
    encoded_hand_landmarks = base64.b64encode(custom_hand_landmarks.to_json().encode("utf-8")).decode("utf-8")
    return encoded_hand_landmarks

def decode_hand_landmarks(encodedString):
    decoded_string = base64.b64decode(encodedString).decode("utf-8")
    decoded_hand_landmarks = HandLandmarks.from_json(decoded_string)
    return decoded_hand_landmarks

def process_image(hands, args):
    frame = cv2.imread(args.input)
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    processed_hands = []
    metadata = {}
    
    if result.multi_handedness:
        for handedness in result.multi_handedness:
            processed_hands.append(handedness.classification[0].label)
    
    index = 0
    for hand_landmarks in result.multi_hand_landmarks:
        if args.save is True:
            # change the frame to transparent if needed
            if args.transparent is True:
                height, width, _ = frame.shape
                frame = np.zeros((height, width, 3), dtype=np.uint8)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
            
            if args.transparent is True:
                transparent_frame = np.zeros((height, width, 4), dtype=np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                transparent_frame = cv2.addWeighted(transparent_frame, 1.0, frame, 1.0, 0)

            # if cropping image
            if args.crop is True:
                frame = crop_hand(frame=frame, hand_landmarks=hand_landmarks,offset=100)

            # encoding hand landmarks
            ecoded_landmarks = encode_hand_landmarks(hand_landmarks)
            
            metadata[processed_hands[index]] = ecoded_landmarks
            index = index + 1
    
    if args.save is True:
        # save the update frame
        filename = Path(args.input).name.split(".")
        filename.remove("jpg")
        filename = ".".join(filename)
        filename = f"./output_images/{filename}-output.png"

        # save metadata
        pil_image = Image.fromarray(frame)
        meta = PngImagePlugin.PngInfo()
        for key, value in metadata.items():
            meta.add_text(key, value)
        pil_image.save(filename, "PNG", pnginfo=meta)

def get_view_dimensions(frame, x_offset, y_offset, width, height):
    dimensions = {}
    frame_height, frame_width, _ = frame.shape
    x_center = frame_width / 2
    y_center = frame_height / 2
    
    dimensions["x_start"] = int(x_center - x_offset)
    dimensions["y_start"] = int(y_center - y_offset)
    dimensions["x_end"] = int(dimensions["x_start"] + width)
    dimensions["y_end"] = int(dimensions["y_start"] + height)
    return dimensions

def crop_hand(frame, hand_landmarks,offset):
    image_height, image_width, _ = frame.shape
    x_coords = [int(landmark.x * image_width) for landmark in hand_landmarks.landmark]
    y_coords = [int(landmark.y * image_height) for landmark in hand_landmarks.landmark]
    
    # Calculate the bounding box around the hand
    x_min, x_max = max(0, min(x_coords)), min(image_width, max(x_coords))
    y_min, y_max = max(0, min(y_coords)), min(image_height, max(y_coords))
    return frame[y_min-offset:y_max+offset, x_min-offset:x_max+offset]