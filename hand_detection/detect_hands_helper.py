from enum import Enum
import json
from math import pi
from google.protobuf.json_format import MessageToDict
import mediapipe as mp
import cv2
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
save_width = 550
save_height = 550

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

def get_save_frame_size():
    return (save_width*2, save_height)

def add_text_to_frame(text, frame):
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (255, 255, 255)

    # Get the text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Calculate the x-coordinate to center the text horizontally
    x = (frame.shape[1] - text_width) // 2  # Center horizontally

    # Calculate the y-coordinate to position the text at the bottom
    y = frame.shape[0] - 20 

    # Put the text on the frame
    return cv2.putText(frame, text, (x, y), font, font_scale, font_color, font_thickness)

def process_landmark_for_fixed_frame(result, saveMetadata):
    processed_hands = []
    metadata = {}
    
    if result.multi_handedness:
        for handedness in result.multi_handedness:
            processed_hands.append(handedness.classification[0].label)
            
    frames = {}
    index = 0
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            frame = np.zeros(shape=(save_height, save_width, 3), dtype=np.uint8)
            frame = draw_in_center(frame=frame, hand_landmarks=hand_landmarks)
            frame = add_text_to_frame(processed_hands[index], frame)
            frames[processed_hands[index]] = frame
            
            if saveMetadata is True:
                # encoding hand landmarks
                ecoded_landmarks = encode_hand_landmarks(hand_landmarks)
                metadata[processed_hands[index]] = ecoded_landmarks
            index = index + 1
    
    if len(frames.values()) == 1:
        emptyFrame = np.zeros(shape=(save_height, save_width, 3), dtype=np.uint8)
        if frames.keys().__contains__("Left"):
            frames["Right"] = emptyFrame
        else:
            frames["Left"] = emptyFrame
    
    return np.hstack((frames["Left"], frames["Right"])), metadata

def process_landmark_for_full_frame(result, incoming_frame):
    frame_height, frame_width, _ = incoming_frame.shape
    full_frame = np.zeros(shape=(frame_height, frame_width, 3), dtype=np.uint8)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(full_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

    return full_frame

def process_landmark_from_image(hands, filepath, args):
    frame = cv2.imread(filepath)
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    frame_to_save = None
    if args.center is True:
        frame_to_save, metadata = process_landmark_for_fixed_frame(result,True)
    else:
        frame_to_save = process_landmark_for_full_frame(result, frame)

    # save the update frame
    filename = Path(filepath).name.split(".")
    filename.remove("jpg")
    filename = ".".join(filename)
    filename = f"./output_images/{filename}-output.png"
    
    pil_image = Image.fromarray(frame_to_save)
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

def draw_in_center(frame, hand_landmarks):
    target_height, target_width, _ = frame.shape
    
    x_coords = [landmark.x for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y for landmark in hand_landmarks.landmark]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Calculate centering offsets
    center_x = target_width // 2
    center_y = target_height // 2
    box_center_x = (min_x + max_x) / 2 * target_width
    box_center_y = (min_y + max_y) / 2 * target_height

    offset_x = center_x - box_center_x
    offset_y = center_y - box_center_y
    
    # Draw landmarks centered in the target frame
    for landmark in hand_landmarks.landmark:
        # Scale normalized coordinates to the target frame
        x = int(landmark.x * target_width + offset_x)
        y = int(landmark.y * target_height + offset_y)

        # Draw a circle at each landmark
        cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-2)
        
    # draw the connections between the landmarks
    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_landmark = hand_landmarks.landmark[start_idx]
        end_landmark = hand_landmarks.landmark[end_idx]

        # Scale and center start and end points
        start_point = (
            int(start_landmark.x * target_width + offset_x),
            int(start_landmark.y * target_height + offset_y),
        )
        end_point = (
            int(end_landmark.x * target_width + offset_x),
            int(end_landmark.y * target_height + offset_y),
        )

        # Draw the connection
        cv2.line(frame, start_point, end_point, color=(255, 0, 0), thickness=2)
    return frame