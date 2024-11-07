import mediapipe as mp
import cv2
import csv
import numpy as np
import pandas as PD

# result classes
from gesture_recognition_result import Gesture

# variables
args = None
gestures_dataset = None
stored_gestures_names = None
CAP_HEIGHT = None
CAP_WIDTH = None

def initialize_helper_variables(_args, _width, _height):
    global args, CAP_WIDTH, CAP_HEIGHT
    args = _args
    CAP_WIDTH = _width
    CAP_HEIGHT = _height
    
    # load csv files
    reload_csv("gestures")

def reload_csv(filename):
    global gestures_dataset
    global stored_gestures_names
    gestures_dataset = PD.read_csv(f'{filename}.csv')
    stored_gestures_names = gestures_dataset.iloc[:, 0].values

def write_toCsv(filename, data):
    with open(file=f"{filename}.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def writeToFile(filename, data):
    with open(filename, "w") as file:
        for row in data:
            file.write(" ".join(map(str, row)) + "\n")

# convert mediapipe image to image compatible with cv2
def mediapipe_image_to_matlike(mediapipe_image):
    # convert Mediapipe image to numpy array
    image_data = mediapipe_image.numpy_view()
    
    if mediapipe_image.format == mp.ImageFormat.SRGB:
        image_array = np.frombuffer(image_data, dtype=np.uint8).reshape((CAP_HEIGHT, CAP_WIDTH, 3))
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    elif mediapipe_image.format == mp.ImageFormat.GRAY8:
        image_array = np.frombuffer(image_data, dtype=np.uint8).reshape((CAP_HEIGHT, CAP_WIDTH))
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Unsupported image format")
    return image_array

# create a gesture recognizer instance with the live stream mode:
def process_result(result, output_image, timestamp_ms):
    # if anything was found by the frame save the image to disk
    found_something = False
    if args.cgd is not None and len(result.gestures) > 0:
        global stored_gestures_names
        found_something = True
        for index in range(0, len(result.gestures)):
            gesture = Gesture.from_category(result.gestures[index][0])
            if gesture.categoryName != 'None' and not stored_gestures_names.__contains__(gesture.categoryName):
                write_toCsv("gestures", gesture.csv_data())
                reload_csv("gestures")

    if args.chd is not None and len(result.handedness) > 0:
        found_something = True
        # print(f"Handedness - {result.handedness}")

    if args.chld is not None and len(result.hand_landmarks) > 0:
        found_something = True
        # print(f"Hand Landmarks - {result.hand_landmarks}")

    if args.chwld is not None and len(result.hand_world_landmarks) > 0:
        found_something = True
        # print(f"Hand World Landmarks - {result.hand_world_landmarks}")

    if args.si is not None and found_something is True:
        try:
            filename = f"{timestamp_ms}.jpg"
            cv2.imwrite(filename, output_image.numpy_view())
        except Exception as e:
            print(f"error showing the gesture recognition frame - {e}")