from hmac import new
import json
import mediapipe as mp
import cv2
import time
import numpy as np
import argparse

# Initialize the parser
parser = argparse.ArgumentParser("Config")
parser.add_argument("--saveImage", required=False)
args = parser.parse_args()

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_asset_path = "./gesture_recognizer.task"

IMAGE_WIDTH = 680
IMAGE_HEIGHT = 680

# open camera and start capturing frames
cap = cv2.VideoCapture(0)
CAP_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
CAP_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# FUNCTIONS

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
    if len(result.gestures) > 0:
        found_something = True
        print(f"Gestures - {result.gestures}")

    if len(result.handedness) > 0:
        found_something = True
        # print(f"Handedness - {result.handedness}")

    if len(result.hand_landmarks) > 0:
        found_something = True
        # print(f"Hand Landmarks - {result.hand_landmarks}")

    if len(result.hand_world_landmarks) > 0:
        found_something = True
        # print(f"Hand World Landmarks - {result.hand_world_landmarks}")

    if args.saveImage is not None and found_something is True:
        try:
            filename = f"{timestamp_ms}.jpg"
            cv2.imwrite(filename, output_image.numpy_view())
        except Exception as e:
            print(f"error showing the gesture recognition frame - {e}")

if not cap.isOpened():
    print("Error: could not access the camera.")
    exit()

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=process_result,
)
recognizer = GestureRecognizer.create_from_options(options)

# capture frames in a loop
while True:
    ret,frame = cap.read()
    if not ret:
        print("failed to capture video")
        break

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # this will show that current frame from the camera
    cv2.imshow("Camera", frame)

    mp_image = mp.Image(mp.ImageFormat.SRGB, frame)
    mp_timestamp = mp.Timestamp(int(time.time() * 100))

    recognizer.recognize_async(mp_image, int(time.time() * 100))