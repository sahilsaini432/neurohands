import mediapipe as mp
import cv2
import time
import argparse

# helpers
from helpers import initialize_helper_variables, process_result

# Initialize the parser
parser = argparse.ArgumentParser("Config")
parser.add_argument("--saveImage", required=False)
args = parser.parse_args()

# open camera and start capturing frames
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# initialize helper
initialize_helper_variables(args, width, height)

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_asset_path = "./gesture_recognizer.task"

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