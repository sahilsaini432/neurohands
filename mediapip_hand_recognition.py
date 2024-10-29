import mediapipe as mp
import cv2
import time
import numpy as np
from cv2.cuda import GpuMat

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

if not cap.isOpened():
    print("Error: could not access the camera.")
    exit()
    
def writeToFile(filename, data):
    with open(filename, "w") as file:
        for row in data:
            file.write(" ".join(map(str, row)) + "\n")  

# create a gesture recognizer instance with the live stream mode:
def print_result(result, output_image, timestamp_ms):
    if output_image is not None:
        try:
            image_data = np.array(object=output_image.numpy_view(), dtype=np.uint8)
            image_bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
            umat_image = cv2.UMat(image_bgr)
            # print(f"umat_image -: {umat_image.get()}")
            cv2.imshow("Gesture_Recognition_Feedback", umat_image.get())
        except Exception as e:
            print(f"error showing the gesture recognition frame - {e}")
    else:
        print(F"output image not found")
    
    # result.gestures = gestures found from the frame
    # result.handedness = ???
    # result.hand_landmarks = ??
    # result.hand_world_landmarks = ??
    
    if len(result.gestures) > 0:
        print(f"gestures -: {result.gestures}")
        
    if len(result.handedness) > 0:
        print(f"handedness -: {result.handedness}")

    if len(result.hand_landmarks) > 0:
        print(f"hand_landmarks -: {result.hand_landmarks}")

    if len(result.hand_world_landmarks) > 0:
        print(f"hand_world_landmarks -: {result.hand_world_landmarks}")

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
)
recognizer = GestureRecognizer.create_from_options(options)

# capture frames in a loop
while True:
    ret,frame = cap.read()
    # print(f"frame -: {frame}")
    if not ret:
        print("failed to capture video")
        break

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    # cv2.imshow("Camera", frame)

    mp_image = mp.Image(mp.ImageFormat.SRGB, resized_frame)
    mp_timestamp = mp.Timestamp(int(time.time() * 100))

    recognizer.recognize_async(mp_image, int(time.time() * 100))