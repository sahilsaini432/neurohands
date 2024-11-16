from ast import arg
import mediapipe as mp
import cv2
import argparse
import os
from pprint import pprint as _print

from detect_hands_helper import HandLandmark, get_view_dimensions, process_image

# Initialize the parser
parser = argparse.ArgumentParser("Config")
parser.add_argument("-l", "--live", required=False, action="store_true", help="Live hand detection")
parser.add_argument("-p", "--photo", required=False, action="store_true", help="Photo hand detection")
parser.add_argument("-d", "--dir", required=False, action="store_true", help="Detect photo from directory")
parser.add_argument("-i", "--input", required=False, type=str, help="Input path for file or directory")
parser.add_argument("-s", "--save", required=False, action="store_true", help="Save the processed frame/s")
parser.add_argument("-c", "--crop", required=False, action="store_true", help="Crop images to only show hands")
args = parser.parse_args()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define the coordinates and size of the square region of interest (ROI)
x, y = 100, 100  # Top-left corner of the square
width, height = 200, 200  # Width and height of the square (same value for a square)

with mp_hands.Hands(static_image_mode=False, max_num_hands=2,min_detection_confidence=0.5) as hands:
    if args.live is True:
        # open camera and start capturing frames
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not cap.isOpened():
            print("ERROR: could not access the camera.")
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: failed to capture video")
                break
            frame = cv2.flip(frame, 1)
            
            # offset = width / 4
            if args.crop is True:
                dimensions = get_view_dimensions(frame, x_offset=250,y_offset=350, width=700, height=700)
                cv2.rectangle(frame, (dimensions["x_start"], dimensions["y_start"]), (dimensions["x_end"], dimensions["y_end"]), (0, 255, 0), 2)
                frame = frame[dimensions["y_start"]:dimensions["y_end"], dimensions["x_start"]:dimensions["x_end"]]

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

                    # wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    # thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Display frame
            cv2.imshow("Hand Detection", frame)

            # Exit the loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    elif args.photo is True:
        if args.input is None:
            print("ERROR: missing input image path")
            exit(0)
        
        process_image(hands, args)

    elif args.dir is True:
        if args.input is None:
            print("ERROR: missing dir path")
            exit(0)
        
        for filename in os.listdir(args.input):
            filepath = os.path.join(args.input, filename)
            process_image(hands, args)