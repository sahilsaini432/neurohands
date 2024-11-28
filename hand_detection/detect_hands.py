import threading
import mediapipe as mp
import cv2
import argparse
from datetime import datetime, timedelta
import os
from pprint import pprint as _print
import speech_recognition as sr
import pyttsx3

from detect_hands_helper import HandLandmark, get_save_frame_size, process_landmark_for_fixed_frame, process_landmark_for_full_frame, process_landmark_from_image, speak, start_voiceCommands, stop_program_event

# Initialize the parser
parser = argparse.ArgumentParser("Config")
parser.add_argument("-l", "--live", required=False, action="store_true", help="Live hand detection")
parser.add_argument("-p", "--photo", required=False, action="store_true", help="Photo hand detection")
parser.add_argument("-d", "--dir", required=False, action="store_true", help="Detect photo from directory")
parser.add_argument("-i", "--input", required=False, type=str, help="Input path for file or directory")
parser.add_argument("-t", "--time", required=False, type=int, help="timed hand detection")
parser.add_argument("-c", "--center", required=False, action="store_true", help="Draw the landmark at the center of the frame")
parser.add_argument("-v", "--video", required=False, action="store_true", help="Video save mode")
parser.add_argument("-vc", "--voice", required=False, action="store_true", help="Run with voice commands")
args = parser.parse_args()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define the coordinates and size of the square region of interest (ROI)
x, y = 100, 100  # Top-left corner of the square
width, height = 200, 200  # Width and height of the square (same value for a square

# wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
# thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

with mp_hands.Hands(static_image_mode=False, max_num_hands=2,min_detection_confidence=0.5) as hands:
    if args.live is True:
        # open camera and start capturing frames
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = None
        if args.video is True:
            # Video writer
            output_filename = "./output_data/hand_visible_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 20.0
            out = cv2.VideoWriter(output_filename, fourcc, fps, get_save_frame_size())
        
        condition = True
        if args.time is not None:
            start_time = datetime.now()
            duration = timedelta(seconds=args.time)
            condition = datetime.now() < start_time + duration
        
        if not cap.isOpened():
            print("ERROR: could not access the camera.")
            exit()
        
        if args.voice is True:
            voice_thread = threading.Thread(target=start_voiceCommands, daemon=True)
            voice_thread.start()
            speak("Voice command system activated.")

        while condition:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: failed to capture video")
                break
            frame = cv2.flip(frame, 1)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
            frame = process_landmark_for_full_frame(result, frame)
            # Display frame
            cv2.imshow("Hand Detection", frame)
            
            if result.multi_hand_landmarks and args.video is True:
                save_frame, _ = process_landmark_for_fixed_frame(result=result, saveMetadata=False)
                out.write(save_frame)
            
            # Exit the loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if args.time is not None:
                condition = datetime.now() < start_time + duration
        
        cap.release()
        cv2.destroyAllWindows()
        
        if args.voice is True:
            voice_thread.join()

    elif args.photo is True:
        if args.input is None:
            print("ERROR: missing input image path")
            exit(0)
        
        process_landmark_from_image(hands,args.input, args)

    elif args.dir is True:
        if args.input is None:
            print("ERROR: missing dir path")
            exit(0)
        
        for filename in os.listdir(args.input):
            filepath = os.path.join(args.input, filename)
            process_landmark_from_image(hands, filepath, args)