import mediapipe as mp
import cv2
import argparse
import os
from pprint import pprint as _print

from detect_hands_helper import  process_landmark_from_image
from media_capture import VideoCaptureThread, VoiceCommandThread

# Initialize the parser
parser = argparse.ArgumentParser("Config")
parser.add_argument("-l", "--live", required=False, action="store_true", help="Live hand detection")
parser.add_argument("-p", "--photo", required=False, action="store_true", help="Photo hand detection")
parser.add_argument("-d", "--dir", required=False, action="store_true", help="Detect photo from directory")
parser.add_argument("-i", "--input", required=False, type=str, help="Input path for file or directory")
parser.add_argument("-t", "--time", required=False, type=int, help="timed hand detection")
parser.add_argument("-c", "--center", required=False, action="store_true", help="Draw the landmark at the center of the frame")
parser.add_argument("-v", "--video", required=False, type=str, help="Video save mode")
parser.add_argument("-vc", "--voice", required=False, action="store_true", help="Run with voice commands")
args = parser.parse_args()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define the coordinates and size of the square region of interest (ROI)
x, y = 100, 100  # Top-left corner of the square
width, height = 200, 200  # Width and height of the square (same value for a square

# wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
# thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

if __name__ == "__main__":
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2,min_detection_confidence=0.5) as hands:
        if args.live is True:
            # start video capture
            video_thread = VideoCaptureThread(args=args, hands=hands)
            video_thread.start()
            condition = video_thread.Running
            
            # start voice commands
            if args.voice is True:
                vc_thread = VoiceCommandThread(args=args)
                vc_thread.start()
                condition = video_thread.Running and vc_thread.Running
            
            # While the video thread is running
            while condition:
                if not video_thread.FrameQueue.empty():
                    frame = video_thread.FrameQueue.get()
                    cv2.imshow("Hand Detection", frame)
                
                # Exit the loop when 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    video_thread.stop()
                    if args.voice is True: vc_thread.stop()
                    condition = False
        
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