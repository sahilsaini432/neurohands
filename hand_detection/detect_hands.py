from math import fabs
import threading
import mediapipe as mp
import cv2
import argparse
import os
from pprint import pprint as _print

from detect_hands_helper import process_frame_from_filepath, save_photo
from media_capture import VideoCaptureThread, VoiceCommandThread
from event_manager import event_manager

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define the coordinates and size of the square region of interest (ROI)
x, y = 100, 100  # Top-left corner of the square
width, height = 200, 200  # Width and height of the square (same value for a square

condition = True
vc_thread = None
video_thread = None
save_photo_thread = None


def parse_args():
    # Initialize the parser
    parser = argparse.ArgumentParser("Config")
    parser.add_argument(
        "-l", "--live", required=False, action="store_true", help="Live hand detection"
    )
    parser.add_argument(
        "-p",
        "--photo",
        required=False,
        action="store_true",
        help="Photo hand detection",
    )
    parser.add_argument(
        "-d",
        "--dir",
        required=False,
        action="store_true",
        help="Detect photo from directory",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=False,
        type=str,
        help="Input path for file or directory",
    )
    parser.add_argument(
        "-t", "--time", required=False, type=int, help="timed hand detection"
    )
    parser.add_argument(
        "-c",
        "--center",
        required=False,
        action="store_true",
        help="Draw the landmark at the center of the frame",
    )
    parser.add_argument(
        "-sv", "--saveVideo", required=False, action="store_true", help="Save video"
    )
    parser.add_argument(
        "-vc",
        "--voice",
        required=False,
        action="store_true",
        help="Run with voice commands",
    )
    return parser.parse_args()


def on_stop_event():
    global vc_thread, video_thread, condition
    vc_thread.stop()
    video_thread.stop()
    condition = False


def main():
    global vc_thread, video_thread, condition
    args = parse_args()

    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        if args.live is True:
            # register events
            event_manager.add_listener("stop", on_stop_event)

            # start video capture
            video_thread = VideoCaptureThread(args=args, hands=hands)
            video_thread.start()
            condition = video_thread.Running

            # start recording if enabled
            # if args.saveVideo is True:
            # video_thread.start_recording()

            # start voice commands
            if args.voice is True:
                vc_thread = VoiceCommandThread(args=args)
                vc_thread.start()
                condition = video_thread.Running and vc_thread.Running
                event_manager.add_listener("save_take_photo", save_photo)

            # While the video thread is running
            while condition:
                if not video_thread.FrameQueue.empty():
                    frame = video_thread.FrameQueue.get()
                    cv2.imshow("Hand Detection", frame)

                if not video_thread.PhotoQueue.empty():
                    result = video_thread.PhotoQueue.get()
                    event_manager.trigger_event("save_take_photo", {"result": result})

                # Exit the loop when 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    video_thread.stop()
                    if args.voice is True:
                        vc_thread.stop()
                    condition = False

        elif args.photo is True:
            if args.input is None:
                print("ERROR: missing input image path")
                exit(0)

            process_frame_from_filepath(hands, args.input)

        elif args.dir is True:
            if args.input is None:
                print("ERROR: missing dir path")
                exit(0)

            for filename in os.listdir(args.input):
                filepath = os.path.join(args.input, filename)
                process_frame_from_filepath(hands, filepath)


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
    print("Program ended successfully")
