from math import fabs
import threading
import mediapipe as mp
import cv2
import argparse
import os
from pprint import pprint as _print

from detect_hands_helper import process_frame_from_filepath, save_photo, set_current_frame
from media_capture import VideoCaptureThread, VoiceCommandThread
from event_manager import event_manager
from dataset_collector import DatasetCollector

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

    # For live hand detection
    parser.add_argument("-l", "--live", required=False, action="store_true", help="Live hand detection")

    # For photo hand detection
    parser.add_argument(
        "-p",
        "--photo",
        required=False,
        action="store_true",
        help="Photo hand detection",
    )

    # For directory hand detection
    parser.add_argument(
        "-d",
        "--dir",
        required=False,
        action="store_true",
        help="Detect photo from directory",
    )

    # For input file or directory
    parser.add_argument(
        "-i",
        "--input",
        required=False,
        type=str,
        help="Input path for file or directory",
    )

    # For timed hand detection
    parser.add_argument("-t", "--time", required=False, type=int, help="timed hand detection")

    # For center landmark drawing
    parser.add_argument(
        "-c",
        "--center",
        required=False,
        action="store_true",
        help="Draw the landmark at the center of the frame",
    )

    # For recording video
    parser.add_argument("-sv", "--saveVideo", required=False, action="store_true", help="Save video")

    # For voice commands
    parser.add_argument(
        "-vc",
        "--voice",
        required=False,
        action="store_true",
        help="Run with voice commands",
    )

    # Dataset collection mode
    parser.add_argument("--collect", required=False, action="store_true", help="Dataset collection mode")
    parser.add_argument("--label", required=False, type=str, default=None, help="Initial label for dataset collection")
    parser.add_argument("--signer", required=False, type=str, default="unknown", help="Signer ID for dataset collection")

    return parser.parse_args()


def on_stop_event():
    global vc_thread, video_thread, condition
    vc_thread.stop()
    video_thread.stop()
    condition = False


def _run_collect_mode(args, hands):
    collector = DatasetCollector(signer=args.signer, initial_label=args.label or "")
    video_thread = VideoCaptureThread(args=args, hands=hands)
    video_thread.start()

    print("\n=== Dataset Collection Mode ===")
    print("n - Set new label (type in terminal)")
    print("m - Toggle static/sequence mode")
    print("p / space - Capture static sample")
    print("r - Hold to record sequence, release to save")
    print("q - Quit")
    print(f"Current label: '{collector.current_label}' | Mode: {collector.mode}\n")

    r_was_pressed = False
    condition = video_thread.Running

    while condition:
        if not video_thread.FrameQueue.empty():
            frame = video_thread.FrameQueue.get()
            set_current_frame(frame)

            # HUD overlay
            label_display = collector.current_label or "(no label)"
            counts = collector.stats()
            count = counts.get(collector.current_label, 0)
            hud_lines = [
                f"Label: {label_display} ({count} samples)",
                f"Mode: {collector.mode}",
                "n=new label  m=toggle  p=static  r=sequence  q=quit",
            ]
            y_pos = 30
            for line in hud_lines:
                cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_pos += 28
            cv2.imshow("Dataset Collection", frame)

        # Drain photo queue for static captures
        if not video_thread.PhotoQueue.empty():
            result = video_thread.PhotoQueue.get()
            collector.add_static(result)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            video_thread.stop()
            condition = False

        elif key == ord("n"):
            new_label = input("Enter new label: ").strip()
            if new_label:
                collector.set_label(new_label)
                print(f"Label set to: '{collector.current_label}'")

        elif key == ord("m"):
            collector.toggle_mode()
            print(f"Mode: {collector.mode}")

        elif key == ord("p") or key == ord(" "):
            if collector.mode == "static":
                event_manager.trigger_event("take_photo")

        # Hold-to-record sequence
        r_pressed = keyboard.is_pressed("r")
        if collector.mode == "sequence":
            if r_pressed and not r_was_pressed:
                if collector.current_label:
                    video_thread.start_sequence()
                    collector.begin_sequence()
                    print("Recording sequence... (hold r)")
                else:
                    print("Set a label first (press n)")
            elif r_pressed and r_was_pressed:
                while not video_thread.SequenceQueue.empty():
                    collector.add_sequence_frame(video_thread.SequenceQueue.get())
            elif not r_pressed and r_was_pressed:
                while not video_thread.SequenceQueue.empty():
                    collector.add_sequence_frame(video_thread.SequenceQueue.get())
                video_thread.stop_sequence()
                collector.end_sequence()
        r_was_pressed = r_pressed

        if not video_thread.Running:
            condition = False

    cv2.destroyAllWindows()


def main():
    global vc_thread, video_thread, condition
    args = parse_args()

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        if args.live is True:
            # register events
            event_manager.add_listener("stop", on_stop_event)
            event_manager.add_listener("save_take_photo", save_photo)

            # start video capture
            video_thread = VideoCaptureThread(args=args, hands=hands)
            video_thread.start()
            condition = video_thread.Running

            # start voice commands
            if args.voice is True:
                vc_thread = VoiceCommandThread(args=args)
                vc_thread.start()
                condition = video_thread.Running and vc_thread.Running

            # Show initial help message
            print("\n=== Keyboard Controls (press keys while video window is in focus) ===")
            print("q - Quit application")
            print("p - Take photo")
            if args.saveVideo:
                print("r - Start recording")
                print("s - Stop recording")
            print("h - Show this help message again")
            print("=====================================================================\n")

            # While the video thread is running
            while condition:
                if not video_thread.FrameQueue.empty():
                    frame = video_thread.FrameQueue.get()
                    set_current_frame(frame)
                    cv2.imshow("Hand Detection", frame)

                if not video_thread.PhotoQueue.empty():
                    result = video_thread.PhotoQueue.get()
                    event_manager.trigger_event("save_take_photo", {"result": result})

                # Check for key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    # Exit Program
                    video_thread.stop()
                    if args.voice is True:
                        vc_thread.stop()
                    condition = False

                # Take Photo
                if key == ord("p"):
                    event_manager.trigger_event("take_photo")
                    print("📸 Photo taken!")

                if args.saveVideo:

                    # Start Recording
                    if key == ord("r"):
                        video_thread.start_recording()
                        print("🔴 Recording started...")

                    # Stop Recording
                    if key == ord("s"):
                        video_thread.stop_recording()
                        print("⏹️  Recording stopped!")

                # Show help
                if key == ord("h"):
                    print("\n=== Keyboard Controls ===")
                    print("q - Quit application")
                    print("p - Take photo")
                    if args.saveVideo:
                        print("r - Start recording")
                        print("s - Stop recording")
                    print("h - Show this help")
                    print("==========================")

        elif args.collect is True:
            _run_collect_mode(args, hands)

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
