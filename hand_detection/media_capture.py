from datetime import datetime, timedelta
import cv2
from pprint import pprint as _print
import threading
import queue

import speech_recognition as sr
import pyttsx3
import pyaudio

from detect_hands_helper import get_save_frame_size, process_landmark_for_fixed_frame, process_landmark_for_full_frame

class VideoCaptureThread:
    def __init__(self, args, hands):
        # Video Capture setup
        self.VideoCapture = cv2.VideoCapture(0)
        self.Width = int(self.VideoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.Height = int(self.VideoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.Hands = hands
        self.Running = True
        self.FrameQueue = queue.Queue()
        self.TimedLoop = False
        self.VideoOutputWriter = None
        
        # Timed Loop Setup
        if args.time is not None:
            self.TimedLoop = True
            self.StartTime = datetime.now()
            self.Duration = timedelta(seconds=args.time)
            self.Running = datetime.now() < self.StartTime + self.Duration
        
        # Video Writer Setup
        if args.video is not None:
            _print(f"args.video: {args.video}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 20.0
            self.VideoOutputWriter = cv2.VideoWriter(args.video, fourcc, fps, get_save_frame_size())

    def start(self):
        if not self.VideoCapture.isOpened():
            print("ERROR: could not access the camera.")
            exit(0)

        self.Thread = threading.Thread(target=self.start_loop, daemon=False)
        self.Thread.start()

    def start_loop(self):
        while self.Running:
            ret, frame = self.VideoCapture.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                self.Running = False
                break
            
            frame = cv2.flip(frame, 1)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.Hands.process(frame_rgb)
            frame = process_landmark_for_full_frame(result, frame)
            
            # Enter frame to display queue
            self.FrameQueue.put(frame)
            
            if result.multi_hand_landmarks and self.VideoOutputWriter is not None:
                save_frame, _ = process_landmark_for_fixed_frame(result=result, saveMetadata=False)
                self.VideoOutputWriter.write(save_frame)
            
            if self.TimedLoop:
                self.Running = datetime.now() < self.StartTime + self.Duration

    def stop(self):
        self.Running = False
        if self.VideoCapture.isOpened(): self.VideoCapture.release()
        cv2.destroyAllWindows()
        print("Video capture stopped.")

class VoiceCommandThread:
    def __init__(self, args):
        self.Running = True
        self.Recognizer = sr.Recognizer()
        self.SpeakLock = threading.Lock()
        self.Engine = pyttsx3.init()
    
    def start(self):
        self.Thread = threading.Thread(target=self.start_loop, daemon=True)
        self.Thread.start()
    
    def start_loop(self):
        with sr.Microphone() as source:
            self.speak("Voice command initialized...")
            self.Recognizer.adjust_for_ambient_noise(source=source, duration=1)
            
            while self.Running:
                try:
                    _print(f"Listening for voice command...")
                    self.Audio = self.Recognizer.listen(source)
                    _print(f"Processing voice command...")  
                    command = self.Recognizer.recognize_google(self.Audio)
                    _print(f"Voice Command: {command}")
                except sr.UnknownValueError:
                    print("sorry, i didn't catch that")
                    continue
                except sr.RequestError as e:
                    print("error with the service")
                    continue
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue
    
    def join(self):
        self.Thread.join()
    
    def stop(self):
        self.Running = False
        print("Voice command stopped.")
    
    def speak(self, value):
        with self.SpeakLock:
            self.Engine.say(value)
            self.Engine.runAndWait()