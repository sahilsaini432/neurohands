from datetime import datetime, timedelta
from email.mime import audio
from tkinter.tix import Tree
import cv2
from pprint import pprint as _print
import threading
import queue
import asyncio

import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import playsound
import os

from detect_hands_helper import get_save_frame_size, draw_gesture_for_fixed_frame, draw_gesture
from event_manager import event_manager

class VideoCaptureThread:
    def __init__(self, args, hands):
        # Video Capture setup
        self.VideoCapture = cv2.VideoCapture(0)
        self.Width = int(self.VideoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.Height = int(self.VideoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.Hands = hands
        self.Running = True
        self.FrameQueue = queue.Queue(maxsize=3)
        self.TimedLoop = False
        self.ProcessFrame = True
        
        # Video Setup
        self.VideoWriter = None
        self.SaveVideo = False
        
        # Timed Loop Setup
        if args.time is not None:
            self.TimedLoop = True
            self.StartTime = datetime.now()
            self.Duration = timedelta(seconds=args.time)
            self.Running = datetime.now() < self.StartTime + self.Duration

    def start(self):
        if not self.VideoCapture.isOpened():
            print("ERROR: could not access the camera.")
            exit(0)

        self.Thread = threading.Thread(target=self.start_loop, daemon=False)
        self.Thread.start()
    
    def start_frame_processing(self):
        self.ProcessFrame = True
    
    def stop_frame_processing(self):
        self.ProcessFrame = False
        self.FrameQueue = queue.Queue(maxsize=3)
        cv2.destroyAllWindows()
    
    def start_recording(self, data):
        video_name = f"./output_data/{data["fileName"]}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.VideoWriter = cv2.VideoWriter(filename=video_name, fourcc=fourcc, fps=20.0, frameSize=get_save_frame_size())
        self.ProcessFrame = True
        self.SaveVideo = True
    
    def stop_recording(self):
        if self.VideoWriter is None: return
        self.SaveVideo = False
        self.ProcessFrame = False
        self.VideoWriter.release()
        self.VideoWriter = None
        cv2.destroyAllWindows()

    def start_loop(self):
        event_manager.add_listener("start_frame_processing", self.start_frame_processing)
        event_manager.add_listener("stop_frame_processing", self.stop_frame_processing)
        event_manager.add_listener("start_recording", self.start_recording)
        event_manager.add_listener("stop_recording", self.stop_recording)
        
        while self.Running:
            ret, frame = self.VideoCapture.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                self.Running = False
                break
            
            frame = cv2.flip(frame, 1)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.Hands.process(frame_rgb)
            frame = draw_gesture(result, frame)
            
            # Enter frame to display queue
            if self.ProcessFrame: self.FrameQueue.put(frame)
            
            if result.multi_hand_landmarks and self.SaveVideo:
                save_frame, _ = draw_gesture_for_fixed_frame(result=result)
                self.VideoWriter.write(save_frame)
            
            if self.TimedLoop:
                self.Running = datetime.now() < self.StartTime + self.Duration

    def stop(self):
        self.Running = False
        if self.VideoCapture.isOpened(): self.VideoCapture.release()
        cv2.destroyAllWindows()
        print("Video capture stopped.")

class VoiceCommandThread:
    def __init__(self, args):
        self.RecordVideo = args.saveVideo
        event_manager.trigger_event("stop_frame_processing")
        self.Running = True
        self.Recognizer = sr.Recognizer()
        self.Recognizer.energy_threshold = 300
        self.Recognizer.dynamic_energy_threshold = True
        self.SpeakLock = threading.Lock()
        self.Engine = pyttsx3.init()
    
    def start(self):
        self.Thread = threading.Thread(target=self.start_loop, daemon=True)
        self.Thread.start()
    
    def start_loop(self):
        with sr.Microphone(2) as source:
            self.Recognizer.adjust_for_ambient_noise(source)
            self.speak("Voice recognizer initialized...")
            while self.Running:
                try:
                    audio = self.Recognizer.listen(source)
                    command = self.Recognizer.recognize_google(audio)
                    self.process_commands(command)
                except sr.UnknownValueError as e:
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
        tts = gTTS(text=value, lang="en", slow=False)
        tts.save("temp.mp3")
        playsound.playsound("temp.mp3")
        os.remove("temp.mp3")

    def process_commands(self, commandReceived):
        command = f"{commandReceived.lower()}"
        command = command.replace(" ", "_")
        _print(f"command: {command}")
        
        if command.__contains__("start_processing"):
            self.speak("Processing frames now..")
            event_manager.trigger_event("start_frame_processing")
        elif command.__contains__("start_recording"):
            if self.RecordVideo:
                self.speak("Enter video name...")
                fileName = input("Enter filename for video: ")
                self.speak("Starting recording in 3, 2, 1...")
                event_manager.trigger_event("start_recording", {"fileName": fileName})
            else:
                self.speak("recording is not enabled")
        elif self.RecordVideo and command.__contains__("stop_recording"):
            if self.RecordVideo:
                self.speak("stoping recording now")
                event_manager.trigger_event("stop_recording")
            else:
                self.speak("recording is not enabled")
        elif command.__contains__("stop_processing"):
            self.speak("Stopping frame processing..")
            event_manager.trigger_event("stop_frame_processing")
        elif command.__contains__("exit_program") or command.__contains__("quit_program"):
            self.speak("stoping program execution...")
            event_manager.trigger_event("stop")