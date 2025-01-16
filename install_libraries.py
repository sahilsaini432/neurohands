import sys
import subprocess

libraries = ["mediapipe","matplotlib", "numpy", "pandas","SpeechRecognition", "pyttsx3", "pyaudio","blinker", "gtts", "openai"]

print("Installing required libraries...")
print()

# install all libraries
for library in libraries:
    print("Installing", library, "...")
    subprocess.check_call([sys.executable, "-m", 'pip', 'install', library])