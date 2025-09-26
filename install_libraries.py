import sys
import subprocess
import platform

libraries = [
    "mediapipe",
    "matplotlib",
    "portaudio",
    "pyaudio",
    "numpy",
    "pandas",
    "SpeechRecognition",
    "pyttsx3",
    "blinker",
    "gtts",
    "openai",
    "playsound",
    "keyboard",
]

print("Installing required libraries...")
print()

# install all libraries except pyaudio first
for library in libraries:
    try:
        print("Installing", library, "...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", library])
    except subprocess.CalledProcessError:
        if platform.machine() == "arm64":  # M1/M2 Mac
            try:
                subprocess.check_call(["brew", "install", library])

                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--global-option=build_ext",
                        "--global-option=-I/opt/homebrew/include",
                        "--global-option=-L/opt/homebrew/lib",
                        library,
                    ]
                )
            except subprocess.CalledProcessError:
                print(f"Warning: Failed to install {library} with brew. Continuing anyway...")
            except FileNotFoundError:
                print("Warning: Homebrew not found. Please install Homebrew first.")
