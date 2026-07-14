# Drowsiness Alert System

This project detects driver drowsiness by classifying whether the driver's eyes are open or closed and sounding an alarm when prolonged eye closure is detected.

Overview

- Uses the webcam (OpenCV) to capture frames.
- Detects face and eyes using Haar cascades located in the `haar cascade files/` folder.
- A trained CNN model in `models/cnnCat2.h5` classifies each eye image as Open or Closed.
- A running `score` aggregates consecutive closed-eye frames; if it exceeds a threshold an alarm plays.

Files and folders

- `drowsiness detection.py`: main script to run the system.
- `model.py`: model definition (used only if model weights need restoring).
- `haar cascade files/`: contains Haar cascade XML files used for face/eye detection. Ensure these XML files are present.
- `models/`: contains the trained model file `cnnCat2.h5`.
- `alarm.wav`: alarm sound file used when drowsiness is detected.

Requirements

- Python 3.8+ (recommended)
- Packages: `opencv-python`, `numpy`, `tensorflow` (or `tensorflow-cpu`), `pygame`

Quick setup

1. Create and activate a virtual environment:

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1    # PowerShell on Windows
```

2. Install dependencies:

```powershell
pip install opencv-python numpy pygame tensorflow
```

Running the system

1. From the project root run:

```powershell
python "Drowsiness-Alert-System\drowsiness detection.py"
```

Controls while running

- `s`: stop the current alarm (camera keeps running). The alarm will be allowed again after you open your eyes and then close them again.
- `q`: quit the application (stops camera and any sound).

Notes and troubleshooting

- If you hear no alarm sound, check your system audio and ensure `alarm.wav` exists in the project root.
- If the webcam feed is blank or frozen, confirm the correct camera index is used in `drowsiness detection.py` (default is `0`).
- If the model fails to load, `model.py` contains a fallback architecture that the script will attempt to load weights into; check console output for detailed errors.
- Ensure the Haar cascade XML files are present in `haar cascade files/` — missing or empty files will prevent face/eye detection.

Behavior details

- The script maintains a `score` value that increments for each frame with both eyes detected closed and decrements when eyes are open. When `score` exceeds a threshold the alarm starts. The alarm stops automatically when eyes are detected open for a short period, or you can press `s` to stop the currently playing alarm.

No external links are included in this README.

## Project Summary

This repository contains a lightweight driver drowsiness detection system built with OpenCV and a small convolutional neural network. The system captures frames from a webcam, detects the face and eyes using Haar cascades, classifies each eye crop as Open or Closed using a trained model, and raises an audible alarm when the user's eyes remain closed for a sustained period.

Key behaviors:
- The program raises an alarm when a running `score` (counts consecutive closed-eye frames) exceeds a threshold.
- Press `s` to stop the currently sounding alarm while keeping the camera feed running. The alarm will be allowed again after you open your eyes (detected) and then close them again.
- Press `q` to quit the application and stop the camera and any sound.

## Files in this repository
- `drowsiness detection.py` — main script to run the detection and alarm.
- `model.py` — model architecture (used as a fallback if the saved model needs restoring).
- `haar cascade files/` — contains Haar cascade XML files for face/eye detection. Ensure the XML files are present and non-empty.
- `models/` — contains the trained model file `cnnCat2.h5` (weights/architecture).
- `alarm.wav` — alarm audio file used when drowsiness is detected.

## Requirements
- Python 3.8 or later
- Packages: `opencv-python`, `numpy`, `tensorflow` (or `tensorflow-cpu`), `pygame`

Install dependencies:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1   # PowerShell on Windows
pip install --upgrade pip
pip install opencv-python numpy pygame tensorflow
```

## Running the system

From the project root run:

```powershell
python "Drowsiness-Alert-System\drowsiness detection.py"
```

While running, the program displays a window with the webcam feed and a small status overlay (score). Use the keyboard controls described above to stop the alarm or quit.

## Troubleshooting
- No camera feed: ensure no other application is using the webcam and try changing the capture index in the script (`cv2.VideoCapture(0)` to `1`, etc.).
- No alarm sound: verify `alarm.wav` exists in the project root and system audio is enabled. On some systems `pygame` may require additional audio backends.
- Model load errors: if `cnnCat2.h5` fails to load directly, the script includes a fallback architecture in `model.py` and will attempt to load weights into it — check console output for details.
- Haar cascades missing: confirm the XML files are present in `haar cascade files/`; empty or corrupted XML files will prevent face/eye detection.



