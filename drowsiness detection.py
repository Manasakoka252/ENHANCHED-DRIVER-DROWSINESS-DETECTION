import cv2
import os
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np
from pygame import mixer
import time

mixer.init()
# enable debug logging to terminal for troubleshooting
DEBUG = True
# base directory (script directory) to make paths robust regardless of CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    sound = mixer.Sound(os.path.join(BASE_DIR, 'alarm.wav'))
except Exception:
    # fallback: try simple name (in case working directory differs)
    try:
        sound = mixer.Sound(os.path.join(BASE_DIR, 'alarm.wav'))
    except Exception:
        sound = None
# channel used to control playback (so we can stop the alarm)
channel = None
# if True, alarm is silenced by user until score drops below threshold
silenced = False


face = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'haar cascade files', 'haarcascade_frontalface_alt.xml'))
leye = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'haar cascade files', 'haarcascade_lefteye_2splits.xml'))
reye = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'haar cascade files', 'haarcascade_righteye_2splits.xml'))

lbl = ['Close', 'Open']

try:
    # try loading without compiling (avoids optimizer/legacy issues)
    model = load_model(os.path.join(BASE_DIR, 'models', 'cnncat2.h5'), compile=False)
except Exception:
    # If loading the legacy HDF5 full model fails due to keras version mismatch,
    # rebuild the architecture (from model.py) and load weights from the HDF5 file.
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
        MaxPooling2D(pool_size=(1,1)),
        Conv2D(32,(3,3),activation='relu'),
        MaxPooling2D(pool_size=(1,1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(1,1)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    try:
        # load weights from the HDF5 file that contains 'model_weights'
        model.load_weights(os.path.join(BASE_DIR, 'models', 'cnnCat2.h5'))
    except Exception:
        # try lowercase name as fallback
        try:
            model.load_weights(os.path.join(BASE_DIR, 'models', 'cnncat2.h5'))
        except Exception:
            # if weights cannot be loaded, proceed with uninitialized model (will likely misclassify)
            pass
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
# default prediction classes (1 -> Open)
rpred_class = [1]
lpred_class = [1]

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = model.predict(r_eye)
        rpred_class = rpred.argmax(axis=-1)

        if rpred_class[0] == 1:
            lbl = 'Open'
        if rpred_class[0] == 0:
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict(l_eye)
        lpred_class = lpred.argmax(axis=-1)

        if lpred_class[0] == 1:
            lbl = 'Open'
        if lpred_class[0] == 0:
            lbl = 'Closed'
        break

    if rpred_class[0] == 0 and lpred_class[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0

    # debug: show score changes
    try:
        if DEBUG:
            print(f"DEBUG: score={score}, silenced={silenced}")
    except Exception:
        pass

    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    if score > 15:
        cv2.imwrite(os.path.join(BASE_DIR, 'image.jpg'), frame)
        # start alarm if not muted and not already playing
        try:
            if (not silenced) and sound is not None:
                # play in loop until stopped
                if channel is None or not channel.get_busy():
                    channel = sound.play(-1)
                    if DEBUG:
                        print("DEBUG: alarm started (sound.play)")
        except Exception:
            if DEBUG:
                print("DEBUG: exception trying to start alarm")
            pass
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    
    cv2.imshow('frame', frame)
    # read a single keypress (handle quit and mute)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        # stop alarm now; keep camera running. alarm will be allowed again
        # once score drops below or equal to threshold (i.e., eyes open)
        try:
            mixer.stop()
        except Exception:
            pass
        try:
            if channel is not None:
                try:
                    if channel.get_busy():
                        channel.stop()
                except Exception:
                    pass
                channel = None
        except Exception:
            pass
        silenced = True
        if DEBUG:
            print("DEBUG: silenced set True by key 's'")

    # stop alarm when score drops below threshold
    # clear silenced when we detect eyes open or score drops below threshold
    if score <= 15 or (rpred_class[0] == 1 or lpred_class[0] == 1):
        if silenced:
            silenced = False
            if DEBUG:
                print("DEBUG: silenced cleared because eyes detected open or score <= 15")
        try:
            if channel is not None and channel.get_busy():
                channel.stop()
                channel = None
                if DEBUG:
                    print("DEBUG: alarm stopped because eyes open or score <= 15")
        except Exception:
            try:
                if sound is not None:
                    sound.stop()
                    if DEBUG:
                        print("DEBUG: sound.stop() called in exception handler")
            except Exception:
                pass

cap.release()
cv2.destroyAllWindows()
# ensure alarm stopped on exit
try:
    if channel is not None and channel.get_busy():
        channel.stop()
except Exception:
    pass
try:
    if sound is not None:
        sound.stop()
except Exception:
    pass
