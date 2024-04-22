# touchless-ui-poc
Proof of concept on hand gesture control system based on image processing.  
The project consists of
1. Hand gesture controller python app (app_gesture.py)
2. QT UI python app for a demo (app_ui.py)

### Prerequisite
1. OpenCV & OpenCV-python
2. Mediapipe & its python module
3. ScikitLearn
4. PyQT
5. Redis-server & Redis tools, python
Python packages are in `requirements.txt` file.

### Hand gesture controller
OpenCV, Mediapipe, Scikit-Learn are used for image processing and create machine learning model
to classify the gesture signal.

To run app_gesture.py
```
$ python app_gesture.py -j config.json
```
All configurations are inside config.json. Example, the camera device, `mediapipe`'s model, SVM model from `sklearn` etc.
```
{
    "cam": "/dev/v4l/by-id/usb-camera0",
    "model": {
        "landmark":"model/hand_landmarker.task",
        "gesture":"model/gesture_recognizer.task",
        "click":"click_detector/svm_model.joblib"
    }
}
```

Mediapipe's hand gesture default model is still being used in this project. 

To train `SVM` model for double-click gesture detection by using `Index-to-Thumb Distance` signal
```
$ cd click_detector 
$ python train_model.py -d click.csv -s 42
```
By click.csv is the features dataset of Index-to-Thumb signal from the experiment, generated by label_signal.py.
To generate click.csv, run
```
$ python label_signal.py -s log_4apr2024.csv -o click.csv
```

### QT UI
QT Designer is used to generate .ui file and use `pyuic` the generate the python app from .ui file.
To run the app, simply run
```
$ python app_ui.py
```

### Build binary with PyInstaller
```
$ mkdir build_gesture
$ cd build_gesture
$ pyinstaller --name="AppGesture" ../app_gesture.py
$ cp -r ../model dist/AppGesture
$ cp -r ../config.json dist/AppGesture
$ cp ../matplotlibrc dist/AppGesture

$ mkdir build_ui
$ cd build_ui
$ pyinstaller --name="AppUI" ../app_ui.py
$ cp -r ../qt/assets ./dist/AppUI/qt
```

### REDIS
Two apps, app_ui.py and app_gesture.py talk to each other (sending coordinate) via REDIS.
Installation of REDIS is needed.

### More to do
- Prerequisite (Host Machine setup)
- requirements.txt
- Training a custom model of Mediapipe's HandGesture
