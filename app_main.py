import argparse
import sys
import json
import cv2
from app_gesture import GestureController
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from app_ui import Ui_Form



class HandGestureControl_UI():
    def __init__(self):
        self.form = QWidget()
        self.ui = Ui_Form()
        self.ui.setupUi(self.form)
        #self.ui.labelOTP.setText("")

    def show(self):
        self.form.show()



def task_cv(cam_url, model_path):

    # Hand Gesture Ctrl
    g = GestureController(16, 16, control_hand='Right')
    g.config(model_path)

    # cv capture for webcam input
    # start capture
    cap = cv2.VideoCapture(cam_url)
    while cap.isOpened():
        # get frame
        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # pre-process frame
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform hand landmarks detection on the provided single image.
        # The hand landmarker must be created with the image mode.
        pos, frame_display = g.step(frame)
        print(pos)
        print("x pos:{}, y pos:{}".format(g.x, g.y))

        # display the result
        frame_display = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        cv2.imshow('MediaPipe Hands', frame_display)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()



def task_ui():
    pass
    app = QApplication(sys.argv)
    w = HandGestureControl_UI()
    w.show()
    sys.exit(app.exec())



if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-j", "--json", help="JSON file for the configuration", default='config.json')

    # Read config file (for camera source, model etc)
    args = parser.parse_args()
    f = open(args.json)
    data = json.load(f)
    cam = data['cam']
    model_path = data['model']['gesture']
    f.close()

    # Run CV task
    #task_cv(cam, model_path)

    # Run CV task
    task_ui()
