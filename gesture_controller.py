import cv2
import math
import csv
import matplotlib
matplotlib.use('svg')
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import normalize
import joblib
import click_detector.feature_extract as fe

#HANDPOSE_START = 'Closed_Fist'
#HANDPOSE_CONFIRM = 'Love'
#HANDPOSE_CANCEL = 'Open_Palm'
#HANDPOSE_UP = 'Thumb_Down'
#HANDPOSE_DOWN = 'Thumb_Up'
#HANDPOSE_RIGHT = 'Pointing_Up'
#HANDPOSE_LEFT = 'Victory'
HANDPOSE_START = 'Open_Palm'
HANDPOSE_CONFIRM = 'Thumb_Up'
HANDPOSE_CANCEL = 'Thumb_Down'
D_BUFFERSIZE = 24



class Hand:
    def __init__(self, handedness):
        self.handedness = None
        self.gesture = None
        self.landmarks = None



class GestureController:
    def __init__(self, gridsize_x, gridsize_y, finger_distance_max=0.3, overlay=True, selfie=True, logging=False):
        self.overlay_flag = overlay
        self.selfie = selfie
        self.logging = logging
        self.state = 'IDLE'
        self.slot_num = 0
        self.x_slot = 0
        self.y_slot = 0
        self.slot_size = gridsize_x * gridsize_y
        self.gridsize_x = gridsize_x
        self.gridsize_y = gridsize_y
        self.d_previous = None
        self.finger_distance_max = finger_distance_max
        self.d_buffer = np.zeros(D_BUFFERSIZE)
        self.overlap = np.zeros(D_BUFFERSIZE//2)
        self.ptr = 0
        self.click_detected_cnt=0

    def config(self, hand_model_path, click_model_path):
        # Create a gesture recognizer instance with the image mode:
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=3,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.recognizer = GestureRecognizer.create_from_options(options)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        # calculate threshold values
        self.threshold_fingerdist = []
        stepsize = float(self.finger_distance_max) / self.slot_size
        for i in range(1, self.slot_size):
            self.threshold_fingerdist.append(i*stepsize)
        self.threshold_x = None
        self.threshold_y = None
        # SVM model for click, double-click detection
        self.click_model = joblib.load(click_model_path)
        # setup DSP parameters
        # create ref signal for correlation feature
        framesize = 24
        amplitude = 2
        frequency = 2 # 2 sinewave within framesize
        x_value = np.linspace(0, framesize, framesize)
        self.autocorr_ref = amplitude*np.sin(2*np.pi*frequency*x_value + np.pi/2.5)

    def applyHandDetector(self, frame):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detect_result = self.recognizer.recognize(mp_img)
        return detect_result

    def extractHandInfo(self, detect_result):
        lh = None
        rh = None
        l_gestures = detect_result.gestures
        l_handedness = detect_result.handedness
        l_hand_landmarks = detect_result.hand_landmarks
        for i in range(len(l_handedness)):
            handedness = l_handedness[i][0].category_name
            gesture = l_gestures[i][0].category_name
            landmarks = l_hand_landmarks[i]
            if handedness == 'Left':
                lh = Hand('Left')
                lh.gesture = gesture
                lh.landmarks = landmarks
            if handedness == 'Right':
                rh = Hand('Right')
                rh.gesture = gesture
                rh.landmarks = landmarks
        return lh, rh

    def findHandPosition(self, hand_obj, frame):
        # let middle finger MCP as the center point of hand
        x_hand = hand_obj.landmarks[9].x
        y_hand = hand_obj.landmarks[9].y
        x_hand, y_hand = self.denormalizeLandmark(frame, x_hand, y_hand)
        # draw point as center of hand (if overlay is True)
        if frame is not None:
            if self.selfie:
                x_hand, y_hand = self.flipCoordinate(frame, x_hand, y_hand)
            cv2.circle(frame, (x_hand, y_hand), 20, (0, 0, 255), -1)
        return x_hand, y_hand, frame

    def handPosToSlotXY(self, x_hand, y_hand, frame):
        x_slot = None
        y_slot = None
        # calc threshold at first call
        if not self.threshold_x:
            h, w, _ = frame.shape
            self.threshold_x = []
            self.threshold_y = []
            for i in range(1, self.gridsize_x + 1):
                tmp = i * w/self.gridsize_x
                self.threshold_x.append(tmp)
            for i in range(1, self.gridsize_y + 1):
                tmp = i * h/self.gridsize_y
                self.threshold_y.append(tmp)
        # thresholding from less to most value
        for i in range(0, len(self.threshold_x)):
            if x_hand < self.threshold_x[i]:
                x_slot = i
                break
        for i in range(0, len(self.threshold_y)):
            if y_hand < self.threshold_y[i]:
                y_slot = i
                break
        return x_slot, y_slot

    def findSlotNumber(self, x, y):
        if x > self.gridsize_x:
            x = self.gridsize_x
        if y > self.gridsize_y:
            y = self.gridsize_y
        index = (self.gridsize_x * y) + x + 1
        return index

    def findFingerDistance(self, hand_obj, frame=None):
        x_thumb = hand_obj.landmarks[4].x
        x_index = hand_obj.landmarks[8].x
        y_thumb = hand_obj.landmarks[4].y
        y_index = hand_obj.landmarks[8].y
        d = math.dist((x_thumb, y_thumb), (x_index, y_index))
        if frame is not None and self.overlay_flag:
            x_thumb, y_thumb = self.denormalizeLandmark(frame, x_thumb, y_thumb)
            x_index, y_index = self.denormalizeLandmark(frame, x_index, y_index)
            if self.selfie:
                x_thumb, y_thumb = self.flipCoordinate(frame, x_thumb, y_thumb)
                x_index, y_index = self.flipCoordinate(frame, x_index, y_index)
            cv2.circle(frame, (x_thumb, y_thumb), 20, (0, 0, 255), -1)
            cv2.circle(frame, (x_index, y_index), 20, (0, 0, 255), -1)
            cv2.line(frame, (x_thumb, y_thumb), (x_index, y_index), (255, 255, 0), 5)
            cv2.putText(frame, "%.3f" % d, (x_index - 10, y_index - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)
        return d, frame

    def flipCoordinate(self, frame, x, y):
        # This function is used in overlayFrame() when selfie-mode is used
        height, width, _ = frame.shape
        x_flip = width - x
        y_flip = y
        return x_flip, y_flip

    def denormalizeLandmark(self, frame, x, y):
        height, width, _ = frame.shape
        x_denom = int(x*width)
        y_denom = int(y*height)
        return x_denom, y_denom

    def applyFilter(self, d, alpha=0.5):
        if self.d_previous is None:
            self.d_previous = d
        else:
            d = (alpha * d) + ((1 - alpha) * self.d_previous)
            self.d_previous = d
        return d

    def click_detector_svm(self, finger_dist, norm=True):
        # feature extraction
        finger_dist = np.array(finger_dist)
        features = fe.getFeatures(finger_dist, self.autocorr_ref)
        features = features.reshape(1,-1)
        if norm:
            features = normalize(features, norm="l2")
        # classify
        y_predict = self.click_model.predict(features)
        return y_predict

    def overlayFrame(self, frame, detect_result):
        l_gestures = detect_result.gestures
        l_hand_landmarks = detect_result.hand_landmarks
        l_handedness = detect_result.handedness
        l_text_gesture = []
        l_text_x = []
        l_text_y = []

        # Draw hand landmark
        for i in range(len(l_gestures)):
            gesture = l_gestures[i]
            hand_landmarks = l_hand_landmarks[i]
            handedness = l_handedness[i]

            #Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            MARGIN = 20  # pixels
            height, width, _ = frame.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            if self.selfie:
                # flip the text coordinate
                text_x, text_y = self.flipCoordinate(frame, text_x, text_y)

            l_text_gesture.append(gesture[0].category_name)
            l_text_x.append(text_x)
            l_text_y.append(text_y)

        if self.selfie:
            # Flip the image horizontally for a selfie-view display.
            frame = cv2.flip(frame, 1)

        # Draw text for gesture type
        FONT_SIZE = 1
        FONT_THICKNESS = 2
        TEXT_COLOR = (200, 240, 240)
        for g, x, y in zip(l_text_gesture, l_text_x, l_text_y):
            cv2.putText(frame, str(g), (x - 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        return frame

    def step(self, frame):
        selected_slot_num = None
        confirm_flag = False

        # run detector
        detect_result = self.applyHandDetector(frame)

        # extract detect_result to left hand, right hand infomation
        # None -> not detected
        lh, rh = self.extractHandInfo(detect_result)

        # overlay if hand is detected
        # else, just pass or flip the frame (if selfie mode is used)
        if not lh and not rh:
            if self.selfie:
                # Flip the image horizontally for a selfie-view display.
                frame = cv2.flip(frame, 1)
            return confirm_flag, selected_slot_num, frame
        else:
            if self.overlay_flag:
                frame = self.overlayFrame(frame, detect_result)
            else:
                if self.selfie:
                    frame = cv2.flip(frame, 1)

        if rh:
            # start event
            if self.state == 'IDLE':
                if rh.gesture == HANDPOSE_START:
                    print("IDLE to SELECT ITEM")
                    self.state = 'SELECT_ITEM'

            elif self.state == 'SELECT_ITEM':
                if rh.gesture == HANDPOSE_CONFIRM:
                    confirm_flag = True
                    # reset and goto IDLE
                    print("Confirmed! To IDLE")
                    selected_slot_num = self.slot_num
                    self.slot_num = 0
                    self.state = 'IDLE'

                elif rh.gesture == HANDPOSE_CANCEL:
                    print("Cancel, To IDLE")
                    self.state = 'IDLE'
                    self.slot_num = 0

                else:
                    # Thumb-to-Index Distance
                    d, frame = self.findFingerDistance(rh, frame)
                    d = self.applyFilter(d, alpha=0.6)
                    # logging the distance value as .csv file
                    if self.logging:
                        with open("log.csv", "a", newline="") as csvfile:
                            # Create a csv writer object
                            writer = csv.writer(csvfile)

                    self.d_buffer[self.ptr] = d
                    self.ptr += 1
                    svm_result = None
                    if self.ptr >= D_BUFFERSIZE:
                        svm_result = self.click_detector_svm(self.d_buffer)
                        print("svm result for click : ", svm_result)
                        _ , self.overlap = np.array_split(self.d_buffer, 2)
                        self.d_buffer[:D_BUFFERSIZE//2] = self.overlap
                        self.ptr = D_BUFFERSIZE//2

                    # Find center of palm position
                    x_hand, y_hand, frame = self.findHandPosition(rh, frame)
                    x_slot, y_slot = self.handPosToSlotXY(x_hand, y_hand, frame)
                    #print("slot x,y : {},{}".format(x_slot, y_slot))
                    self.slot_num = self.findSlotNumber(x_slot, y_slot)
                    if svm_result == 'double_click':
                        self.click_detected_cnt += 1
                        if self.click_detected_cnt >= 2:
                            selected_slot_num = self.slot_num
                            self.click_detected_cnt = 0

        # return confirmed cursor's position, None -> Not confirm yet
        # and orignal frame or overlayed frame
        return confirm_flag, selected_slot_num, frame
