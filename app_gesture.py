import argparse
import sys
import json
import cv2
from gesture_controller import GestureController
import threading
import time
import redis
import queue



# Queue
q_slot = queue.Queue()
q_slot_confirm = queue.Queue()
istaskrun_cv = False
istaskrun_redis = False



def task_cv(cam_url, hand_model_path, click_model_path):

    # Hand Gesture Ctrl
    g = GestureController(5, 5, logging=False)
    g.config(hand_model_path, click_model_path)

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
        confirm_slot, frame_display = g.step(frame)

        # Put event and item position to REDIS
        if confirm_slot:
            q_slot_confirm.put(confirm_slot)
        if g.state == 'SELECT_ITEM':
            q_slot.put(g.slot_num)

        # display the result
        frame_display = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        cv2.imshow('MediaPipe Hands', frame_display)
        if cv2.waitKey(50) & 0xFF == 27:
            break
    cap.release()
    istaskrun_cv = False




def task_redis():
    # redis client
    r = redis.Redis(host='localhost', port=6379)
    # redis channels for tags
    ch_slot = 'handgesture.slot_num'
    ch_slot_confirm = 'handgesture.confirm_slot_num'
    ch_state = 'handgesture.state'

    while istaskrun_cv:
        # tag for cursor position
        if not q_slot.empty():
            val = q_slot.get()
            r.set(ch_slot, val)

        if not q_slot_confirm.empty():
            val = q_slot_confirm.get()
            r.set(ch_slot_confirm, val)

        time.sleep(0.1)




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
    hand_model_path = data['model']['gesture']
    click_model_path = data['model']['click']
    f.close()

    # Run CV task
    istaskrun_cv = True
    istaskrun_redis = True
    t1 = threading.Thread(target=task_cv, args=(cam, hand_model_path, click_model_path))
    t2 = threading.Thread(target=task_redis)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


