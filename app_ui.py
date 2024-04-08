import argparse
import sys
import json
import cv2
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from qt.item_array import Ui_ItemArray
import redis



# redis channels for tags
ch_slot = 'handgesture.slot_num'
ch_selected_slot = 'handgesture.selected_slot_num'
ch_state = 'handgesture.state'



class HandGestureControl_UI():
    def __init__(self, size_x, size_y):
        self.form = QWidget()
        self.ui = Ui_ItemArray()
        self.ui.setupUi(self.form)
        self.pix_nonselect = QPixmap("qt/assets/pixel-white.jpg")
        self.pix_cursor = QPixmap("qt/assets/pixel-black.jpg")
        self.pix_selected = QPixmap("qt/assets/pixel-green.jpg")
        self.size_x = size_x
        self.size_y = size_y
        self.slot_previous = 0
        self.selected_slot_previous = 0

        self.ui.lcdNumber_x.display("0")
        self.ui.lcdNumber_y.display("0")

        # redis client
        self.r = redis.Redis(host='localhost', port=6379)

        # Create a timer object
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_pos)
        self.timer.setInterval(5)  # 1 second interval
        self.timer.start()

    def run(self):
        self.draw_item_pos(1,1)
        self.ui.lcdNumber_x.display("0")
        self.ui.lcdNumber_y.display("0")
        self.timer.start()
        self.form.show()
        sys.exit(app.exec())

    def coordinate_conversion(self, coor):
        # UI's label index (1,2,3,...) into (x, y)
        if isinstance(coor, int):
            x = int(0)
            y = int(0)
            coor -= 1
            y = int(coor / self.size_x)
            x = coor % self.size_x
            return (x, y)

        # (x, y) into UI's label index (1,2,3,...)
        elif isinstance(coor, tuple):
            x, y = coor
            index = (self.size_x * y) + x + 1
            return index

        return None

    def draw_item_pos(self, cursor_num, selected_num):
        attributes = dir(self.ui)
        for attr in attributes:
            if attr.startswith("label_") and isinstance(getattr(self.ui, attr), QLabel):
                label = getattr(self.ui, attr)
                num_label = int(attr.split('label_')[1])
                if num_label == cursor_num and num_label != selected_num:
                    label.setPixmap(self.pix_cursor)
                elif num_label == selected_num:
                    label.setPixmap(self.pix_selected)
                else:
                    label.setPixmap(self.pix_nonselect)

    def update_pos(self):
        slot_num = int(self.r.get(ch_slot))
        selected_slot_num = int(self.r.get(ch_selected_slot))
        if slot_num != self.slot_previous or selected_slot_num != self.selected_slot_previous:
            (x, y) = self.coordinate_conversion(slot_num)
            self.draw_item_pos(slot_num, selected_slot_num)
            self.ui.lcdNumber_x.display(str(x))
            self.ui.lcdNumber_y.display(str(y))
        self.slot_previous = slot_num
        self.selected_slot_previous = selected_slot_num



if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = HandGestureControl_UI(5, 5)
    w.run()

