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
ch_pos_x = 'handgesture.pos_x'
ch_pos_y = 'handgesture.pos_y'
ch_pos_confirm_x = 'handgesture.pos_confirm_x'
ch_pos_confirm_y = 'handgesture.pos_confirm_y'
ch_state = 'handgesture.state'



class HandGestureControl_UI():
    def __init__(self, size_x, size_y):
        self.form = QWidget()
        self.ui = Ui_ItemArray()
        self.ui.setupUi(self.form)
        self.pix_nonselect = QPixmap("qt/assets/pixel-black.jpg")
        self.pix_select = QPixmap("qt/assets/pixel-white.jpg")
        self.pix_confirm = QPixmap("qt/assets/pixel-white.jpg")
        self.size_x = size_x
        self.size_y = size_y

        self.ui.lcdNumber_x.display("0")
        self.ui.lcdNumber_y.display("0")

        # redis client
        self.r = redis.Redis(host='localhost', port=6379)

        # Create a timer object
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_pos)
        self.timer.setInterval(250)  # 1 second interval
        self.timer.start()

    def run(self):
        self.draw_item_pos(1)
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
            y = coor / self.size_x
            x = coor % self.size_x
            return (x, y)

        # (x, y) into UI's label index (1,2,3,...)
        elif isinstance(coor, tuple):
            x, y = coor
            index = (self.size_x * y) + x + 1
            return index

        return None

    def draw_item_pos(self, input_index):
        attributes = dir(self.ui)
        for attr in attributes:
            if attr.startswith("label_") and isinstance(getattr(self.ui, attr), QLabel):
                label = getattr(self.ui, attr)
                num_label = int(attr.split('label_')[1])
                if input_index == num_label:
                    label.setPixmap(self.pix_select)
                else:
                    label.setPixmap(self.pix_nonselect)

    def update_pos(self):
        x = int(self.r.get(ch_pos_x))
        y = int(self.r.get(ch_pos_y))
        index = self.coordinate_conversion((x, y))
        print("index from REDIS : ", index)
        self.draw_item_pos(index)
        self.ui.lcdNumber_x.display(str(x))
        self.ui.lcdNumber_y.display(str(y))




if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = HandGestureControl_UI(5, 5)
    w.run()

