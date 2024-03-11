import sys
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import uic

app = QtWidgets.QApplication(sys.argv)
window = uic.loadUi("untitled.ui")
window.btnQuit.clicked.connect(app.quit)
window.show()
sys.exit(app.exec())
