import sys
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class CameraWidget(QLabel):
    def __init__(self):
        super().__init__()

        self.camera = cv2.VideoCapture(0)  # Используем камеру по умолчанию

    def update_image(self):
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(image))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Camera Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.camera_widget = CameraWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.camera_widget)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.timer = self.startTimer(1000/30)  # Обновляем изображение каждые 30 кадров в секунду

        def timerEvent(self, event):
            self.camera_widget.update_image()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
