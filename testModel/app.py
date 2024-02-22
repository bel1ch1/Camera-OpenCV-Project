# объявление библиотек
import numpy as np
import cv2
import subprocess as sp
import multiprocessing as mp
from os import remove


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while(True):
  ret, frame = cap.read()
  frame = cv2.rectangle(frame,(384,0),(510,128),(0,0,255),10)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (7, 7), 1.5) # Параметры позволяют регулировать шумность

  edges = cv2.Canny(gray, 1, 50) # Нахождение контуров
  cv2.imshow("edges",edges) # вывод  на экран # оригинальный вариант


cap.release()
cv2.destroyAllWindows()
