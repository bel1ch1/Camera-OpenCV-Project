# Определитель маркеров на видео


# подключаем библиотеки
import cv2
import numpy as np
import os


# Метод выполнят поиск маркеров и рисует контуры
def findArucoMarkers(img, markerSize = 5, totalMarkers=50, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(cv2.aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = cv2.aruco.getPredefinedDictionary(key)
    arucoParam = cv2.aruco.DetectorParameters()
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
    # print(ids)
    if draw:
        cv2.aruco.drawDetectedMarkers(img, corners)
    return [corners, ids]

# захват видео с камеры и обработка
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    arucofound = findArucoMarkers(img)
    if len(arucofound[0])!=0:
        for corner, id in zip(arucofound[0], arucofound[1]):
            print(corner)
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
