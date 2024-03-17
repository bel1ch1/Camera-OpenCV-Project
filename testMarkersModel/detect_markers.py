# Определитель маркеров на видео

# подключаем библиотеки
import cv2

# Метод выполнят поиск маркеров и рисует контуры
def findArucoMarkers(img, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #negative = cv2.bitwise_not(img) # вариант использования со специальными маркерами
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoParam = cv2.aruco.DetectorParameters()
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
    if draw:
        cv2.aruco.drawDetectedMarkers(img, corners)
    return [corners, ids]

# захват видео с камеры
cap = cv2.VideoCapture('rtsp://admin:Camera12@192.168.0.10:8000')
#cap = cv2.VideoCapture(0)
# Обработка видео
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
