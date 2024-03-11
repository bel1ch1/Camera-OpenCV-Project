# Определитель маркеров на видео


# подключаем библиотеки
import cv2
import numpy as np

# Метод выполнят поиск маркеров и рисует контуры
def findArucoMarkers(img, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Калибруем
    grayRoi = gray#[:,:] # поправь
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoParam = cv2.aruco.DetectorParameters()
    corners, ids, rejected = cv2.aruco.detectMarkers(grayRoi, arucoDict, parameters = arucoParam)
    if draw:
        return [corners, ids]

if __name__ == "__main__":
    # захват видео с камеры и обработка
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            break
        # Когда сможешь посчитать на GPU
        # ratio = (1920, 1080)
        # imgResize = cv2.resize(img, ratio, interpolation= cv2.INTER_LINEAR)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        # резкость 
        sharpened_image = cv2.filter2D(img, -1, kernel)

        arucofound = findArucoMarkers(sharpened_image)
        if len(arucofound[0])!=0:
            for corner, id in zip(arucofound[0], arucofound[1]):
                print(corner)
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
