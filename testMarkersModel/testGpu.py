
# подключаем библиотеки
import cv2

# Метод выполнят поиск маркеров и рисует контуры
def findArucoMarkers(img, draw=True):
    gray = cv2.cuda.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Калибруем
    grayRoi = gray[120:360,160:480]

    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoParam = cv2.aruco.DetectorParameters()
    corners, ids, rejected = cv2.aruco.detectMarkers(grayRoi, arucoDict, parameters = arucoParam)
    if draw:

       # cv2.aruco.drawDetectedMarkers(img, corners)
        return [corners, ids]

# захват видео с камеры и обработка
cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()

    img_gpu = cv2.cuda_GpuMat()
    img_gpu.upload(img)

    gray_gpu = cv2.cuda.cvtColor(img_gpu, cv2.COLOR_BGR2GRAY)
    gray = gray_gpu.download()

    arucofound = findArucoMarkers(gray)

    if len(arucofound[0]) != 0:
        for corner, id in zip(arucofound[0], arucofound[1]):
            print(corner)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
