import cv2

# Метод выполнят поиск маркеров и рисует контуры
def findArucoMarkers(img, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoParam = cv2.aruco.DetectorParameters()

    markerSizeCM = 16.5
    rvec , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerSizeCM, mtx, dist)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
    if draw:
        cv2.aruco.drawDetectedMarkers(img, corners)
    return [corners, tvec]


# захват видео с камеры и обработка
cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()
    arucofound = findArucoMarkers(img)
    if len(arucofound[0])!=0:
        for corner, tvec in zip(arucofound[0], arucofound[1]):
            print(corner[0][0], corner[0][1], corner[0][2], corner[0][3], tvec[2]/100)
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
