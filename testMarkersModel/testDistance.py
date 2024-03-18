import cv2
import pickle


# Подгружаем данные для определения дистанции
with open('dist.pkl', 'rb') as f:
    dist = pickle.load(f)

with open('cameraMatrix.pkl', 'rb') as g:
    mtx = pickle.load(g)



# Метод выполнят поиск маркеров и рисует контуры
def findArucoMarkers(img, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoParam = cv2.aruco.DetectorParameters()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)
    if draw:
        cv2.aruco.drawDetectedMarkers(img, corners)
    return [corners]


# захват видео с камеры и обработка
cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()
    arucofound = findArucoMarkers(img)
    if len(arucofound[0])!=0:
        for corner in arucofound[0]:
            print(corner[0][0], corner[0][1], corner[0][2], corner[0][3])
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
