import cv2
import pickle
import numpy as np


# Подгружаем данные для определения дистанции
with open('dist.pkl', 'rb') as f:
    dist_coef = pickle.load(f)

with open('cameraMatrix.pkl', 'rb') as g:
    cam_mat = pickle.load(g)


# Константы
MARKER_SIZE_CM = 16.5 # Размер маркера
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) # набор маркеров

# Параметры
arucoParam = cv2.aruco.DetectorParameters()


# Поиск позиции
def pose_esitmation(img, arucoDict):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Ищем маркеры
    corners, ids, _ = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)
    # если нашли
    if corners:
        # Определяем сдвиг
        _ , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_CM, cam_mat, dist_coef)
        # Определяем дистанцию до маркера
        distance_to_marker = np.sqrt(
                        tvec ** 2 + tvec ** 2 + tvec ** 2
                    )
        # Вывод в консоль
        for corner in corners:
            print(corner[0][0], corner[0][1], corner[0][2], corner[0][3])
        print(distance_to_marker)
    return 0


# захват видео с камеры и обработка
cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    if not success:
        continue

    pose_esitmation(frame, ARUCO_DICT)

    cv2.imshow("distance", frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
