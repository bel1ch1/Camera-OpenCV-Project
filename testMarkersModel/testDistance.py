import cv2
import pickle
import numpy as np
import streamlit as st

# Подгружаем данные для определения дистанции
with open('dist.pkl', 'rb') as f:
    dist_coef = pickle.load(f)

with open('cameraMatrix.pkl', 'rb') as g:
    cam_mat = pickle.load(g)


# Константы
MARKER_SIZE_M = 0.165 # Размер маркера
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) # набор маркеров

# Параметры
arucoParam = cv2.aruco.DetectorParameters()


# Поиск позиции
def pose_esitmation(img, arucoDict):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Ищем маркеры
    corners, _, _ = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)
    # если нашли
    if corners:
        # Определяем сдвиг
        _ , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_M, cam_mat, dist_coef)
        # Определяем дистанцию до маркера
        # distance_to_marker = np.sqrt(
        #                 tvec ** 2 + tvec ** 2 + tvec ** 2
        #             )

        distance_to_marker = np.linalg.norm(tvec)

        # координаты и дистанция
        c1 = corners[0][0][0]
        c2 = corners[0][0][1]
        c3 = corners[0][0][2]
        c4 = corners[0][0][3]
        d = distance_to_marker#[0][0]
        # Вывод в консоль
        print(c1, c2, c3, c4, d)
        return f"1 {c1[:1]},\n2 {c2[:1]},\n3 {c3[:1]},\n4 {c4[:1]},\ndistance {d}"


# Составляющие веб-страницы
# main
st.title("corners coords distasnce")
col1, col2 = st.columns(2)
with col1:
    frame_placeholder = st.empty()
with col2:
    pose_estmition_data_area = st.empty()
start_btn_pressed = st.button("Start")
stop_btn_pressed = st.button("Stop")
# sidebar
with st.sidebar:
    st.button("Log In")
    st.button("Register")



# захват видео с камеры и обработка
if start_btn_pressed:
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and not stop_btn_pressed:

        success, frame = cap.read()
        if not success:
            break

        # Вывод видео
        frame_placeholder.image(frame, channels="BGR")
        # Вывод координат и дистанции
        output=pose_esitmation(frame, ARUCO_DICT)
        pose_estmition_data_area.text(f"data\n{output}")

        # Остановка вывода
        k = cv2.waitKey(30) & 0xff
        if k == 27 or stop_btn_pressed:
            break


    cap.release()
    cv2.destroyAllWindows()
