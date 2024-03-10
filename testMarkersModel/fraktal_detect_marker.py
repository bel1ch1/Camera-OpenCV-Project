import cv2


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Пример кода для поиска фрактальных маркеров с использованием алгоритма FAST
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(gray, None)

    # Отрисовка фрактальных маркеров на изображении
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(255,0,0))

    cv2.imshow('Fractal Markers', frame_with_keypoints)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
