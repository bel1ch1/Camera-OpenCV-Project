import cv2

cap = cv2.VideoCapture(
    'rtsp://admin:Camera12@192.168.0.10:554/1'
    )

while True:

    ret, frame = cap.read()

    cv2.imshow('Processed Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
