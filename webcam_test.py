import cv2
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  480)

print("Opened:", cap.isOpened())

while True:
    ret, frame = cap.read()
    print(frame.shape)

    if ret:
        cv2.imshow("Webcam", frame)

        time.sleep(0.01)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()