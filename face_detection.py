from ctypes.macholib import framework

import cv2
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0

while True:
    ret, frame = cap.read()

    frame_count += 1 

    if not ret:
        continue

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )
    faces = []
    if frame_count % 3 == 0:
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor =1.05,
            minNeighbors = 8,
            minSize = (80,80)
        )

    if len(faces) > 0:
        faces = sorted(
            faces,
            key=lambda x: x[2] * x[3],
            reverse=True
        )

        x, y, w, h  = faces[0]

        cv2.rectangle(
            frame,
            (x,y),
            (x+w, y+h),
            (0,255,0),
            2
        )

        cv2.imshow(
            "Face Detection",
            frame
        )

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:
            break


cap.release()
cv2.destroyAllWindows()