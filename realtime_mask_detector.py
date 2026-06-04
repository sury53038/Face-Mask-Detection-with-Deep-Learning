import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('face_mask_detector_finetuned.h5')

face_detection = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while True:
    ret,frame = cap.read()

    if not ret:
        continue
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detection.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors = 8,
        minSize = (80,80)
    )

    for(x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        face = cv2.resize(
            face,
            (128,128)
        )

        face = face.astype("float32")/255.0

        face = np.expand_dims(
            face, 
            axis = 0
        )

        prediction = model.predict(
            face,
            verbose=0
        )[0][0]

        print(prediction)

        if prediction >= 0.75:
            label = "MASK"
            color = (0,255,0)

            confidence = prediction * 100

        else:
            label = "NO MASK"
            color = (0,0,255)

            confidence = (1-prediction) * 100

        text = f"{label} : {confidence:.2f}%"


        cv2.rectangle(
            frame,
            (x,y),
            (x+w, y+h),
            color,
            2
        )

        cv2.putText(
            frame,
            text,
            (x,y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow(
        "Face Mask Detection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()