import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

IMG_HEIGHT = 84
IMG_LENGTH = 84

capture = cv2.VideoCapture(0)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
mask_model = tf.keras.models.load_model('my_model')
faces_detected = 0
masks_detected = 0

mask_model.summary()

while True:
    ret, frame = capture.read()
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(frame, 1.1, 4)
    faces_detected += len(faces)
    for x, y, w, h in faces: 
        roi_color = frame[y:y+h, x:x+w]
        resized = cv2.resize(roi_color, (IMG_LENGTH, IMG_HEIGHT))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, IMG_LENGTH, IMG_HEIGHT, 3))
        result = mask_model.predict(reshaped)
        print(result)
        classes = np.argmax(result, axis=1)
        print(classes)
        if (classes[0] == 0):
            print("Wearing Mask")
            masks_detected += 1
        else:
            print("Not Wearing Mask")
        data = str(faces_detected) + ' , ' + str(masks_detected)
        print(data)
        time.sleep(1.5)

    cv2.imshow('webcam', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break