import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_HEIGHT = 84
IMG_LENGTH = 84

capture = cv2.VideoCapture(0)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
mask_model = tf.keras.models.load_model('saved_model/my_model')

mask_model.summary()

while True:
    ret, frame = capture.read()
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(frame, 1.1, 4)
    for x, y, w, h in faces:
        modified_y = y + 32
        modified_x = x - 32
        print(x, y, w, h)
        # roi_gray = grey[modified_y - 100:modified_y + h + 64, modified_x - 16:modified_x + w + 64]
        roi_color = frame[modified_y - 100:modified_y + h + 64, modified_x - 16:modified_x + w + 64]
        resized = cv2.resize(roi_color, (IMG_LENGTH, IMG_HEIGHT))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, IMG_LENGTH, IMG_HEIGHT, 3))
        # plt.imshow(roi_color[...,::-1])
        # plt.show()
        result = mask_model.predict(reshaped)
        print(result)
        classes = np.argmax(result, axis=1)
        print(classes)

    cv2.imshow('object detection', cv2.resize(frame, (800,600)))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break