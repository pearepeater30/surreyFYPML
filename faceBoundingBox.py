import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# dataset/with_mask/00006_Mask.jpg
IMG_HEIGHT = 84
IMG_LENGTH = 84
image_path = "webcamImages/mask-tips-ft-765x310.jpg"

im = cv2.imread(image_path)
mask_model = tf.keras.models.load_model('saved_model/my_model')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
color = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
faces = faceCascade.detectMultiScale(color,1.1,4)
print(len(faces))
for x,y,w,h in faces:
    modified_y = y + 32
    modified_x = x - 32
    print(x, y, w, h)
    # roi_gray = grey[modified_y - 100:modified_y + h + 64, modified_x - 16:modified_x + w + 64]
    roi_color = color[modified_y - 100:modified_y + h + 64, modified_x - 16:modified_x + w + 64]
    resized = cv2.resize(roi_color, (IMG_LENGTH, IMG_HEIGHT))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, IMG_LENGTH, IMG_HEIGHT, 3))
    plt.imshow(roi_color[...,::-1])
    plt.show()
    result = mask_model.predict(reshaped)
    print(result)
    classes = np.argmax(result, axis=1)
    print(classes)

