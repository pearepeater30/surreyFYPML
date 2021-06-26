import cv2
import numpy as np
import tensorflow as tf

IMG_HEIGHT = 84
IMG_LENGTH = 84
list = ['with_mask', 'without_mask']


mask_model = tf.keras.models.load_model('saved_model/my_transfer_model')

image_path = 'webcamImages/pycharm64_gEUkiVsqb4.png'

im = cv2.imread(image_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) #needs cv2.COLOR_BGR2GRAY if using grayscale model
im = np.array(im)
im = im/255.0
im = cv2.resize(im, (IMG_HEIGHT, IMG_LENGTH))

print(im.shape)

im = tf.expand_dims(im, axis=0)
# im = tf.expand_dims(im, axis=-1) Needed only if using grayscale model

print(im.shape)

predictions = mask_model.predict(im)

print(predictions)

classes = np.argmax(predictions, axis = 1)
print(classes)