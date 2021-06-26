import cv2
import numpy as np
import tensorflow as tf

mask_model = tf.keras.models.load_model('saved_model/my_model')

mask_model.summary()