import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import datetime
import time
import pickle

IMG_HEIGHT = 84
IMG_LENGTH = 84


pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

pickle_in = open("X_test.pickle", "rb")
X_test = pickle.load(pickle_in)

pickle_in = open("y_test.pickle", "rb")
y_test = pickle.load(pickle_in)

model = MobileNet(weights='imagenet', include_top=False, input_shape=(IMG_LENGTH, IMG_HEIGHT, 3))

base_input = model.layers[0].input

base_output = model.layers[-4].output

Flat_layer = layers.Flatten()(base_output)
final_output = layers.Dense(2)(Flat_layer)
final_output = layers.Activation('sigmoid')(final_output)


new_model = keras.Model(inputs = base_input, outputs = final_output)

new_model.summary()

log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

new_model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

toc = time.perf_counter()
model_history = new_model.fit(X, y, batch_size=16, epochs=4, callbacks=[tensorboard], shuffle=True, validation_split=0.2)
tic = time.perf_counter()
time_taken = tic - toc
print("Time taken is: " + str(time_taken) + " seconds")

test_eval = new_model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

new_model.save('saved_model/my_transfer_model')

