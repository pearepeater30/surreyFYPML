import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
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

model = Sequential()

#First Layer
model.add(Conv2D(input_shape=(IMG_LENGTH, IMG_HEIGHT, 3), filters=8, kernel_size=16, strides=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

#Second Layer
model.add(Conv2D(filters=8, kernel_size=3, strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Third Layer
model.add(Conv2D(filters=8, kernel_size=3, strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Flatten the convolutional data to feed to the fully-connected layer
model.add(Flatten())

#Fully Connected Layer
model.add(Dense(units=100, activation='relu'))

#Output Layer
model.add(Dense(units=2, activation='softmax'))

model.summary()

log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)


model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

toc = time.perf_counter()

model_history = model.fit(X, y, batch_size=16, epochs=4, callbacks=[tensorboard], shuffle=True, validation_split=0.2)

tic = time.perf_counter()
time_taken = tic - toc
print("Time taken is: " + str(time_taken) + " seconds")

test_eval = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

#model.save('saved_model/my_model')