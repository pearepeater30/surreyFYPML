import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import numpy
import time
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint

list = ['with_mask', 'without_mask']

DATADIR = 'C:/Users/Nathan/Documents/FYPML/dataset'
features_list = []
features_label = []

IMG_HEIGHT = 84
IMG_LENGTH = 84


def create_training_data():
    for entry in list:
        print(entry)
        path = os.path.join(DATADIR, entry)
        class_num = list.index(entry)
        print(path)
        for img in os.listdir(path):
            try:
                jointpath = os.path.join(path, img)
                stream = open(jointpath, 'rb')
                bytes = bytearray(stream.read())
                numpyarray = numpy.asarray(bytes, dtype=numpy.uint8)
                # imread doesn't support unicode characters
                img_array = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_LENGTH, IMG_HEIGHT))
                features_list.append(new_array)
                features_label.append(class_num)
            except Exception as e:
                pass


create_training_data()

features = np.array(features_list)
labels = np.array(features_label)

features = features/255.0

print(features.shape)
print(labels.shape)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20)

X_train = X_train.reshape(X_train.shape[0], IMG_HEIGHT, IMG_LENGTH, 1)
X_test = X_test.reshape(X_test.shape[0], IMG_HEIGHT, IMG_LENGTH, 1)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

model=Sequential()

model.add(Conv2D(200,(3,3),input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(50,activation='relu'))
#Dense layer of 64 neurons
model.add(Dense(2,activation='softmax'))
#The Final layer with two outputs for two categories

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history=model.fit(X_train,y_train,epochs=4,validation_split=0.2)

test_eval = model.evaluate(X_test, y_test, verbose=0)

model.save('saved_model/my_keras_model')