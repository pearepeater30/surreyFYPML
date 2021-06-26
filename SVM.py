from sklearn import svm
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import numpy
import time

list = ['with_mask', 'without_mask']

DATADIR = 'C:/Users/Nathan/Documents/FYPML/dataset'
features_list = []
features_label = []

IMG_HEIGHT = 64
IMG_LENGTH = 64


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

X_train = X_train.reshape(X_train.shape[0], IMG_HEIGHT*IMG_LENGTH)
X_test = X_test.reshape(X_test.shape[0], IMG_HEIGHT*IMG_LENGTH)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

for kernel in ('linear', 'poly', 'rbf'):
    clf = svm.SVC(kernel=kernel)
    toc = time.perf_counter()
    clf.fit(X_train, y_train)
    tic = time.perf_counter()
    time_taken = tic - toc
    score = clf.score(X_test, y_test)
    print("The score predicted is for " + kernel + " kernel is: " + str(score))
    print("Time taken for " + kernel + " kernel" + " is: " + str(time_taken) + " seconds")