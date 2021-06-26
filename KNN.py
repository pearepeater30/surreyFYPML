import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import numpy
import time
import matplotlib.pyplot as plt


list = ['with_mask', 'without_mask']

DATADIR = 'C:/Users/Nathan/Documents/FYPML/dataset'
features_list = []
features_label = []

IMG_HEIGHT = 64
IMG_LENGTH = 64
k_range = (1, 26)
cv_scores = []


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


print(features.shape)
print(labels.shape)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20)

X_train = X_train.reshape(X_train.shape[0], IMG_HEIGHT*IMG_LENGTH)
X_test = X_test.reshape(X_test.shape[0], IMG_HEIGHT*IMG_LENGTH)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

toc = time.perf_counter()
#knn.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=1)
scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
cv_scores.append(scores.mean())
tic = time.perf_counter()
time_taken = tic - toc
cv_scores.append(scores.mean())
# changing to misclassification error
mse = [1 - x for x in cv_scores]

# determining best k
optimal_k = k_range[mse.index(min(mse))]
print("Score for KNN is: " + str(scores.mean()) + "for neighbor: " + str(1))
print("The optimal number of neighbors is {}".format(optimal_k))

print("Time taken is: " + str(time_taken) + " seconds")
