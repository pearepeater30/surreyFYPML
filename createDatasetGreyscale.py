import pickle
import numpy as np
import os
import cv2
import numpy

list = ['with_mask', 'without_mask']

DATADIR = 'C:/Users/Nathan/Documents/FYPML/dataset'
TEST_DATADIR = 'C:/Users/Nathan/Documents/FYPML/og_dataset'

IMG_HEIGHT = 84
IMG_LENGTH = 84

def create_training_data():
    features_list = []
    features_label = []
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

    features = np.array(features_list)
    labels = np.array(features_label)

    features = features / 255.0

    print(features.shape)
    print(labels.shape)

    X_train = features.reshape(features.shape[0], IMG_HEIGHT, IMG_LENGTH, 1)

    pickle_out = open("X.pickle", 'wb')
    pickle.dump(X_train, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", 'wb')
    pickle.dump(labels, pickle_out)
    pickle_out.close()

def create_separate_testing_data():
    test_features_list = []
    test_features_label = []
    for entry in list:
        print(entry)
        path = os.path.join(TEST_DATADIR, entry)
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
                test_features_list.append(new_array)
                test_features_label.append(class_num)
            except Exception as e:
                pass

    X_test_features = np.array(test_features_list)
    y_test_labels = np.array(test_features_label)

    test_features = X_test_features / 255.0

    print(X_test_features.shape)

    X_separate_test = test_features.reshape(test_features.shape[0], IMG_HEIGHT, IMG_LENGTH, 1)

    pickle_out = open("X_test.pickle", "wb")
    pickle.dump(X_separate_test, pickle_out)
    pickle_out.close()

    pickle_out = open("y_test.pickle", "wb")
    pickle.dump(y_test_labels, pickle_out)
    pickle_out.close()


create_training_data()
create_separate_testing_data()














