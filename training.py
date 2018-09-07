import h5py
import numpy as np
import os
import glob
import cv2
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

def train(datas, output_dir):
    h5f_data = h5py.File('{}/data.h5'.format(output_dir), 'r')
    h5f_label = h5py.File('{}/labels.h5'.format(output_dir), 'r')

    features_string = h5f_data['dataset_1']
    labels_string = h5f_label['dataset_1']

    features = np.array(features_string)
    labels = np.array(labels_string)

    # Shuffle Data
    seed = 1234
    np.random.seed(seed)
    np.random.shuffle(features)
    np.random.seed(seed)
    np.random.shuffle(labels)

    h5f_data.close()
    h5f_label.close()

    train_ratio = 0.8
    train_set_cnt = int(features.shape[0] * train_ratio)
    test_set_cnt = features.shape[0] - train_set_cnt
    train_set, test_set = np.split(features, [train_set_cnt])
    train_labels, test_labels = np.split(labels, [train_set_cnt])

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(train_set, train_labels)

    label_datas = [data[1] for data in datas]
    label_names = LabelEncoder().fit(label_datas).classes_

    def print_training(data_set, label_set, label_names, model):
        correct = 0
        total = len(data_set)
        for i in range(total):
            feature = data_set[i].reshape(1, -1)
            predicted = model.predict(feature)[0]
            label = label_set[i]
            if predicted == label:
                correct += 1
        print("{}/{}({:.4f}%)".format(correct, total, 100 * correct / total))

    print("[TRAIN SET RESULT]")
    print_training(train_set, train_labels, label_names, model)
    print("[TEST SET RESULT]")
    print_training(test_set, test_labels, label_names, model)
