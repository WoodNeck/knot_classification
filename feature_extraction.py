import cv2
import numpy as np
import h5py
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def extract_feature(img_paths, img_labels):
    total_descriptors = np.array([])
    descriptor_lengths = []
    train_labels = []

    total = len(img_paths)
    processed = 0
    for i in range(total):
        img_path = img_paths[i]
        label = img_labels[i]
        train_labels.append(label)

        img = cv2.imread(img_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()

        (keypoints, descriptors) = sift.detectAndCompute(gray, None)
        if descriptors is not None:
            descriptors = descriptors.astype(np.float32, copy=False)
            total_descriptors = np.append(total_descriptors, descriptors)
            descriptor_lengths.append(descriptors.shape[0])
        else:
            descriptor_lengths.append(0)
        processed += 1
        if processed % 20 == 0:
            print("[SIFT]{}/{}".format(processed, total))

    total_descriptors = total_descriptors.reshape(-1, 128)
    print(total_descriptors.shape)

    # Perform k-means clustering
    print("Starting k-means")
    k = 500 # 데이터의 개수에 비례하게 설정
    kmeans = KMeans(n_clusters=k, verbose=1).fit(total_descriptors)
    print("k-means clustering done...")

    # Calculate the histogram of features
    index = 0
    feature_vectors = np.array([])
    for desc_length in descriptor_lengths:
        predicted = np.zeros(k, "float32")
        for i in range(desc_length):
            descriptor = total_descriptors[index]
            descriptor = descriptor.reshape(1, -1)
            predicted[kmeans.predict(descriptor)[0]] += 1
            index += 1
        feature_vectors = np.append(feature_vectors, predicted)
    feature_vectors = feature_vectors.reshape(-1, k)
    print(feature_vectors.shape)

    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(train_labels)

    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(feature_vectors)

    h5f_data = h5py.File('data/data.h5', 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

    h5f_label = h5py.File('data/labels.h5', 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))

    h5f_data.close()
    h5f_label.close()
