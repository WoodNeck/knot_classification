import cv2
import numpy as np
import h5py
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from sklearn.preprocessing import LabelEncoder
from random import choice

DESCRIPTOR_LENGTH = 128  # 4 * 4 * 8(논문 기준)
CLASS_NUM = 7


def extract_feature(img_paths, img_labels):
    summed_descriptors = np.array([])
    descriptors_per_class = np.zeros(shape=(CLASS_NUM, DESCRIPTOR_LENGTH))
    descriptor_counts = np.zeros(CLASS_NUM, "int32")
    images_per_class = np.zeros(CLASS_NUM, "int32")

    # 라벨을 String에서 Integer로 변환
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(img_labels)

    total = len(img_paths)  # 총 이미지 개수
    processed = 0  # 현재 처리한 이미지 개수
    most_descriptors = 0
    most_desc_index = 0
    for i in range(total):
        img_path = img_paths[i]
        current_label = target[i]

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        (keypoints, descriptors) = sift.detectAndCompute(gray, None)

        if descriptors is not None:
            descriptors = descriptors.astype(np.float32, copy=False)
            descriptors = descriptors.reshape(-1, DESCRIPTOR_LENGTH)
            descriptor_cnt = descriptors.shape[0]

            descriptor_counts[current_label] += descriptor_cnt

            descriptor_summed = np.add.reduce(descriptors)
            descriptors_per_class[current_label] += descriptor_summed

            summed_descriptors = np.append(summed_descriptors, descriptor_summed)

            if descriptor_cnt > most_descriptors:
                most_descriptors = descriptor_cnt
                most_desc_index = i
        else:
            zero_descriptor = np.zeros(DESCRIPTOR_LENGTH, "float32")
            summed_descriptors = np.append(summed_descriptors, zero_descriptor)

        images_per_class[current_label] += 1

        processed += 1
        if processed % 1000 == 0:
            print("[SIFT]{}/{}".format(processed, total))

    summed_descriptors = summed_descriptors.reshape(-1, DESCRIPTOR_LENGTH)
    for i in range(CLASS_NUM):
        descriptors_per_class[i] = descriptors_per_class[i] / descriptor_counts[i]

    # Feature Extracted된 저장
    h5f_data = h5py.File('data/data.h5', 'w')
    h5f_data.create_dataset('dataset_1', data=summed_descriptors)

    h5f_label = h5py.File('data/labels.h5', 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))

    h5f_data.close()
    h5f_label.close()

    # k-means 수행
    print("Starting k-means clustering...")
    k = 5
    kmeans = KMeans(n_clusters=k).fit(summed_descriptors)
    descriptor_per_cluster = np.zeros(k, "int32")

    for descriptor in summed_descriptors:
        descriptor = descriptor.reshape(1, 128)
        descriptor_per_cluster[kmeans.predict(descriptor)] += 1
    print("k-means clustering done...")

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA

    fig, ax = plt.subplots(figsize=(10,4))
    plt.title('Descriptor counts')
    x = np.arange(CLASS_NUM)
    ax.bar(x, descriptor_counts)
    ax.set_xticks(x)
    ax.set_xticklabels(label_encoder.classes_)
    plt.show()

    fig, ax = plt.subplots(figsize=(10,4))
    plt.title('Images per class')
    x = np.arange(CLASS_NUM)
    ax.bar(x, images_per_class)
    ax.set_xticks(x)
    ax.set_xticklabels(label_encoder.classes_)
    plt.show()

    fig, ax = plt.subplots(figsize=(10,10))
    plt.title('Average descriptor per class')
    x = np.arange(DESCRIPTOR_LENGTH)
    for i in range(CLASS_NUM):
        plt.plot(x, descriptors_per_class[i])
    plt.legend(label_encoder.classes_, loc='upper left')
    plt.show()

    fig, ax = plt.subplots(figsize=(10,4))
    plt.title('Descriptor per k-means cluster')
    x = np.arange(k)
    ax.bar(x, descriptor_per_cluster)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    plt.show()

    # k-means데이터 차원축소
    fig, ax = plt.subplots(figsize=(10,10))
    pca = PCA(n_components=2).fit(summed_descriptors)
    shrinked_descriptors = pca.transform(summed_descriptors)
    shrinked_cluster_centers = pca.transform(kmeans.cluster_centers_)
    sns.scatterplot(x=shrinked_descriptors[:, 0], y=shrinked_descriptors[:, 1], hue=img_labels)
    voc_labels = np.arange(k)
    sns.scatterplot(x=shrinked_cluster_centers[:, 0], y=shrinked_cluster_centers[:, 0], hue=voc_labels)
    plt.show()

    # descriptor type classification
    # img_path = img_paths[most_desc_index]
    img_path = choice(img_paths)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Descriptor on img, each for kmeans classified types
    sift = cv2.xfeatures2d.SIFT_create()
    (keypoints, descriptors) = sift.detectAndCompute(gray, None)
    for i in range(k):
        if descriptors is not None:
            filtered_keypoints = []
            for j in range(descriptors.shape[0]):
                keypoint = keypoints[j]
                descriptor = descriptors[j]
                descriptor = descriptor.reshape(1, DESCRIPTOR_LENGTH)
                
                if kmeans.predict(descriptor) == i:
                    filtered_keypoints.append(keypoint)
            kmeans_img = cv2.drawKeypoints(gray, filtered_keypoints, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            kmeans_img = img
        plt.subplot(1, 5, i + 1)
        plt.imshow(kmeans_img)
        plt.xticks([]), plt.yticks([])
        plt.xlabel("Cluster {}".format(i))
    plt.show()

    # Most
    img_path = img_paths[most_desc_index]
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Descriptor on img, each for kmeans classified types
    img_path = img_paths[most_desc_index]
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Descriptor on img, each for kmeans classified types
    sift = cv2.xfeatures2d.SIFT_create()
    (keypoints, descriptors) = sift.detectAndCompute(gray, None)
    for i in range(k):
        if descriptors is not None:
            filtered_keypoints = []
            for j in range(descriptors.shape[0]):
                keypoint = keypoints[j]
                descriptor = descriptors[j]
                descriptor = descriptor.reshape(1, DESCRIPTOR_LENGTH)
                
                if kmeans.predict(descriptor) == i:
                    filtered_keypoints.append(keypoint)
            kmeans_img = cv2.drawKeypoints(gray, filtered_keypoints, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            kmeans_img = img
        plt.subplot(1, 5, i + 1)
        plt.imshow(kmeans_img)
        plt.xticks([]), plt.yticks([])
        plt.xlabel("Cluster {}".format(i))
    plt.show()
