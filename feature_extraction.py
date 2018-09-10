import cv2
import numpy as np
import h5py
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from sklearn.preprocessing import LabelEncoder

DESCRIPTOR_LENGTH = 128  # 4 * 4 * 8(논문 기준)


def extract_feature(img_paths, img_labels):
    total_descriptors = np.array([])

    total = len(img_paths)  # 총 이미지 개수
    processed = 0  # 현재 처리한 이미지 개수
    for img_path in img_paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        (keypoints, descriptors) = sift.detectAndCompute(gray, None)

        if descriptors is not None:
            descriptors = descriptors.astype(np.float32, copy=False)
            descriptors = descriptors.reshape(-1, DESCRIPTOR_LENGTH)
            descriptor_summed = np.add.reduce(descriptors)
            total_descriptors = np.append(total_descriptors, descriptor_summed)
        else:
            zero_descriptor = np.zeros(DESCRIPTOR_LENGTH, "float32")
            total_descriptors = np.append(total_descriptors, zero_descriptor)

        processed += 1
        if processed % 100 == 0:
            print("[SIFT]{}/{}".format(processed, total))

    total_descriptors = total_descriptors.reshape(-1, DESCRIPTOR_LENGTH)
    print(total_descriptors.shape)

    # 라벨을 String에서 Integer로 변환
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(img_labels)

    # Feature Extracted된 저장
    h5f_data = h5py.File('data/data.h5', 'w')
    h5f_data.create_dataset('dataset_1', data=total_descriptors)

    h5f_label = h5py.File('data/labels.h5', 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))

    h5f_data.close()
    h5f_label.close()
