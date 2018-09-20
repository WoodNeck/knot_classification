import cv2
import h5py
import numpy as np
from .constants import *


class SiftData:
    """SIFT를 통해 이미지로부터 feature extracted된 데이터를 관리하는 클래스
    """

    def __init__(self, data_path):
        keypoint_info_path = data_path[0]
        keypoint_path = data_path[1]
        descriptor_path = data_path[2]

        keypoint_info_str = h5py.File(keypoint_info_path, 'r')["keypoint_info"]
        keypoints_str = h5py.File(keypoint_path, 'r')["keypoints"]
        descriptors_str = h5py.File(descriptor_path, 'r')["descriptors"]

        self._keypoint_info = np.array(keypoint_info_str)
        self._keypoints = np.array(keypoints_str)
        self._descriptors = np.array(descriptors_str)

        makeKeypoint = lambda kp: cv2.KeyPoint(x=kp[0],
                                               y=kp[1],
                                               _size=kp[2],
                                               _angle=kp[3],
                                               _response=kp[4],
                                               _octave=kp[5],
                                               _class_id=kp[6])

        self._keypoints = [makeKeypoint(kp) for kp in self._keypoints]

    def keypoints_length_of(self, idx):
        keypoint_idx, keypoint_len = self._keypoint_info[idx]
        return keypoint_len

    def keypoints_of(self, idx):
        keypoint_idx, keypoint_len = self._keypoint_info[idx]
        return self._keypoints[keypoint_idx : keypoint_idx + keypoint_len]
    
    def descriptors_of(self, idx):
        keypoint_idx, keypoint_len = self._keypoint_info[idx]
        return self._descriptors[keypoint_idx : keypoint_idx + keypoint_len]

    @property
    def all_keypoints(self):
        return self._keypoints

    @property
    def all_descriptors(self):
        return self._descriptors

    @property
    def summed_descriptors(self):
        """각 이미지마다의 descriptor의 합계를 구함
        """
        data_length = self._keypoint_info.shape[0]
        summed_descriptors = np.ndarray(shape=(data_length, DESCRIPTOR_LENGTH))

        for i in range(data_length):
            keypoint_idx, keypoint_len = self._keypoint_info[i]
            current_descriptors = self._descriptors[keypoint_idx : keypoint_idx + keypoint_len]
            summed_descriptors[i][:] = np.add.reduce(current_descriptors)
    
        return summed_descriptors
