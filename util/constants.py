DATA_PATH = "labels.csv"

SIFT_KEYPOINT_INFO_PATH = "data/keypoint_info.h5"
SIFT_KEYPOINT_PATH = "data/keypoint.h5"
SIFT_DESCRIPTOR_PATH = "data/descriptor.h5"

EXCLUDED_KEYPOINT_INFO_PATH = "data/excluded_keypoint_info.h5"
EXCLUDED_KEYPOINT_PATH = "data/excluded_keypoint.h5"
EXCLUDED_DESCRIPTOR_PATH = "data/excluded_descriptor.h5"

DESCRIPTOR_LENGTH = 128  # 4 * 4 * 8(논문 기준)
CLASS_NUM = 7

## TODO: Descriptor 전체 개수 알아내는 코드 짜서 미리 전처리하여 상수값 구하기..
## TOTAL_DESCRIPTOR_CNT = calc_descriptor_cnt()
TOTAL_DESCRIPTOR_CNT = 443431  # 실제로 SIFT를 돌려서 알아내야함