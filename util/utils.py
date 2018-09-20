def load_original_datas(data_path):
    """데이터와 라벨을 경로로부터 로드하여 리턴

    Args:
        data_path (str): "파일명 라벨" 순으로 데이터가 정리된 bsv파일의 경로
    Returns:
        image_paths (list<str>): 개별 이미지의 경로가 담긴 리스트
        encoded_labels (list<int>): 개별 이미지의 0~n으로 인코딩된 라벨이 담긴 리스트
        label_classes (list<str>): 0~n으로 인코딩된 라벨 각각의 클래스 이름이 담긴 리스트
    Example:
        image_paths, encoded_labels, label_classes = load_original_datas()
    """

    import csv
    from sklearn.preprocessing import LabelEncoder

    img_paths = []
    img_labels = []
    with open(data_path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            img_path = row[0]
            img_label = row[1]

            img_paths.append(img_path)
            img_labels.append(img_label)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(img_labels)
    label_classes = label_encoder.classes_

    return img_paths, encoded_labels, label_classes

def timer(name):
    """함수 시간 측정용 데코레이터

    Args:
        name [Optional](str): 시작, 종료시 표기할 함수의 이름
    Returns:
        None
    Example:
        @timer("SIFT")
        def sift():
            # ...
    """
    def timer_decorator(orig_func):
        import time

        func_name = name if name else orig_func.__name__
        def wrapper_function(*args, **kwargs):
            print("[{}]-----Begin-----".format(func_name)) if func_name else None

            time_start = time.time()
            orig_func(*args, **kwargs)
            time_end = time.time()

            time_diff = time_end - time_start

            print("[{}]-----End-----".format(func_name)) if func_name else None
            print("[{}]Done in {}s".format(func_name, time_diff))

        return wrapper_function
    return timer_decorator
