# kNN(k-Nearest Neighbor)

- 머신러닝 모델 중 직관적이며 간단한 **지도학습** 모델 중 하나
- Lazy Learning
  - 학습에 필요한 데이터를 메모리에 기억만 하고 있다가, 인스턴스가 발생될 때 비로소 일반화 작업을 수행
  - prediction을 수행하기 전까지는 모델을 구성하지 않는다

## 개요

![kNN classification](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/63621/versions/2/screenshot.gif)

- 기본적으로, 새로운 데이터가 어느 그룹에 속하는지 알아보기 위해서 가장 가까이에 있는 k개의 학습 데이터가 속한 그룹을 판별
- 가장 가까이의 있는 학습 데이터들 중 가장 높은 빈도로 나타나는 그룹이 새로운 데이터의 그룹이 된다
- 의사 결정을 위해서, 항상 **홀수**개의 인접한 데이터를 탐색해야 한다

  - 따라서, k는 2n+1의 형태가 된다 (n=0, 1, 2, ...)

- 구현
  ![kNN Circle](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/KnnClassification.svg/220px-KnnClassification.svg.png)

  - 특정 데이터를 기준, 가상의 원을 확장시켜 나가면서 가장 가까운 데이터들을 탐색하는 것이 기본 골조가 된다

  - 이 때, 데이터들을 n개의 특성에 따라 n차원 공간에 배치하여 거리를 탐색해야 하고, 각 데이터마다 정량적 스케일이 상이하기 때문에 데이터에 대한 정규화 과정이 필요하다

  - Distance function

    - 데이터가 연속적인 경우

    ![c_distance](https://www.saedsayad.com/images/KNN_similarity.png)

    - 데이터가 불연속적인 경우

      ![uc_distance](https://www.saedsayad.com/images/KNN_hamming.png)

## 특성

![kNN overfitting](https://elvinouyang.github.io/assets/images/Introduction%20to%20Machine%20Learning%20with%20Python%20-%20Chapter%202%20-%20Datasets%20and%20kNN_files/Introduction%20to%20Machine%20Learning%20with%20Python%20-%20Chapter%202%20-%20Datasets%20and%20kNN_31_1.png)

- kNN의 경우 k의 크기가 작을수록 overfitting 현상이 발생한다

- 반대로, k의 크기가 커질수록 underfitting 현상이 발생한다

- 올바른 k를 찾기 위해, 데이터를 7:3이나 9:1정도로 나눠서 테스트 데이터를 통해 에러율을 비교한다

  ![error](https://www.analyticsvidhya.com/wp-content/uploads/2014/10/training-error_11.png)

  - 에러가 최소가 되는 k값을 탐색

- 규제화 기법을 kNN에도 사용 가능

  - 규제화는 노이즈의 영향도를 줄이는 구속조건을 추가해서 loss(cost) 함수를 최소화
  - L^2 norm, L^1 norm, Dropout과 같은 방법이 사용됨



## Reference

1. 김의중, "알고리즘으로 배우는 인공지능, 머신러닝, 딥러닝 입문", 2016, 위키북스

2. TAVISH SRIVASTAVA, "Introduction to k-Nearest Neighbors: Simplified (with implementation in Python)", https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/