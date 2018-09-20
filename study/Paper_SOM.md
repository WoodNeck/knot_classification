# Wood Inspection with non-supervised clustering

해당 논문에서는, 동일한 모델을

1. sound wood와 defect의 구분,
2. defect간의 구분

두 가지 경우에 적용하여, 80% 이상의 accuracy를 얻어낸다.

## Knot Inspection시의 문제점

- Lumber grading을 사람이 직접 수행할 때, 그 정확도는 70%에 미처 달하지 못한다.

- Knot classification의 경우도 이와 마찬가지로, 같은 타입의 knot의 경우에도 색상이 다른것들과 크게 차이나는 경우도 있기 때문에, 육안 구별시 그 정확도는 좋지 못하다.

- 그런 이유에서, Knot Classification은 지도 학습이 적절하지 않은 케이스이다. 인간이 직접 라벨링을 수행할 때, 그 라벨링의 정확도가 좋지 못하기 때문이다.

- 이러한 이유로 다른 논문에서는 Knot classification은 기계학습시 80%의 정확도를 넘기기 어려운 문제라고 한다.

## 모델

해당 논문에서는 사용 알고리즘이 동일한 모델을 두 가지에 동시에 적용한다.

1. 목재에서 Defect가 있는 부분 탐색
2. Defect간의 구분

### MODEL

>Feature Extraction: **Color Histogram**
>Additional Feature: **1, 2차 모멘트**

- knot을 구성하는 타원의 방향과, major 축과 minor 축에서의 길이

>Classification: **Self-Organizing Map**

- 특성 벡터를 차원 축소하고, 군집화

## Color Histogram

![Color Histogram](https://www.researchgate.net/profile/Muhammad_Usman_Tariq/publication/258885776/figure/fig4/AS:340827483197444@1458271012330/Color-Histogram-The-features-extracted-from-the-above-histogram-of-the-image-are-stored.jpg)

이미지에 컬러 히스토그램을 적용하면 명암(1차원)/RGB(3차원)값이 나타난 빈도수를 표현하게 되며, 이를 이미지의 특징으로 이용할 수 있다.

사용하고 싶은 데이터에 따라 명암(L, 1차원), HS(2차원), RGB(3차원) 등을 만들 수 있다

## Self-Organizing Map

자기조직화지도(SOM)는 차원 축소와 군집화(Clustering)를 동시에 수행하는 모델이다

![SOM](https://image.slidesharecdn.com/somchuc-110117091410-phpapp01/95/sefl-organizing-map-7-728.jpg?cb=1295255891)

기본적으로, 사람이 눈으로 확인 가능한 2차원, 또는 3차원 공간에 고차원의 입력을 매핑하는 역할을 한다.

이 때, 고차원에서의 유사도가 높을수록 저차원에서의 유사도도 높도록 매핑하는 방식이다.

![SOM Learn](http://i.imgur.com/eHUVAtr.gif)

SOM은 인스턴스 기반 학습을 하여 fitting되며 최초 격자 위치에서, 학습 데이터 인스턴스를 하나씩 추가할때마다 격자의 위치를 학습 데이터에 맞게 조정하여 추가하며, 추가하는 x와 현재 가까운 격자를 더 많이 업데이트한다.
