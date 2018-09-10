기존 모델
- SIFT로 추출된 모든 descriptors -> k-Means를 통해서 실제로 어떤 타입의 descriptor들이 어떠한 빈도를 갖고 분포하는지를 **가공된** 특징 벡터로써 추출 -> kNN을 통해서 가공된 특징 벡터를 이용하여 이미지 분류 수행


새로운 모델
SIFT 단계는 동일하게 구성 -> 하나의 이미지에서 발생하는 모든 descriptor에 대해 128size의 하나의 descriptor로 reduce -> 이를 kNN의 input으로 사용

이 때 reduce시 summation과 average 방법을 둘 다 사용해보고, 이를 비교