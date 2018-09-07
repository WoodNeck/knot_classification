# SIFT(Scale Invariant Feature Transform)

Scale, Rotation에 무관한 주요 특징점을 추출하는 알고리즘



## 개요

![eye cell](https://t1.daumcdn.net/cfile/tistory/9992523359C4C2C818)

- SIFT는 우리 눈에서 변화, 그리고 Edge에 대한 정보만을 다루는 망막 신경절 세포를 모사하여 동작한다

- 망막 신경절 세포는 ON-centre cell과 OFF-centre cell이 존재한다

  - On-centre cell의 경우 cell의 중앙 부분에 빛을 받을 경우 활성화되고, 가장자리에 빛을 받을 경우 억제되는 특성이 있다
  - Off-centre cell은 On-centre cell과 반대의 방식으로 동작한다

  ![Light on cell](https://t1.daumcdn.net/cfile/tistory/99725A3359C63A9E31)

  ![Edge Detection](https://t1.daumcdn.net/cfile/tistory/9947753359C4BFD417)

- 망막 신경절 세포의 해당 특성 덕에 중앙과 가장자리에 전부 빛을 비출 경우 안 비춘것과 활성화 정도가 거의 동일하다
- 이러한 특성 때문에, 빛의 세기가 달라지는 Edge를 경계로 망막 신경절 세포를 이동할 때 가장 큰 활성/억제 정도를 나타내게 된다

![LOG](https://slideplayer.com/slide/8727137/26/images/21/2D+edge+detection+filters.jpg)

- 이를 모사하기 위해서, SIFT는 기본적으로 **LoG**(Laplacian of gaussian) filter의 개념을 사용한다

  - LoG의 경우, Off-centre cell의 동작과 굉장히 유사한 형태의 그래프를 나타낸다
    - x축을 기준으로 flip할 경우 On-centre cell의 동작을 모방할 수 있다

- 하지만 **LoG**의 경우 연산비용이 높다는 단점이 있기 때문에, SIFT는 이를 **DoG**(Differential of Gaussian)를 통해 구현했다

  ![DoG2](https://sensblogs.files.wordpress.com/2011/08/dog.jpg)

  - **DoG**는 **LoG**와 유사한 그래프의 형태를 갖으면서도, 더 빠르게 연산이 가능하기 때문에 대체하여 사용한다

- Gaussian의 경우 Gaussian filter를 이미지에 컨볼루션하여 구현한다

  ![Gaussian Filter](http://forum.falinux.com/zbxe/files/attach/images/583/787/543/Bluring-GaussianProfile.png)

  - 이러한 Gaussian mask의 경우 기존 영상에서 새로운 블롭이나 에지가 생성되지 않고, 연속 공간에 정의되는 특징이 있기 때문에 유용하다
    - 즉, 단지 기존의 이미지를 smoothing하는 역할만을 한다

## SIFT 특징점 추출

#### Scale Space 이론

같은 물체가 다른 두 이미지에 다른 크기로 나타날 때, 같은 특징을 얻어내기 위해서는 다음과 같은 방법이 있다

>  **1. 특징점 알고리즘의 연산자(어떤 크기의 특징점을 추출할 것인지)를 작은 크기에서 시작하여, 점점 키워가며 여러 스케일의 특징점 집합을 얻는다**
>
> **2. 이미지의 해상도를 점점 줄여 다중 스케일 영상을 구축하고, 특징점 알고리즘 연산자를 동일 크기로 여러 스케일에 적용**

하지만, 

>  **1. 비슷한 특징이 여러 스케일에 걸쳐 나타날 수 있고**
>
> **2. 특징점의 개수가 많아져 효율이 좋지 못하다**

이를 해결하기 위해서는 Scale-invariant(스케일 불변)한 특징을 찾는것이 중요하다.

```python
# 이미지 f에 대해, 스케일 불변한 특징점 리스트 F 출력
f에서 스케일 정도에 따라 다중 스케일 영상, M = {f^s0, f^s1, f^s2, ...}를 구성
M에서 3차원 극점을 찾아 특징점 집합 F로 취한다
```

다중 스케일 영상 M은 두 가지 방식으로 구현할 수 있다.

1. 거리가 멀어질 수록, 물체의 세밀한 내용은 사라지고, 윤곽은 점점 흐릿해진다는 원리에서 사용할 수 있는 Gaussian Smoothing
2. 영상의 해상도를 반씩 줄여나가면서 피라미드 영상 구축

Gaussian Smoothing의 경우 σ이 연속적이므로, 실제 영상의 Scaling에 대응하여 구현하는 것이 가능하지만, 해상도를 반씩 줄여나갈 경우 스케일은 항상 두 배씩 감소하므로 임의의 스케일에 대응할 수 없는 문제가 있다

스케일 공간은, (y, x, t)로 표현될 수 있다. 스무싱의 강도를 나타내는 σ(표준편차)에 대해 분산 t=σ^2로 정의되고 이를 이미지의 x, y축에 추가하여 3차원의 스케일 공간을 나타내는 것이 가능하다.

스케일 공간 내에서 영상을 미분하여 얻은 도함수를 이용하면 다양한 정보를 얻는 것이 가능한데, n차까지의 도함수 n-jet에 대해 2-jet (d_y, d_x, d_yy, d_xx, d_yx, d_xy)의 경우에 대해 가령 d_x의 경우 x방향으로 경계선의 존재 여부를 알려주며, 이 도함수는 모두 회전에 공변(covariant)이다. 이 값들을 통해 회전에 불변인 값인 gradient size, laplacian, det(H)같은 값을 정의 가능하다.

#### SIFT keypoint extraction

![DoG](http://www.aishack.in/static/img/tut/sift-dog-idea.jpg)

- SIFT에서는 앞선 Scale space를 구현할 때, Gaussian smoothing과 image downsampling을 같이 사용한다.

- 먼저, image downsampling의 경우 이미지를 downsampling하여 이미지의 사이즈가 4*4정도가 될때까지 downsampling한다

- 원본 이미지와, 각각의 downsampling된 이미지에 대해 σ, kσ, k^2σ, ... k^5σ까지 6번의 Gaussian Filtering을 적용하여 6개의 Smoothing된  이미지를 만들어낸다.

  - k를 논문에서는 실험적으로 2^(1/3)으로 정했다

- 이렇게 Smoothing된 6개의 이미지 한 세트를 하나의 옥타브라고 하며, 제일 아랫단계를 옥타브 0, 이후를 1, 2, 3, ...으로 표기한다

- 각각의 옥타브에 대해 DoG를 적용한다. 그러면 하나의 옥타브에서 6개의 이미지에 대해 총 5장의 DoG 이미지가 생성된다.

- 이러한 방식으로 DoG 피라미드를 생성하게 되면, 하나의 옥타브의 5개의 DoG 이미지에 대해 맨 아래와 맨 위의 것을 제외한 세 개의 이미지에 대해서 3차원 공간에서의 극점을 탐색한다.

  ![DoG Feature Extraction](https://www.researchgate.net/publication/273897393/figure/fig7/AS:325364292309037@1454584300829/The-process-of-local-extrema-detection-based-on-Difference-of-Gaussian-DoG.png)

- 이미지의 각 픽셀마다, 해당 픽섹 주위의 8픽셀과, 위 아래 이미지의 9픽셀씩 총 26픽셀(8+9+9)을 해당 픽셀과 비교해 주변의 모든 픽셀의 값에 비해 해당 픽셀의 값이 최소 또는 최대가 될 때 특징점으로 추출하게 된다

- 특징점 추출시 출력값은 (y, x, σ) -> 이미지 내에서의 좌표와, scale 정보를 갖고 있게 된다

## SIFT 특징 기술

주변과 두드러지게 다른 점을 관심점(지역 특징)으로 추출해낸 다음에는 이 벡터들에 대해 기술자(descriptor)를 생성해내야 한다.

이는, 특징점의 경우 위치나 scale 같은 정보만을 갖고있을 뿐, 매칭이나 인식을 수행하기 위해서는 정보가 부족하기 때문이다.

#### SIFT에서의 descriptor 추출방법

1. 지배방향 탐색(Dominant Orientation)

   기술자 추출에 앞서, 회전 불변성을 달성하기 위해 지배방향을 먼저 탐색해야 한다.

   먼저, 키포인트를 중심으로 일정 크기의 윈도우를 씌우고, 윈도우 내의 화소 집합에 대해 그레디언트 방향 히스토그램을 구하는데 이를 10도 간격으로 양자화(Quantization)하여 36개의 칸을 가진 히스토그램을 생성한다.

   이 히스토그램에서 가장 큰 값을 갖는 방향을 지배 방향으로 선정하며, 최고값의 0.8배 이상인 값이 있다면, 이를 분리하여 새로운 특징점으로 만든다.

   때문에, 같은 좌표 내에서 방향이 다른 두 개의 특징점이 만들어질 수도 있다

2. Descriptor calculation

   지배방향 탐색 후 특징점의 데이터는 지배방향을 더하여 (y, x, σ, θ)가 된다.

   먼저, 지배방향을 이용하여 이미지 내에서 축을 -θ만큼 회전하여 새로운 좌표계를 설정하여 방향 불변성을 달성한다.

   해당 좌표계를 기준으로 윈도우를 씌우고, 이를 4\*4 크기로 분할하여 16개의 블록으로 나눈다(*이 때, 4 = 3\*σ0*)

   각각의 블록에 대해 블록 내의 화소의 그래디언트 계산 후, 이를 8방향으로 양자화하여 히스토그램을 구한다.

   16개의 각 블록에 대해 8방향을 포함하는 벡터가 발생하므로, 총 16 * 8 = 128차원의 특징 벡터가 발생한다

3. Normalization

   광도 불변성 달성을 위해, x를 |x|로 나눠 단위 벡터로 변환한다

   단위 벡터에 0.2보다 큰 요소가 존재시 0.2로 바꾸고 다시 단위 벡터로 변환한다

위 과정을 모두 거치면, 스케일, 회전, 광도 변환에 모두 불변성을 갖는 128차원의 기술자가 생성된다