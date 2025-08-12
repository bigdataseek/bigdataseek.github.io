---
title: 2차시 6:머신러닝 From Scratch 3
layout: single
classes: wide
categories:
  - ML
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

## 7. PCA(주성분 분석) 알고리즘

### 7.1 PCA란 무엇인가?
#### (1)**PCA(Principal Component Analysis, 주성분 분석)**:
고차원 데이터를 저차원으로 축소하는 대표적인 차원 축소 기법입니다. 쉽게 말해, 복잡한 데이터에서 가장 중요한 정보만 뽑아내는 방법입니다.
* 핵심요소
    *   **PCA = 데이터에서 가장 중요한 방향 찾기**
    *   **주성분 = 데이터가 가장 많이 퍼진 방향**
    *   **차원 축소 = 중요한 정보만 남기고 나머지 버리기**
    *   **손실 압축 = 100% 복원은 불가능하지만 주요 정보는 보존**

*   실생활 예시로 이해하기:학생 100명의 성적표. 각 학생마다 10개 과목 점수가 있습니다.
    - 원본 데이터: 100명 × 10과목 = 1000개의 숫자
    - PCA 적용 후: 100명 × 2개 주요 요인 = 200개의 숫자
    - 이때 2개 주요 요인은 "문과 성향"과 "이과 성향" 같은 개념일 수 있습니다.

#### (2) PCA의 장점과 한계

*   장점
    - **차원 축소**: 저장 공간 절약, 계산 속도 향상
    - **시각화 가능**: 고차원 데이터를 2D/3D로 볼 수 있음
    - **노이즈 제거**: 중요하지 않은 정보 제거

*   한계
    - **해석의 어려움**: 주성분이 실제로 무엇을 의미하는지 알기 어려움
    - **선형 관계만**: 비선형 관계는 잡아내지 못함
    - **정보 손실**: 차원을 줄이면서 일부 정보는 사라짐

#### (3) 언제 PCA를 사용하나?

- **고차원 데이터 시각화**: 100차원 데이터를 2D 그래프로 보고 싶을 때
- **전처리**: 다른 머신러닝 알고리즘 전에 차원 축소
- **저장 공간 절약**: 큰 데이터를 압축해서 저장
- **노이즈 제거**: 데이터의 주요 패턴만 추출

### 7.2 PCA.py 코드 단계별 해석

#### (1) 클래스 초기화
```python
def __init__(self, n_components):
    self.n_components = n_components  # 축소할 차원 수
    self.components = None           # 주성분들 (나중에 계산됨)
    self.mean = None                # 데이터 평균 (나중에 계산됨)
```

*   **의미**: PCA 객체를 만들고, 몇 개의 주성분을 사용할지 정합니다.

#### (2) 학습 단계 (fit 메서드)

1.평균 중심화 (Mean Centering)
```python
self.mean = np.mean(X, axis=0)  # 각 특성의 평균 계산
X = X - self.mean               # 모든 데이터에서 평균을 빼기
```

*   **왜 하는가?**: 데이터의 중심을 원점(0,0)으로 옮깁니다.
    - **예시**: 키와 몸무게 데이터가 있다면, 평균 키 170cm, 평균 몸무게 70kg를 각각 0으로 만듭니다.

2.공분산 행렬 계산
```python
cov = np.cov(X.T)  # 공분산 행렬 계산
```
*   np.cov는 입력 데이터를 **행(row)이 변수(variables)**이고 **열(column)이 관측값(observations)**으로 간주. 따라서 X가 아닌, X.T를 사용

*   **공분산이란?**: 두 변수가 함께 변하는 정도
    - 양수: 한 변수가 커지면 다른 변수도 커짐 (키가 클수록 몸무게도 무거움)
    - 음수: 한 변수가 커지면 다른 변수는 작아짐
    - 0에 가까움: 두 변수는 관계없음

3.고유벡터와 고유값 계산
*   고유벡터(Eigenvector)와 고유값(Eigenvalue)의 수학적 정의

$ A \mathbf{v} = \lambda \mathbf{v} $

- A : 변환 행렬 (예: 공분산 행렬, 선형 변환 행렬 등)
- v : 고유벡터 (eigenvector), $\( \mathbf{v} \neq \mathbf{0} \)$
- $\lambda$: 고유값 (eigenvalue), 실수 또는 복소수


의미 해석: **행렬 $\( A \)$를 벡터 $\( \mathbf{v} \)$에 곱했을 때, 결과 벡터는 원래 벡터 $\( \mathbf{v} \)$와 같은 방향**(또는 정반대)  
→ 즉, 벡터 $\( \mathbf{v} \)$는 행렬 $\( A \)$의 선형 변환을 통해 **방향이 바뀌지 않고**, 단지 **크기만 $\( \lambda \)$배로 늘어나거나 줄어듦**을 의미

```python
eigenvectors, eigenvalues = np.linalg.eig(cov)
```
*   공분산 행렬의 고유벡터와 고유값을 구해 데이터의 주요 분산 방향과 크기를 분석
*   **고유벡터(Eigenvector)**: 데이터가 가장 많이 퍼진 방향
*   **고유값(Eigenvalue)**: 그 방향으로 얼마나 많이 퍼졌는지
*   **직관적 이해**: 럭비공을 생각해보세요
    - 가장 긴 축 = 1번째 주성분 (가장 큰 고유값)
    - 두 번째로 긴 축 = 2번째 주성분
    - 가장 짧은 축 = 마지막 주성분

4.주성분 정렬 및 선택
```python
idxs = np.argsort(eigenvalues)[::-1]  # 고유값 큰 순서로 정렬
eigenvalues = eigenvalues[idxs]
eigenvectors = eigenvectors[idxs]
self.components = eigenvectors[:self.n_components]  # 상위 n개만 선택
```

*   **의미**: 가장 중요한 방향들만 골라냅니다.

#### (3) 변환 단계 (transform 메서드)
```python
def transform(self, X):
    X = X - self.mean                    # 평균 중심화
    return np.dot(X, self.components.T)  # 주성분으로 투영
```

*   **변환 과정**:
    - 새로운 데이터도 같은 평균으로 중심화
    - 주성분 방향으로 데이터를 투영 (그림자 만들기)



### 7.3 train.py 코드 실행 예시
*   실제 예시 분석: 코드에서 사용한 아이리스 데이터셋:
    - **원본**: 150개 꽃 × 4개 특성 (꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비)
    - **PCA 후**: 150개 꽃 × 2개 주성분

```python
import matplotlib.pyplot as plt
from sklearn import datasets

# data = datasets.load_digits()
data = datasets.load_iris()
X = data.data
y = data.target

# Project the data onto the 2 primary principal components
pca = PCA(2)
pca.fit(X) #주성분 정렬 및 선택
X_projected = pca.transform(X) #변환,주성분 방향으로 데이터를 투영

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(
    x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()

```


### 7.4 Scikit-learn을 활용한 PCA 구현

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

# PCA 적용 (2차원으로 축소)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# 결과 시각화
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                        c=y, cmap='viridis', alpha=0.7)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Iris Dataset - PCA Visualization')
plt.colorbar(scatter)
plt.show()

# 정보 출력
print(f"원본 데이터 차원: {X.shape}")
print(f"PCA 후 차원: {X_reduced.shape}")
print(f"설명된 분산 비율: {pca.explained_variance_ratio_}")
print(f"누적 설명 분산: {np.cumsum(pca.explained_variance_ratio_)}")

```

## 8. 퍼셉트론(Perceptron)

### 8.1 퍼셉트론이란 무엇인가?

#### (1) 퍼세트론 정의
퍼셉트론(Perceptron)은 1957년에 발명된 **가장 간단한 인공신경망**입니다. 사람의 뇌에서 뉴런이 정보를 처리하는 방식을 모방한 알고리즘으로, 두 개의 그룹으로 데이터를 분류하는 **이진 분류(Binary Classification)**에 사용됩니다.**신경망**의 기본 단위, **딥러닝**의 출발점

- 실생활 예시로 이해하기
    - 스팸 메일 vs 일반 메일
    - 강아지 사진 vs 고양이 사진
    - 시험 합격 vs 불합격

#### (2) 퍼셉트론의 작동 원리

-   기본 구조
```
입력값들 → [가중치와 곱하기] → [모두 더하기] → [활성화 함수] → 결과
```

- 수식으로 표현
    1. **선형 결합**: `linear_output = w₁×x₁ + w₂×x₂ + ... + bias`
    2. **활성화 함수**: `결과 = 1 (만약 linear_output > 0), 0 (그렇지 않으면)`

-   핵심 개념들
    - **가중치(Weights)**: 각 입력값의 중요도를 나타내는 값
    - **편향(Bias)**: 결정 경계를 조정하는 값
    - **활성화 함수**: 최종 출력을 결정하는 함수 (여기서는 단위 계단 함수)

#### (3) 퍼셉트론의 장단점

*   장점
    - **단순함**: 이해하기 쉬운 구조
    - **빠른 학습**: 계산이 간단
    - **선형 분리 가능한 데이터**에 효과적

*   단점
    - **XOR 문제**: 선형적으로 분리되지 않는 데이터 처리 불가
    - **단층 구조**: 복잡한 패턴 학습 어려움
    - **이진 분류만**: 다중 클래스 분류 불가

#### (4) Linear Regression과 Perceptron의 업데이트 룰 비교

Linear Regression과 Perceptron은 머신러닝 모델로, 가중치(weights)를 업데이트하는 방식이 다릅니다. Linear Regression은 회귀(연속 값 예측)를 위해 비용 함수를 최소화하는 방식으로, Perceptron은 이진 분류를 위해 오차 기반으로 업데이트합니다. 아래에서 주요 차이점을 비교 설명하겠습니다.

1.**기본 개념**
- **Linear Regression**: 
    *   입력 특징(x)과 가중치(w)를 사용해 연속적인 출력(y)을 예측합니다. 비용 함수(일반적으로 Mean Squared Error, MSE)를 최소화하기 위해 Gradient Descent를 사용합니다.
- **Perceptron**: 
    *   단일 층 신경망으로, 입력을 받아 이진 출력(예: +1 또는 -1)을 예측합니다. 예측이 틀릴 때만 가중치를 업데이트하는 규칙 기반 학습입니다.

2.**업데이트 룰**
- **Linear Regression**:
  - 업데이트는 모든 데이터 포인트에 대해 비용 함수의 기울기(gradient)를 계산해 가중치를 조정합니다.
  - 수식: $\( w \leftarrow w - \alpha \cdot \frac{\partial J}{\partial w} \)$
    - 여기서 \( J \)는 비용 함수 (e.g., $\( J = \frac{1}{2n} \sum (y_i - \hat{y_i})^2 \)$), $\( \alpha \)$는 학습률(learning rate), $\( \frac{\partial J}{\partial w} \)$는 기울기 $(e.g., \( \frac{1}{n} \sum (y_i - \hat{y_i}) \cdot x_i \))$.
  - 특징: 연속적인 오차를 기반으로 매번 업데이트. 배치(batch) 또는 확률적(stochastic) 방식으로 적용 가능.

- **Perceptron**:
  - 업데이트는 예측이 틀릴 때만 발생합니다. (예측이 맞으면 변화 없음)
  - 수식: $\( w \leftarrow w + \eta \cdot (y - \hat{y}) \cdot x \)$
    - 여기서 $\( \eta \)는 학습률, \( y \)는 실제 레이블(+1 또는 -1), \( \hat{y} \)는 예측 출력(sign(w · x)), \( x \)$는 입력 벡터.
  - 특징: 이진 오차(틀림/맞음) 기반으로, 데이터가 선형적으로 분리 가능할 때 수렴합니다.


### 8.2 Perceptron.py 코드 분석

#### (1) 활성화 함수 (unit_step_func)
- x > 0이면 1, 그렇지 않으면 0을 출력하는 계단 함수
```python
def unit_step_func(x):
    return np.where(x > 0 , 1, 0)
```
**역할**: 입력값이 0보다 크면 1, 아니면 0을 출력
**비유**: 스위치처럼 켜짐(1) 또는 꺼짐(0)

#### (2) 퍼셉트론 클래스 초기화
```python
def __init__(self, learning_rate=0.01, n_iters=1000):
    self.lr = learning_rate      # 학습 속도
    self.n_iters = n_iters       # 학습 반복 횟수
    self.activation_func = unit_step_func
    self.weights = None          # 가중치
    self.bias = None             # 편향
```

*   **학습률(Learning Rate)**: 0.01
    - 너무 크면: 학습이 불안정
    - 너무 작으면: 학습이 느림
    - 적당한 값: 안정적이고 효율적인 학습

#### (3) 학습 과정 (fit 메서드)

1.초기화
```python
n_samples, n_features = X.shape
self.weights = np.zeros(n_features)  # 가중치를 0으로 초기화
self.bias = 0                        # 편향을 0으로 초기화

y_ = np.where(y > 0 , 1, 0)     # y를 0과 1로 변환
```

2.학습 알고리즘 (퍼셉트론 학습 규칙)
```python
for _ in range(self.n_iters):           # 지정된 횟수만큼 반복
    for idx, x_i in enumerate(X):       # 모든 데이터 포인트에 대해
        # 1. 예측값 계산
        linear_output = np.dot(x_i, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        
        # 2. 오차 계산 및 가중치 업데이트
        update = self.lr * (y_[idx] - y_predicted)
        self.weights += update * x_i
        self.bias += update
```

*   **학습 과정 설명**:
    1. **가중치와 편향 초기화** (모두 0으로 시작)
    2. **데이터 반복**
        * 입력(`x_i`)과 가중치(`weights`)를 곱하고 편향을 더함 → `linear_output`
        * `linear_output`을 활성화 함수에 넣어 예측값(`y_predicted`) 계산
    3. **업데이트 규칙 적용**
        * `update = 학습률 × (정답 - 예측값)`
        * 가중치: `weights += update × 입력`
        * 편향: `bias += update`

이 과정을 여러 번 반복하면서 **결정 경계(Decision Boundary)**를 점점 데이터에 맞춰 조정.

#### (4) 예측 과정 (predict 메서드)
```python
def predict(self, X):
    linear_output = np.dot(X, self.weights) + self.bias
    y_predicted = self.activation_func(linear_output)
    return y_predicted
```
학습된 가중치와 편향을 사용해 새로운 데이터를 분류합니다.

### 8.3 train.py 실행 예제

```python
#(1) 라이브러리 가져오기
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

#(2) 정확도 계산 함수
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

#(3) 데이터 생성
X, y = datasets.make_blobs( #가짜 데이터를 만드는 함수
    n_samples=150, 
    n_features=2, # 각 데이터 포인트의 특성(차원) 수 
    centers=2, # 생성할 클러스터(군집)의 개수 (2개)
    cluster_std=1.05, ## 각 클러스터의 표준편차 (클러스터의 퍼짐 정도)
    random_state=2
)

#(4) 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

#(5) 퍼셉트론 모델 학습
p = Perceptron(learning_rate=0.01, n_iters=1000)
p.fit(X_train, y_train)

#(6) 예측 및 정확도 출력
predictions = p.predict(X_test)

print("Perceptron classification accuracy", accuracy(y_test, predictions))

#(7) 시각화
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

#(8) 분류 선 그리기
x0_1 = np.amin(X_train[:, 0]) #x축의 최소값과 최대값.
x0_2 = np.amax(X_train[:, 0])

x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1] #직선의 y좌표를 계산
x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], "k") #두 점을 연결해 검은색("k") 직선

#(9) 그래프 범위 설정 및 표시
ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin - 3, ymax + 3]) #y축의 범위를 설정

plt.show()
```

결정 경계: 퍼셉트론은 입력 데이터를 두 그룹(예: 0과 1)으로 나누는 모델입니다.
이때 두 그룹을 나누는 ‘경계선’. 2차원에서는 **직선**, 3차원에서는 **평면**, 고차원에서는 **초평면(hyperplane)**이 됩니다.

- 1.결정 경계식
    -   퍼셉트론의 예측 함수는
    $\hat{y} = f(w \cdot x + b)$

    * $w \cdot x = w_1x_1 + w_2x_2 + \dots + w_nx_n$ (가중치와 입력의 내적)
    * $b$는 편향(bias)

    퍼셉트론이 0과 1을 구분하는 기준은 $w \cdot x + b = 0$인 지점입니다.

    즉, **결정 경계식**은:

    $w_1x_1 + w_2x_2 + \dots + w_nx_n + b = 0$

-   2.2차원 예시:2D 입력 데이터( $x_1, x_2$ )의 경우
    $w_1x_1 + w_2x_2 + b = 0$
    를 풀면:

    $x_2 = -\frac{w_1}{w_2}x_1 - \frac{b}{w_2}$

    * **기울기**: $-\frac{w_1}{w_2}$ → 가중치 비율에 따라 결정
    * **절편**: $-\frac{b}{w_2}$ → 편향이 변하면 직선이 평행 이동

    이 직선이 바로 **두 클래스를 구분하는 경계선**입니다.

-   3.기하학적 해석
    * **경계선의 방향**: 가중치 벡터 $w$는 경계선에 **수직**입니다.
    → $w$가 바뀌면 경계선의 기울기와 위치가 변합니다.
    * **경계선의 위치 이동**: 편향 $b$는 경계선을 **평행하게 이동**시킵니다.
    * **분류 규칙**:
        * $w \cdot x + b > 0$ → 클래스 1
        * $w \cdot x + b < 0$ → 클래스 0

-   4.시각적 비유: 
    -   마치 2D 평면 위에 **길 하나**를 그어서,
    길의 **한쪽**은 클래스 0, **다른 쪽**은 클래스 1로 나누는 것과 같습니다.
    이 길의 기울기와 위치를 조정하는 것이 **가중치와 편향 학습**입니다.



### 8.4 Scikit-learn을 사용한 퍼셉트론 구현

```python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score

# 1. 데이터 생성
X, y = datasets.make_blobs(
    n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
)

# 2. 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# 3. 퍼셉트론 모델 생성 및 훈련 (단 2줄!)
# Perceptron 기본 사용 (학습률 없음):
perceptron = Perceptron(max_iter=1000, random_state=42)
perceptron.fit(X_train, y_train)

# 4. 예측 및 정확도 계산
predictions = perceptron.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Perceptron classification accuracy: {accuracy:.4f}")

# 5. 시각화
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# 훈련 데이터 플롯
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.7)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Perceptron Classification with Decision Boundary')

# 결정 경계 그리기
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

# 격자 생성
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
# xx.shape: (147, 114)

# 각 격자점에 대한 예측
#xx, yy를 ravel()로 1D 배열로 평탄화(Flatten)
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])#xx와 yy를 열(column) 방향으로 결합
# Z.shape:  (16758,)

Z = Z.reshape(xx.shape)

# 결정 경계 표시
#levels=[0.5]→ Z=0과 Z=1 사이의 경계를 찾습니다.

ax.contour(xx, yy, Z, levels=[0.5], colors='red', linestyles='--', linewidths=2)

plt.colorbar(scatter)
plt.show()
```

## 9. SVM(Support Vector Machine)

### 9.1 SVM이란 무엇인가요?

SVM은 **분류(Classification)**를 위한 머신러닝 알고리즘입니다. 쉽게 말해, 데이터를 두 그룹으로 나누는 **가장 좋은 경계선**을 찾는 방법입니다.

-   일상생활 예시로 이해하기
    - 빨간 구슬과 파란 구슬이 섞여 있는 상자에서 둘을 구분하는 선을 그어야 한다고 생각해보세요
    - SVM은 이 두 그룹 사이에 **가장 안전한 경계선**을 그어줍니다
    - "안전하다"는 것은 양쪽 그룹으로부터 **최대한 멀리 떨어진** 경계선을 의미합니다

-   SVM의 핵심 개념
    1. 결정 경계(Decision Boundary)
        - 두 클래스를 나누는 선(2차원에서는 직선, 고차원에서는 평면)
        - 이 선을 **하이퍼플레인(Hyperplane)**이라고 부릅니다

    2. 서포트 벡터(Support Vectors)
        - 경계선에 가장 가까이 있는 데이터 포인트들
        - 이 점들이 경계선의 위치를 결정합니다
        - 마치 "경계선을 떠받치는 기둥" 같은 역할

    3. 마진(Margin)
        - 경계선과 가장 가까운 데이터 포인트들 사이의 거리
        - SVM은 이 **마진을 최대화**하려고 합니다

- 수학적 원리 (간단히)
    1. 기본 수식
    ```
    f(x) = w·x + b
    ```
    - **w**: 가중치 벡터 (경계선의 방향을 결정)
    - **x**: 입력 데이터
    - **b**: 편향(bias, 경계선의 위치를 조정)

    2. 분류 규칙
    - f(x) ≥ 0 이면 → 양성 클래스 (+1)
    - f(x) < 0 이면 → 음성 클래스 (-1)

    3. 목적 함수: SVM은 다음 두 가지를 동시에 최적화합니다:
    - **마진 최대화**: \|\|w\|\|²를 최소화
    - **분류 오류 최소화**: 잘못 분류된 데이터에 대한 페널티

- SVM의 장단점
    - SVM의 장점
    1. **높은 정확도**: 특히 고차원 데이터에서 우수한 성능
    2. **메모리 효율성**: 서포트 벡터만 저장하면 됨
    3. **다양한 커널**: 비선형 데이터도 처리 가능 (RBF, 다항식 등)
    4. **오버피팅 방지**: 정규화를 통한 일반화 성능

    - SVM의 단점
    1. **큰 데이터셋에서 느림**: 학습 시간이 오래 걸림
    2. **스케일링 필요**: 특성들의 크기가 비슷해야 함
    3. **확률 제공 안함**: 단순히 분류만 수행
    4. **노이즈에 민감**: 이상치의 영향을 받기 쉬움

### 9.2 SVM.py 코드 분석

1.클래스 초기화

```python
def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
    self.lr = learning_rate
    self.lambda_param = lambda_param
    self.n_iters = n_iters
    self.w = None
    self.b = None
```
- `learning_rate`: 학습 속도 (너무 크면 불안정, 너무 작으면 느림)
- `lambda_param`: 정규화 매개변수 (오버피팅 방지),가중치의 크기를 제한하여 과적합 방지
- `n_iters`: 학습 반복 횟수

2.학습 과정 (fit 메서드)

```python
def fit(self, X, y):
    n_samples, n_features = X.shape

    y_ = np.where(y <= 0, -1, 1)# 라벨을 -1 또는 1로 변환

    # init weights
    self.w = np.zeros(n_features)# 처음엔 모든 가중치를 0으로
    self.b = 0

    for _ in range(self.n_iters):
        for idx, x_i in enumerate(X):
            condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
            if condition:
                # 올바르게 분류된 경우 → 규제항만 적용
                self.w -= self.lr * (2 * self.lambda_param * self.w)
            else:
                # 잘못 분류된 경우 → 오차를 줄이는 방향으로 업데이트
                self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                self.b -= self.lr * y_[idx]
```
- SVM이 **"안전한 점들은 그대로 두고, 위험한 점들만 집중적으로 처리"**하는 효율적인 학습 전략을 보여줍니다. 정규화항은 항상 적용되어 오버피팅을 방지하면서, 분류 손실항은 필요할 때만 적용되어 계산 효율성도 높입니다.

이 조건문이 핵심입니다(업데이트 규칙):
- SVM의 최적화 문제
    - 최소화: $(1/2)\|\|w\|\|² + C∑ξ_i$
    - 다음 조건에서: $y_i(w·x_i - b) ≥ 1 - ξ_i$
           $,ξ_i ≥ 0$
        - $\|\|w\|\|²$: 정규화항 (마진 최대화를 위해)
        -   $C$: 정규화 강도 (λ의 역수 관계), λ가 클수록 더 단순한 모델
        -   $ξ_i$: 슬랙 변수 (마진 내부나 잘못 분류된 점들에 대한 허용)
- **조건이 참**: 데이터가 올바르게 분류되고 마진 밖에 있음 → 정규화만 수행
    - $∂L/∂w = λw$  (정규화항만)
    - $∂L/∂b = 0$   (편향 업데이트 없음)
- **조건이 거짓**: 데이터가 잘못 분류되었거나 마진 안에 있음 → 가중치 크게 조정
    - $∂L/∂w = λw - y_i*x_i$  (정규화항 + 분류 손실항)
    - $∂L/∂b = y_i$           (편향도 조정)

3.예측 (predict 메서드)

```python
def predict(self, X):
    approx = np.dot(X, self.w) - self.b
    return np.sign(approx) #부호만 보고 +1, -1 판단
```
- 결정 함수 값을 계산하고, 양수면 +1, 음수면 -1로 분류

### 9.3 train.py 실행 예제
```python
# Imports
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_blobs(
    n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
)
# 원래 레이블은 0, 1이지만, SVM은 일반적으로 -1과 +1을 클래스로 사용 → 0을 -1로 변경
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

clf = SVM()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

print("SVM classification accuracy", accuracy(y_test, predictions))

def visualize_svm():
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

    x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()

visualize_svm()
```

### 9.4 scikit-learn으로 SVM 구현하기
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 데이터 생성
X, y = datasets.make_blobs(
    n_samples=100, 
    n_features=2, 
    centers=2, 
    cluster_std=1.5, 
    random_state=42
)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 데이터 표준화 (SVM에서 중요!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM 모델 생성 및 학습
svm_model = SVC(
    kernel='linear',    # 선형 커널
    C=1.0,             # 정규화 매개변수
    random_state=42
)

# 모델 학습
svm_model.fit(X_train_scaled, y_train)

# 예측
y_pred = svm_model.predict(X_test_scaled)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy:.3f}")
print("\n분류 리포트:")
print(classification_report(y_test, y_pred))

# 서포트 벡터 정보
print(f"\n서포트 벡터 개수: {svm_model.n_support_}")
# 예[1, 2] → 첫 번째 클래스(-1)에 속한 서포트 벡터가 1개, 두 번째 클래스(+1)에 속한 것이 2개
# SVM은 각 클래스 경계 근처의 데이터 중 가장 중요한 것들만 서포트 벡터로 선택

print(f"전체 서포트 벡터 개수: {len(svm_model.support_vectors_)}")

def plot_svm_decision_boundary():
    """SVM 결정 경계 시각화"""
    plt.figure(figsize=(12, 5))
    
    # 원본 데이터로 시각화 (해석하기 쉽게)
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.7)
    plt.title('원본 학습 데이터')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # 표준화된 데이터로 결정 경계 그리기
    plt.subplot(1, 2, 2)
    
    # 격자 생성
    h = 0.02
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 예측
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 결정 경계 그리기
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    # 데이터 포인트 그리기
    scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], 
                         c=y_train, cmap='viridis', alpha=0.8)
    
    # 서포트 벡터 강조
    plt.scatter(svm_model.support_vectors_[:, 0], 
               svm_model.support_vectors_[:, 1],
               s=100, facecolors='none', edgecolors='red', linewidth=2,
               label='Support Vectors')
    
    plt.title('SVM 결정 경계 (표준화된 데이터)')
    plt.xlabel('Standardized Feature 1')
    plt.ylabel('Standardized Feature 2')
    plt.legend()
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()

# 시각화 실행
plot_svm_decision_boundary()

```

## 10. K-평균(K-Means)

### 10.1 K-Means란 무엇인가요?

K-Means는 **비지도 학습(Unsupervised Learning)**의 대표적인 **클러스터링(Clustering)** 알고리즘입니다. 쉽게 말해, **비슷한 데이터들끼리 그룹으로 묶어주는** 방법입니다.

1.일상생활 예시로 이해하기
- 학교에서 체육시간에 키 순서로 줄을 설 때를 생각해보세요
- 선생님이 "비슷한 키의 친구들끼리 3개 그룹을 만들어라"라고 하면
- 자연스럽게 **키가 비슷한 학생들끼리 모이게** 됩니다
- K-Means도 이와 같은 방식으로 **비슷한 특성을 가진 데이터들을 묶어줍니다**

2.K-Means의 핵심 개념
- K (클러스터 개수)
    - 몇 개의 그룹으로 나눌지 미리 정해야 합니다
    - 예: K=3이면 데이터를 3개 그룹으로 나눔

- 센트로이드(Centroid)
    - 각 그룹의 **중심점**
    - 그룹에 속한 모든 점들의 **평균 위치**
    - 마치 그룹의 "대표선수" 같은 역할

- 클러스터(Cluster)
    - 센트로이드 주변에 모인 데이터들의 그룹
    - 같은 클러스터에 속한 데이터들은 서로 비슷한 특성을 가짐

3.K-Means 알고리즘의 동작 과정
- 초기화: K개의 센트로이드를 랜덤하게 배치
- 할당: 각 데이터를 가장 가까운 센트로이드에 배정
- 업데이트: 각 그룹의 평균을 계산해 센트로이드를 새로운 위치로 이동
- 반복: 센트로이드가 더 이상 움직이지 않을 때까지 2-3단계 반복

4.K-Means의 특징
- 장점
    - **간단하고 직관적**: 알고리즘이 이해하기 쉬움
    - **빠른 속도**: 계산이 비교적 빠름
    - **확장성**: 큰 데이터셋에도 잘 동작
    - **구현 용이**: 코딩하기 쉬움

- 단점
    - **K값 선택**: 클러스터 개수를 미리 정해야 함
    - **초기값 민감**: 초기 센트로이드 위치에 따라 결과가 달라질 수 있음
    - **구형 클러스터**: 원형/구형 모양의 클러스터에 적합
    - **이상치 민감**: 극단값에 영향을 받기 쉬움


5.수학적 목적함수

$\underset{C}{\text{minimize}} \quad J = \sum_{k=1}^K \sum_{x_i \in C_k} \| x_i - \mu_k \|^2$

* **$\| x_i - \mu_k \|^2$** : 데이터 $x_i$와 그 데이터가 속한 클러스터 중심 $\mu_k$ 간의 **유클리드 거리 제곱**
* **$\sum_{x_i \in C_k}$** : 클러스터 $C_k$에 속한 모든 점에 대해 합산
* **$\sum_{k=1}^K$** : 모든 클러스터에 대해 합산
* 최종적으로 **데이터와 중심점 사이의 거리 제곱합**이 가장 작은 클러스터 구성을 찾는 것이 목표


6.적절한 K값 선택 방법
- 엘보우 방법(Elbow Method)
    - 다양한 K값에 대해 비용함수를 계산
    - K가 증가할 때 비용 감소폭이 급격히 줄어드는 지점을 선택

- 실루엣 분석(Silhouette Analysis)
    - 클러스터 내 응집도와 클러스터 간 분리도를 측정
    - 실루엣 점수가 가장 높은 K값 선택

- 도메인 지식
    - 업무 특성상 자연스러운 그룹 개수가 있다면 활용

### 10.2 KMeans.py 코드 분석

#### (1) 유클리드 거리 함수
```python
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))
```
**설명**: 두 점 사이의 직선 거리를 계산합니다 (피타고라스 정리)

#### (2) 초기화 (__init__)
```python
def __init__(self, K=5, max_iters=100, plot_steps=False):
    self.K = K
    self.max_iters = max_iters
    self.plot_steps = plot_steps

    # list of sample indices for each cluster
    self.clusters = [[] for _ in range(self.K)]

    # the centers (mean vector) for each cluster
    self.centroids = []
```
- `K`: 클러스터 개수 (기본값 5)
- `max_iters`: 최대 반복 횟수 (기본값 100)
- `plot_steps`: 과정을 시각화할지 여부

#### (3) 핵심 학습 과정 (predict)

1.초기 센트로이드 설정 및 클러스터 최적화
```python
def predict(self, X):
    self.X = X
    self.n_samples, self.n_features = X.shape

    # initialize
    random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
    self.centroids = [self.X[idx] for idx in random_sample_idxs]

    # optimize clusters
    for _ in range(self.max_iters):
        # assign samples to closest centroids (create clusters)
        self.clusters = self._create_clusters(self.centroids)

        if self.plot_steps:
            self.plot()

        # calculate new centroids from the clusters
        centroids_old = self.centroids
        self.centroids = self._get_centroids(self.clusters)

        if self._is_converged(centroids_old, self.centroids):
            break

        if self.plot_steps:
            self.plot()

    # classify samples as the index of their clusters
    return self._get_cluster_labels(self.clusters)
```
- 데이터 중에서 K개를 랜덤하게 선택해 초기 센트로이드로 설정

2.클러스터 생성
```python
def _create_clusters(self, centroids):
    # assign the samples to the closest centroids
    clusters = [[] for _ in range(self.K)]
    for idx, sample in enumerate(self.X):
        centroid_idx = self._closest_centroid(sample, centroids)
        clusters[centroid_idx].append(idx)
    return clusters
```
- 각 데이터를 가장 가까운 센트로이드에 배정

3.센트로이드 업데이트
```python
def _closest_centroid(self, sample, centroids):
    # distance of the current sample to each centroid
    distances = [euclidean_distance(sample, point) for point in centroids]
    closest_idx = np.argmin(distances)
    return closest_idx
```
- 각 클러스터의 평균을 계산해 새로운 센트로이드 위치 결정

4.수렴 확인
```python
def _is_converged(self, centroids_old, centroids):
    # distances between old and new centroids, for all centroids
    distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
    return sum(distances) == 0
```
- 센트로이드가 더 이상 움직이지 않으면 학습 완료

#### (4) 주요 헬퍼 메서드들

1.가장 가까운 센트로이드 찾기
```python
def _closest_centroid(self, sample, centroids):
    # distance of the current sample to each centroid
    distances = [euclidean_distance(sample, point) for point in centroids]
    closest_idx = np.argmin(distances)
    return closest_idx
```
- 각 센트로이드까지의 거리를 계산하고 가장 가까운 것을 선택

2.새로운 센트로이드 계산
```python
def _get_centroids(self, clusters):
    centroids = np.zeros((self.K, self.n_features))
    for cluster_idx, cluster in enumerate(clusters):
        cluster_mean = np.mean(self.X[cluster], axis=0)
        centroids[cluster_idx] = cluster_mean
    return centroids
```
- 각 클러스터에 속한 점들의 평균을 계산해 새로운 중심점 결정

3.샘플을 클러스터의 인덱스로 분류
```python
def _get_cluster_labels(self, clusters):
    # each sample will get the label of the cluster it was assigned to
    labels = np.empty(self.n_samples)
    for cluster_idx, cluster in enumerate(clusters):
        for sample_idx in cluster:
            labels[sample_idx] = cluster_idx

    return labels
```

4.시각화
```python
def plot(self):
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, index in enumerate(self.clusters):
        #index는 하나의 인덱스가 아니라 인덱스들의 리스트
        point = self.X[index].T #전치는 각 포인트의 좌표를 x좌표, y좌표 리스트로 전환
        ax.scatter(*point)

    for point in self.centroids:
        ax.scatter(*point, marker="x", color="black", linewidth=2)

    plt.show()
    #예를 들어: point = np.array([[1, 5, 9],[2, 6, 10]])
    #ax.scatter(*point)는 다음과 동일 ax.scatter([1, 5, 9], [2, 6, 10])
```

### 10.3 train.py 실행 예제

```python
np.random.seed(42)
from sklearn.datasets import make_blobs

X, y = make_blobs(
    centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
)
print(X.shape)

clusters = len(np.unique(y))
print(clusters)

k = KMeans(K=clusters, max_iters=150, plot_steps=True)
y_pred = k.predict(X)

k.plot()

```

### 10.3 scikit-learn을 활용한 KMeans 구현

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1. 예제 데이터 생성
X, y = make_blobs(
    centers=3,        # 클러스터 개수
    n_samples=500,    # 데이터 포인트 수
    n_features=2,     # 차원 수
    random_state=42
)

# 2. KMeans 모델 생성 및 학습
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 3. 예측 결과
labels = kmeans.labels_          # 각 데이터의 클러스터 번호
centroids = kmeans.cluster_centers_  # 각 클러스터의 중심점 좌표

# 4. 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.scatter(centroids[:, 0], centroids[:, 1], 
            c='red', marker='X', s=200, label='Centroids')
plt.legend()
plt.show()
```