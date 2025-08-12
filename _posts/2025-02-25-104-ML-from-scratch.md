---
title: 2차시 4:머신러닝 From Scratch 1
layout: single
classes: wide
categories:
  - ML
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

## 1. K-최근접 이웃(K-Nearest Neighbors, KNN)
- 출처:[How to implement KNN from scratch with Python](https://www.youtube.com/watch?v=rTEtEy5o3X0&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=2)


### 1.1 `KNN.py` :K-최근접 이웃(K-Nearest Neighbors) 알고리즘

```python
import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance
```

* **`euclidean_distance`** : 두 점 사이의 거리를 계산하는 함수

  * KNN에서 ‘가장 가까운’ 데이터를 찾기 위해 거리 계산이 필요합니다.
  * 여기서는 **유클리드 거리 공식** 사용

    $거리 = \sqrt{\sum (x_1 - x_2)^2}$


```python
class KNN:
    def __init__(self, k=3):
        self.k = k
```

* **`__init__`** : 모델이 몇 개의 이웃(k)을 고려할지 저장

  * 예: `k=3` → 가장 가까운 3개의 데이터 보고 다수결 결정



```python
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
```

* **`fit`** : 학습 데이터를 저장

  * KNN은 **훈련 과정에서 별도의 계산을 하지 않고**, 단순히 데이터만 저장합니다.
  * 예측 시에 저장된 데이터를 직접 사용하여 거리 계산


```python
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
```

* **`predict`** : 입력된 모든 샘플에 대해 `_predict` 호출

  * 결과는 각 샘플의 예측 라벨 리스트


```python
    def _predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
```

* **`_predict`** : 한 개 샘플에 대한 예측 과정

  1. 모든 훈련 데이터와 거리 계산
  2. 거리 순으로 정렬 후 **가장 가까운 k개의 인덱스** 선택
  3. 그 인덱스에 해당하는 라벨 목록 추출
  4. **다수결**로 가장 많이 나온 라벨 선택

💡 **핵심 개념 정리**

* **KNN은 훈련 시 계산이 거의 없고, 예측 시 거리를 계산하는 방식**
* k 값이 작으면 모델이 민감(노이즈 영향 큼), 크면 부드럽지만 정확도 저하 가능
* 거리 계산 방식은 유클리드 거리 외에도 맨해튼 거리, 코사인 유사도 등 가능

### 1.2 `train.py` — 모델 실행 및 테스트

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN
```

* 필요한 라이브러리 불러오기
* `KNN` 클래스는 우리가 만든 `KNN.py`에서 불러옴

```python
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])
```

* 산점도(Scatter plot) 색상 지정

```python
iris = datasets.load_iris()
X, y = iris.data, iris.target
```

* **Iris 데이터셋 로드**

  * X: 꽃받침/꽃잎 길이·너비(4개 특성)
  * y: 꽃 품종(0, 1, 2)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)
```

* 데이터셋을 80% 훈련 / 20% 테스트로 분리

```python
plt.figure()
plt.scatter(X[:,2], X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()
```

* 꽃잎 길이($X\[:,2]$)와 너비($X\[:,3]$)만 이용하여 시각화

```python
clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

* **KNN 객체 생성(k=5)**
* `fit`으로 훈련 데이터 저장
* `predict`로 테스트 데이터 예측

```python
print(predictions)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)
```

* 예측 결과와 정확도 출력

### 1.3 scikit-learn을 사용해서 같은 작업을 훨씬 간단하게 구현

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 1. 데이터 로드
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 2. 훈련/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# 3. sklearn KNN 모델 생성 및 학습
clf = KNeighborsClassifier(n_neighbors=5)  # k=5
clf.fit(X_train, y_train)

# 4. 예측
predictions = clf.predict(X_test)
print(predictions)

# 5. 정확도 계산
acc = np.mean(predictions == y_test)
print(acc)
```

## 2. **Linear Regression(선형 회귀)**
- 출처: [How to implement Linear Regression from scratch with Python](https://www.youtube.com/watch?v=ltXSoduiVwY&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=3)

### 2.1 `LinearRegression.py` — 선형 회귀 모델 구현

```python
import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr              # 학습률 (learning rate)
        self.n_iters = n_iters    # 반복 횟수
        self.weights = None       # 가중치
        self.bias = None          # 절편

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # 처음에는 가중치를 0으로 초기화
        self.bias = 0

        for _ in range(self.n_iters):
            # 예측 값 계산
            y_pred = np.dot(X, self.weights) + self.bias

            # 가중치와 절편에 대한 기울기(gradient) 계산
            # 편미분에서 나온 곱하기 2는 학습률에 포함시킬 수 있는 상수이기 때문에, 
            # 코드에서는 생략하는 경우가 많습니다
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 경사 하강법(Gradient Descent)으로 파라미터 업데이트
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
```

* **`fit` 메서드**

  * 데이터를 보고 **최적의 직선**(y = wx + b)을 찾는 과정
  * `np.dot(X, self.weights)` : X(입력)와 w(가중치)를 곱해 예측값을 구함
  * **기울기 dw, db**를 계산 → 경사 하강법으로 조금씩 수정
  
    ```python
    #기울기 계산 부분:
    dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
    ```
    - 오차 `(y_pred - y)`를 계산
    - 각 특성별로 오차에 대한 기여도를 구함
    - 평균을 내서 안정적인 그래디언트 계산

* **경사 하강법**
  * 예측값이 실제값과 가까워지도록 w와 b를 반복적으로 조정
  * 학습률(`lr`)이 너무 크면 발산하고, 너무 작으면 학습이 느림
  ```python
    #최적값 찾아가는 부분:
    self.weights = self.weights - self.lr*dw
    ```
    - dw가 양수면 가중치를 줄임 (빼기 때문에)
    - dw가 음수면 가중치를 늘림 (음수를 빼므로)
    - dw가 0에 가까우면 가중치 변화량도 0에 가까움


### 2.2 `train.py` — 모델 학습 및 시각화

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

# 1. 샘플 데이터 생성
X, y = datasets.make_regression(
    n_samples=100,   # 데이터 개수
    n_features=1,    # 독립 변수 개수 (1개 → 단순 회귀)
    noise=20,        # 데이터에 잡음 추가
    random_state=4
)

# 2. 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# 3. 데이터 시각화 (산점도)
plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
plt.show()

# 4. 모델 생성 및 학습
reg = LinearRegression(lr=0.01)  # 학습률 0.01
reg.fit(X_train, y_train)

# 5. 예측
predictions = reg.predict(X_test)

# 6. 평가 (평균 제곱 오차, MSE)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

print("MSE:", mse(y_test, predictions))

# 7. 예측 직선 그리기
y_pred_line = reg.predict(X)
plt.scatter(X_train, y_train, color="orange", s=10)
plt.scatter(X_test, y_test, color="blue", s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
plt.show()
```

*   실행 흐름
1. **데이터 준비** → `make_regression`으로 가상의 선형 데이터 생성
2. **데이터 분리** → 학습용 80%, 테스트용 20%
3. **모델 학습** → 경사 하강법으로 w, b 찾기
4. **예측** → 학습한 w, b로 새로운 값 예측
5. **평가** → MSE 계산 (값이 작을수록 좋음)
6. **시각화** → 데이터와 예측 직선을 함께 그림

*   입문자를 위한 핵심 요약
    * **선형 회귀**: 데이터에 가장 잘 맞는 직선을 찾는 알고리즘
    * **가중치(w)**: 기울기
    * **절편(b)**: 직선이 y축과 만나는 지점
    * **학습률(lr)**: 한 번에 이동하는 폭
    * **MSE**: 예측이 얼마나 틀렸는지 수치로 나타내는 지표

### 2.3 **scikit-learn**을 활용해 선형 회귀를 훨씬 간단하게 구현

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. 샘플 데이터 생성
X, y = datasets.make_regression(
    n_samples=100,   # 데이터 개수
    n_features=1,    # 독립 변수 개수
    noise=20,        # 잡음 추가
    random_state=4
)

# 2. 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# 3. 모델 생성 & 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 4. 예측
y_pred = model.predict(X_test)

# 5. 평가 (MSE)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
print("기울기(w):", model.coef_)
print("절편(b):", model.intercept_)

# 6. 전체 데이터에 대한 예측 직선
y_pred_line = model.predict(X)

# 7. 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color="orange", s=10, label="Train data")
plt.scatter(X_test, y_test, color="blue", s=10, label="Test data")
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction line")
plt.legend()
plt.show()
```


*   실행 흐름
1. **데이터 생성** → `datasets.make_regression`으로 선형 관계가 있는 샘플 데이터 준비
2. **데이터 분리** → 학습용/테스트용으로 나누기
3. **모델 생성** → `LinearRegression()` 객체 생성
4. **학습** → `.fit()`으로 w와 b 자동 계산
5. **예측** → `.predict()` 사용
6. **평가** → `mean_squared_error`로 MSE 계산
7. **시각화** → 학습 데이터, 테스트 데이터, 예측 직선 함께 표시

### 2.4 예측, 평가, 최적화의 수학적 표현

1.예측값 계산 수식
- 선형 회귀에서 **예측값** $\hat{y}$는 다음과 같이 계산합니다.

> $\hat{y} = w \cdot x + b$

* $\hat{y}$ : 예측값 (prediction)
* $w$ : 가중치(Weight, 기울기)
* $x$ : 입력 값
* $b$ : 절편(Bias, y축과 만나는 값)

**여러 개의 입력 변수가 있을 경우(다중 회귀)**:

> $\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$

또는 **벡터/행렬** 형태로:

> $\hat{y} = \mathbf{X} \cdot \mathbf{w} + b$

2.손실 함수 (MSE)

모델이 얼마나 잘 맞는지 측정하기 위해 **평균 제곱 오차(Mean Squared Error)**를 사용합니다.

> $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$

* $n$ : 데이터 개수
* $y_i$ : 실제값
* $\hat{y}_i$ : 예측값
* 값이 작을수록 예측이 잘 된 것

3.경사 하강법 (Gradient Descent)

경사 하강법은 손실(MSE)을 줄이는 방향으로 $w$와 $b$를 조금씩 조정하는 알고리즘입니다.

*   가중치 $w$ 업데이트 수식

> $w := w - \alpha \cdot \frac{\partial \text{MSE}}{\partial w}$

*   절편 $b$ 업데이트 수식

> $b := b - \alpha \cdot \frac{\partial \text{MSE}}{\partial b}$

* $\alpha$ : 학습률(learning rate), 한 번에 이동하는 크기
* $\frac{\partial \text{MSE}}{\partial w}$ : MSE를 $w$에 대해 미분한 값 (기울기)
* $\frac{\partial \text{MSE}}{\partial b}$ : MSE를 $b$에 대해 미분한 값

4.기울기(Gradient) 계산

**단일 변수일 경우**:

> $\frac{\partial \text{MSE}}{\partial w} = \frac{2}{n} \sum_{i=1}^{n} ( \hat{y}_i - y_i ) \cdot x_i$

> $\frac{\partial \text{MSE}}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} ( \hat{y}_i - y_i )$

*  편미분에서 나온 곱하기 2는 학습률에 포함시킬 수 있는 상수이기 때문에, 코드에서는 생략하는 경우가 많다
*   경사 하강법은 **"정확한 기울기 값"**이 아니라 **"기울기의 방향"**이 중요

6.한 줄 요약
* **예측**: $\hat{y} = w \cdot x + b$
* **평가**: MSE로 오차 계산
* **최적화**: 경사 하강법으로 w, b를 반복 조정

7.np.dot() vs np.matmul() vs $*$
- $*$ 또는 np.multiply(): 요소별 곱셈 (Hadamard product).
- np.matmul(): 행렬 곱셈에 특화되어 있으며, 브로드캐스팅 규칙이 다릅니다.
- np.dot(): 더 일반적인 점곱 연산 (스칼라, 벡터, 행렬, 텐서), NumPy 3.5+에서는 @ 연산자로 대체 가능

8.편미분이란?
- 여러 개의 변수를 가진 함수에서 한 변수만 변화시켰을 때 함수가 어떻게 변하는지 보는 것
- 수학 기호:
> $\frac{\partial f}{\partial x}$
  → "함수 $f$를 $x$에 대해 편미분"

-  왜 필요한가?
    * MSE는 $w$와 $b$라는 **두 개의 변수**를 가진 함수입니다.
    * $w$를 바꿨을 때 MSE가 어떻게 변하는지 보려면
    $b$는 **고정**하고 $w$만 변화시키며 미분해야 합니다.
    * 반대로 $b$를 바꿨을 때 변화를 보고 싶으면
    $w$는 **고정**하고 $b$만 변화시키며 미분합니다.

9.체인 룰이란?
* **정의**:
  하나의 함수가 여러 함수로 연결되어 있을 때,
  바깥 함수와 안쪽 함수의 **변화량(미분값)**을 곱해서 전체 변화량을 구하는 방법입니다.
* 기호:
    > $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$

    * $y$는 $u$에 의존하고,
    * $u$는 $x$에 의존할 때 적용

*  쉬운 예시
    >$y = (3x + 2)^2$

    *   **겉 함수**: $\square^2$  → 2 × (안쪽)
    *   **안쪽 함수**: $3x + 2$ → 미분하면 3

    *   $\frac{dy}{dx} = 2(3x+2) \cdot 3= 6(3x+2)$

10.fit()에서 X가 아닌 X.T를 사용하는 이유: 차원 관점에서

가정:
* $X$ : $(n_{\text{samples}}, n_{\text{features}})$ → 예: (100, 3)
* $w$ : $(n_{\text{features}}, 1)$
* $y$ : $(n_{\text{samples}}, 1)$
* $y_{\text{pred}}$ : $(n_{\text{samples}}, 1)$

> `(y_pred - y)` 의 shape = $(n_{\text{samples}}, 1)$

만약 그냥 `np.dot(X, (y_pred - y))`를 하면:

* $(n_{\text{samples}}, n_{\text{features}})$ × $(n_{\text{samples}}, 1)$ → **곱셈 불가** (행렬 곱 조건 불만족)

그래서 $X$ 를 전치(`X.T`)해서:

> $X.T$ : $(n_{\text{features}}, n_{\text{samples}})$

* $X.T$ × `(y_pred - y)` : $(n_{\text{features}}, 1)$ → 각 feature별 기울기 계산 가능


## 3. Logistic Regression
- 출처: [How to implement Logistic Regression from scratch with Python](https://www.youtube.com/watch?v=YYEJ_GUguHw)


### 3.1 LogisticRegression.py - 로지스틱 회귀 클래스 구현

1.시그모이드 함수
```python
def sigmoid(x):
    return 1/(1+np.exp(-x))
```
- 시그모이드 함수는 로지스틱 회귀의 핵심입니다
- 어떤 실수값이든 0과 1 사이의 확률값으로 변환해줍니다
- S자 모양의 곡선을 그리며, 이진 분류에 완벽합니다
- np.exp()는 인자가 스칼라이면 스칼라를, 벡터이면 벡터를 반환

2.클래스 초기화
```python
def __init__(self, lr=0.001, n_iters=1000):
    self.lr = lr          # 학습률 (learning rate)
    self.n_iters = n_iters # 반복 횟수
    self.weights = None    # 가중치
    self.bias = None       # 편향
```
- `lr`: 학습률이 클수록 빠르게 학습하지만 불안정할 수 있습니다
- `n_iters`: 경사하강법을 몇 번 반복할지 정합니다

3.모델 훈련 (fit 메소드)
```python
def fit(self, X, y):
    n_samples, n_features = X.shape  # 샘플 수, 특성 수
    self.weights = np.zeros(n_features)  # 가중치를 0으로 초기화
    self.bias = 0                        # 편향을 0으로 초기화
```

4.**경사하강법 반복:**
```python
for _ in range(self.n_iters):
    # 1. 선형 예측값 계산
    linear_pred = np.dot(X, self.weights) + self.bias
    
    # 2. 시그모이드로 확률 변환
    predictions = sigmoid(linear_pred)
    
    # 3. 그래디언트 계산 (미분값)
    dw = (1/n_samples) * np.dot(X.T, (predictions - y))
    db = (1/n_samples) * np.sum(predictions-y)
    
    # 4. 가중치와 편향 업데이트
    self.weights = self.weights - self.lr*dw
    self.bias = self.bias - self.lr*db
```

이 과정은 **경사하강법**으로:
- 현재 예측값과 실제값의 차이를 계산합니다
- 이 오차를 줄이는 방향으로 가중치를 조금씩 조정합니다
- 이를 반복하여 최적의 가중치를 찾습니다

5.예측 (predict 메소드)

```python
def predict(self, X):
    linear_pred = np.dot(X, self.weights) + self.bias  # 선형 계산
    y_pred = sigmoid(linear_pred)                      # 확률로 변환
    class_pred = [0 if y<=0.5 else 1 for y in y_pred] # 0 또는 1로 분류
    return class_pred
```
- X : 일반적으로 (n_samples, n_features) 형태의 2차원 배열
- self.weights : (n_features,) 형태의 1차원 배열
- np.dot(X, self.weights)의 결과는 (n_samples,) 형태의 1차원 배열
- 확률이 0.5 이상이면 클래스 1, 미만이면 클래스 0으로 분류합니다

### 3.2 train.py - 모델 훈련 및 테스트

1.데이터 준비
```python
bc = datasets.load_breast_cancer()  # 유방암 데이터셋 로드
X, y = bc.data, bc.target          # 특성과 라벨 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
```
- 유방암 진단 데이터를 사용합니다 (악성/양성 이진 분류)
- 데이터를 80% 훈련용, 20% 테스트용으로 나눕니다

2.모델 훈련 및 예측
```python
clf = LogisticRegression(lr=0.01)  # 학습률 0.01로 모델 생성
clf.fit(X_train,y_train)           # 모델 훈련
y_pred = clf.predict(X_test)       # 테스트 데이터 예측
```

3.정확도 계산
```python
def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)
```
- 예측이 맞은 개수를 전체 개수로 나누어 정확도를 계산합니다

* 핵심 개념 정리
1. **로지스틱 회귀는 분류 알고리즘**입니다 - 연속적인 값이 아닌 카테고리를 예측합니다
2. **시그모이드 함수**로 확률을 계산하여 0과 1 사이 값을 만듭니다
3. **경사하강법**으로 오차를 점진적으로 줄여나갑니다
4. **학습률**이 중요합니다 - 너무 크면 발산하고, 너무 작으면 학습이 느립니다

### 3.3 scikit-learn을 사용한 로지스틱 회귀 구현

**1. 데이터 준비**
- `train_test_split`에서 `stratify=y` 옵션으로 클래스 비율을 유지합니다
- `StandardScaler`로 특성들을 표준화합니다 (평균 0, 표준편차 1)

    ```python
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn import datasets
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. 데이터 로드
    print("=== 데이터 로드 ===")
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    print(f"데이터 형태: {X.shape}")
    print(f"특성 수: {X.shape[1]}")
    print(f"클래스 분포: 악성={np.sum(y==0)}, 양성={np.sum(y==1)}")

    # 2. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n훈련 데이터: {X_train.shape}")
    print(f"테스트 데이터: {X_test.shape}")

    # 3. 특성 스케일링 (선택사항이지만 성능 향상에 도움)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```
**2. 모델 생성**
- `LogisticRegression` 클래스를 바로 사용합니다
- `max_iter=1000`으로 충분한 반복 횟수를 보장합니다
- `random_state=42`로 재현 가능한 결과를 만듭니다
    ```python
    # 4. 로지스틱 회귀 모델 생성 및 훈련
    print("\n=== 모델 훈련 ===")
    model = LogisticRegression(
        random_state=42,
        max_iter=1000  # 수렴을 위한 최대 반복 횟수
    )
    ```
**3. 훈련과 예측**
- `.fit()`으로 모델을 훈련시킵니다
- `.predict()`로 클래스를 예측합니다
- `.predict_proba()`로 각 클래스의 확률을 구합니다
    ```python
    # 스케일링된 데이터로 훈련
    model.fit(X_train_scaled, y_train)
    print("모델 훈련 완료!")

    # 5. 예측
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)  # 확률 예측
    ```

**4. 성능 평가**
- `accuracy_score`로 정확도를 계산합니다
- `classification_report`로 정밀도, 재현율, F1-score를 확인합니다
- `confusion_matrix`로 혼동행렬을 만듭니다
    ```python
    # 6. 모델 평가
    print("\n=== 모델 성능 평가 ===")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"정확도: {accuracy:.4f}")

    # 상세한 분류 리포트
    print("\n분류 리포트:")
    print(classification_report(y_test, y_pred, target_names=['악성', '양성']))
    ```

**5. 시각화**
- 혼동행렬을 히트맵으로 표시합니다
- 예측 확률의 분포를 히스토그램으로 보여줍니다
    ```python
    # 7. 혼동 행렬 시각화
    plt.figure(figsize=(12, 5))

    plt.rcParams['font.family'] = 'AppleGothic'  # Mac 기본 한글 폰트
    plt.rcParams['axes.unicode_minus'] = False   # 음수 기호 깨짐 방지

    # 혼동 행렬
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['악성', '양성'], 
                yticklabels=['악성', '양성'])
    plt.title('혼동 행렬')
    plt.ylabel('실제')
    plt.xlabel('예측')

    # 확률 분포
    plt.subplot(1, 2, 2)
    plt.hist(y_pred_proba[:, 1], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('양성 클래스 예측 확률 분포')
    plt.xlabel('예측 확률')
    plt.ylabel('빈도')
    plt.axvline(x=0.5, color='red', linestyle='--', label='결정 경계 (0.5)')
    plt.legend()

    plt.tight_layout()
    plt.show()
    ```

**6. 특성 중요도**
- 모델의 계수(`coef_`)를 분석하여 어떤 특성이 중요한지 확인합니다
    ```python
    # 8. 특성 중요도 분석 (계수의 절댓값으로 판단)
    print("\n=== 특성 중요도 ===")
    feature_importance = pd.DataFrame({
        '특성명': bc.feature_names,
        '계수': model.coef_[0],
        '절댓값': np.abs(model.coef_[0])
    }).sort_values('절댓값', ascending=False)

    print("상위 10개 중요 특성:")
    print(feature_importance.head(10))
    ```
**7. 실제 활용**
- 새로운 데이터에 대한 예측 예제를 포함합니다
- 확률과 최종 분류 결과를 모두 보여줍니다

    ```python
    # 9. 새로운 데이터 예측 예제
    print("\n=== 새로운 데이터 예측 ===")
    # 테스트 데이터의 첫 번째 샘플 사용
    new_sample = X_test_scaled[0:1]  # 2D 배열 형태 유지
    prediction = model.predict(new_sample)[0]
    probability = model.predict_proba(new_sample)[0]

    print(f"예측 결과: {'양성' if prediction == 1 else '악성'}")
    print(f"각 클래스별 확률:")
    print(f"  악성 확률: {probability[0]:.4f}")
    print(f"  양성 확률: {probability[1]:.4f}")
    print(f"실제 정답: {'양성' if y_test[0] == 1 else '악성'}")

    # 10. 모델 파라미터 확인
    print("\n=== 모델 파라미터 ===")
    print(f"절편(bias): {model.intercept_[0]:.4f}")
    print(f"계수 개수: {len(model.coef_[0])}")
    print(f"수렴 여부: {model.n_iter_[0]}번 반복 후 수렴")
```