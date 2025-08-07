---
title: 1차시 13(빅데이터 분석):Machine Learning 2
layout: single
classes: wide
categories:
  - Machine Learning
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---


# 3. 지도 학습
![지도학습](/assets/images/supervised.png)


- 지도학습(Supervised Learning)은 입력 데이터와 출력 데이터가 주어졌을 때, 입력과 출력 간의 관계를 학습하는 방법입니다. 이를 통해 새로운 입력 데이터에 대해 정확한 출력을 예측할 수 있습니다. 



## **1. 회귀 (Regression)**

### **1.1 Simple Linear Regression**
**핵심 개념**:  
단일 독립 변수(x)와 종속 변수(y) 간의 선형 관계를 모델링합니다. `y = wx + b` 형태로 표현됩니다.

**샘플 예제**:
```python
import numpy as np
from sklearn.linear_model import LinearRegression

# 데이터 생성
# scikit-learn은 모든 입력 X를 "특성 행렬"로 간주하며, 그 형태는 반드시(n_samples, n_features) 이어야 

X = np.array([[1], [2], [3], [4], [5]])  # 독립 변수
y = np.array([2, 4, 6, 8, 10])           # 종속 변수

# 모델 학습
model = LinearRegression()
model.fit(X, y)

# 예측
print("기울기(w):", model.coef_[0])
print("절편(b):", model.intercept_)
print("예측값:", model.predict([[6]]))  # x=6일 때 y 예측
```

### **1.2 Multiple Linear Regression**
**핵심 개념**:  
여러 개의 독립 변수(x1, x2, ...)와 종속 변수(y) 간의 선형 관계를 모델링합니다. `y = w1x1 + w2x2 + ... + b` 형태로 표현됩니다.

**샘플 예제**:
```python
from sklearn.linear_model import LinearRegression

# 데이터 생성
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # 다중 독립 변수
y = np.array([3, 5, 7, 9])                      # 종속 변수

# 모델 학습
model = LinearRegression()
model.fit(X, y)

# 예측
print("계수(w):", model.coef_)
print("절편(b):", model.intercept_)
print("예측값:", model.predict([[5, 6]]))  # x1=5, x2=6일 때 y 예측
```


### **1.3 Polynomial Regression**
**핵심 개념**:  
독립 변수를 다항식으로 변환하여 비선형 관계를 모델링합니다. `y = w1x + w2x^2 + ... + b` 형태로 표현됩니다.

**샘플 예제**:
```python
from sklearn.preprocessing import PolynomialFeatures # 입력 변수를 다항식 항으로 확장
from sklearn.linear_model import LinearRegression

# 데이터 생성
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  # y = x^2 관계

# 다항식 변환
# 입력 X에 대해 상수항, 1차항, 2차항을 만들어 새로운 특성 행렬로 변환.
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 모델 학습
model = LinearRegression()
model.fit(X_poly, y)

# 예측
print("계수(w):", model.coef_)
print("절편(b):", model.intercept_)
print("예측값:", model.predict(poly.transform([[6]])))  # x=6일 때 y 예측
```
```python
계수(w): [ 0.0000000e+00 -1.0061107e-14  1.0000000e+00]
절편(b): 8.881784197001252e-15
예측값: [36.]
```


### **1.4 Lasso Regression**
**핵심 개념**:  
L1 정규화를 사용하여 계수의 크기를 제한하고, 일부 계수를 0으로 만듭니다. 특성 선택(feature selection)에 유용합니다.

* 정규화란
    - 모델이 학습 데이터에만 지나치게 맞춰지는 것(과적합)을 막고,
    -  더 일반적이고 자연스러운 패턴을 학습하게 만든다는 의미에서 "정상적인 학습 상태로 조정한다"고 해서 정규화라고 부릅니다.

* 정규화의 강도를 조절하는 하이퍼파라미터.
    - alpha = 0 → 일반 선형 회귀 (정규화 없음)
    - alpha가 클수록 계수를 0에 가깝게 강하게 제약
    - alpha=0.1은 약한 정규화

**샘플 예제**:
```python
from sklearn.linear_model import Lasso

# 데이터 생성
# 예: 불필요한 특성이 많은 경우
X = np.array([[1, 2, 3, 100],
              [2, 4, 6, 200],
              [3, 6, 9, 300],
              [4, 8, 12, 400],
              [5, 10, 15, 500]])
y = np.array([2, 4, 6, 8, 10])  # y = 2*x1

# 모델 학습
model = Lasso(alpha=0.1)  # alpha: 정규화 강도
model.fit(X, y)

# 예측
print("계수(w):", model.coef_)
print("절편(b):", model.intercept_)
print("예측값:", model.predict([[6,12,18,600]]))
```
```python
계수(w): [0.       0.       0.       0.019995]
절편(b): 0.0015000000000036096
예측값: [11.9985]
```

### **1.5 Ridge Regression**
**핵심 개념**:  
L2 정규화를 사용하여 계수의 크기를 제한합니다. 과적합(overfitting)을 방지하는 데 효과적입니다.

**샘플 예제**:
```python
from sklearn.linear_model import Ridge

# 데이터 생성
# 예: 다중공선성 있는 데이터
X = np.array([[1, 2.1],
              [2, 3.9],
              [3, 6.1],
              [4, 7.9],
              [5, 10.1]])  # x2 ≈ 2*x1 → 거의 완전한 상관
y = np.array([2, 4, 6, 8, 10])

# 모델 학습
model = Ridge(alpha=0.1)  # alpha: 정규화 강도
model.fit(X, y)

# 예측
print("계수(w):", model.coef_)
print("절편(b):", model.intercept_)
print("예측값:", model.predict([[6,11.9]]))
```


## **2. 분류 (Classification)**

### **2.1 Logistic Regression**
**핵심 개념**:  
이진 분류 문제에서 확률을 기반으로 클래스를 예측합니다. 시그모이드 함수를 사용합니다.

**샘플 예제**:
```python
from sklearn.linear_model import LogisticRegression

# 데이터 생성
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])  # 이진 분류

# 모델 학습
model = LogisticRegression()
model.fit(X, y)

# 예측
print("예측 클래스:", model.predict([[6]]))
print("예측 확률:", model.predict_proba([[6]]))
```


### **2.2 K-Nearest Neighbors (KNN)**
**핵심 개념**:  
새로운 데이터 포인트를 가장 가까운 k개의 이웃 데이터 포인트로 분류합니다.

**샘플 예제**:
```python
from sklearn.neighbors import KNeighborsClassifier

# 데이터 생성
X = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]])
y = np.array([0, 0, 0, 1, 1])

# 모델 학습
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# 예측
print("예측 클래스:", model.predict([[5, 6]]))
```


### **2.3 Support Vector Machine (SVM)**
**핵심 개념**:  
데이터를 최대 마진으로 분리하는 초평면(hyperplane)을 찾습니다.SVR(회귀)과 SVC(분류)이 있음

**샘플 예제**:
```python
from sklearn.svm import SVC

# 데이터 생성
X = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]])
y = np.array([0, 0, 0, 1, 1])

# 모델 학습
model = SVC(kernel='linear')
model.fit(X, y)

# 예측
print("예측 클래스:", model.predict([[5, 6]]))
```


### **2.4 Kernel SVM**
**핵심 개념**:  
비선형 데이터를 고차원 공간으로 매핑하여 선형 분리가 가능하도록 합니다.

**샘플 예제**:
```python
from sklearn.svm import SVC

# 데이터 생성
X = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]])
y = np.array([0, 0, 0, 1, 1])

# 모델 학습
# RBF 커널 사용,Radial Basis Function 커널 (가우시안 커널이라고도 함)
model = SVC(kernel='rbf')  
model.fit(X, y)

# 예측
print("예측 클래스:", model.predict([[5, 6]]))
```

## **3. 앙상블 (Ensemble)**

### **3.1 Decision Tree**
**핵심 개념**:  
데이터를 규칙 기반으로 분할하여 결정 트리를 구성합니다.

**샘플 예제**:
```python
from sklearn.tree import DecisionTreeClassifier

# 데이터 생성
X = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]])
y = np.array([0, 0, 0, 1, 1])

# 모델 학습
model = DecisionTreeClassifier()
model.fit(X, y)

# 예측
print("예측 클래스:", model.predict([[5, 6]]))
```


### **3.2 Random Forest**
**핵심 개념**:  
여러 개의 결정 트리를 앙상블하여 일반화 성능을 향상시킵니다.

**샘플 예제**:
```python
from sklearn.ensemble import RandomForestClassifier

# 데이터 생성
X = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]])
y = np.array([0, 0, 0, 1, 1])

# 모델 학습
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

# 예측
print("예측 클래스:", model.predict([[5, 6]])) # 내부적으로 np.array로 처리됨
```


### **3.3 XGBoost**
**핵심 개념**:  
Gradient Boosting 알고리즘을 사용하여 순차적으로 모델을 학습합니다.

**샘플 예제**:
```python
from xgboost import XGBClassifier

# 데이터 생성
X = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]])
y = np.array([0, 0, 0, 1, 1])

# 모델 학습
model = XGBClassifier()
model.fit(X, y)

# 예측
print("예측 클래스:", model.predict([[5, 6]]))
```


### **3.4 CatBoost**
**핵심 개념**:  
범주형 데이터 처리에 특화된 Gradient Boosting 알고리즘입니다.

**샘플 예제**:
```python
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

# ▶️ 샘플 데이터 생성 (범주형 + 수치형 혼합)
X = [
    ['남자', '고졸', 25],
    ['여자', '대졸', 30],
    ['여자', '대졸', 45],
    ['남자', '대졸', 35],
    ['여자', '고졸', 22],
    ['남자', '고졸', 28],
]
y = [0, 1, 1, 1, 0, 0]  # 예: 구매 여부 (0: 미구매, 1: 구매)

# CatBoost는 문자열 범주형도 그대로 학습 가능
# 수치형/범주형 혼합일 경우, 범주형 열의 인덱스를 명시
cat_features = [0, 1]  # 0: 성별, 1: 학력

# 학습용 배열로 변환
X = np.array(X)

# 모델 생성 및 학습
model = CatBoostClassifier(silent=True)
model.fit(X, y, cat_features=cat_features)

# 예측
pred = model.predict([['여자', '고졸', 27]])
print("예측 클래스:", pred)

```

## **4. 딥러닝 (Deep Learning)**

### **4.1 Artificial Neural Network (ANN)**
**핵심 개념**:  
- 신경망을 통해 복잡한 비선형 관계를 학습합니다.
- ANN은 신경망의 가장 일반적인 형태이자 기반 개념, 다음 세 가지 유형의 층(layer)으로 구성
    *   입력 층 (Input Layer): 외부 데이터를 모델에 주입하는 부분입니다.
    *   은닉 층 (Hidden Layer): 입력 층과 출력 층 사이에 위치하며, 입력 데이터를 가공하고 복잡한 특징을 추출하는 역할을 합니다. 하나 이상의 은닉 층을 가질 수 있습니다.
    *   출력 층 (Output Layer): 모델의 최종 예측 결과를 생성하는 부분입니다.  
- 각 층은 여러 개의 뉴런(neuron) 또는 **노드(node)**로 구성되어 있으며, 이 뉴런들은 서로 연결되어 있고 각 연결은 **가중치(weight)**를 가집니다. 뉴런은 입력값과 가중치를 곱하고 편향(bias)을 더한 후, **활성화 함수(activation function)**를 통과시켜 다음 층으로 전달합니다.



**샘플 예제**:
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# 데이터 생성
X = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]])
y = np.array([0, 0, 0, 1, 1])

# 모델 정의
model = Sequential([
    Dense(4, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=0)

# 예측
new_data = np.array([[5, 6]])
pred_prob = model.predict(new_data)
pred_class = (pred_prob > 0.5).astype(int)

print("예측 확률:", pred_prob)
print("예측 클래스:", pred_class)
```


### **4.2 Convolutional Neural Network (CNN)**
**핵심 개념**:  
이미지 데이터를 처리하기 위한 신경망으로, 컨볼루션 연산을 사용합니다.

**샘플 예제**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

# 데이터 생성 (간단한 이미지 데이터)
# Conv1D는 3D 입력 필요:  (배치 크기, 길이, 채널 수)
# Conv2D는 4D 입력 필요: (batch, height, width, channels)
X = np.random.rand(5, 8, 8, 1)  # 8x8 크기의 흑백 이미지 5장
y = np.array([0, 0, 1, 1, 0])

# 모델 정의
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(8, 8, 1)),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일 및 학습
# sigmoid + binary_crossentropy는 이진 분류에 적합.
# 다중 분류라면 softmax + categorical_crossentropy 사용.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, verbose=0)

# 예측
print("예측 확률:", model.predict(np.random.rand(1, 8, 8, 1)))
```


### **4.3 Recurrent Neural Network (RNN)**
**핵심 개념**:  
시계열 데이터나 순차적 데이터를 처리하기 위한 신경망입니다.

**샘플 예제**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# 데이터 생성 (시계열 데이터)
# 입력값 : (배치 크기, 타임스텝 수, 피처 수)
X = np.random.rand(5, 10, 1)  # 10개의 타임스텝, 1개의 피처
y = np.array([0, 1, 0, 1, 0])

# 모델 정의
model = Sequential([
    SimpleRNN(10, activation='relu', input_shape=(10, 1)),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, verbose=0)

# 예측
print("예측 확률:", model.predict(np.random.rand(1, 10, 1)))
```

---

# 4. 비지도 학습
## **1. 비지도학습 (Unsupervised Learning)**
비지도학습은 레이블이 없는 데이터에서 패턴을 찾고, 강화학습은 에이전트가 환경과 상호작용하며 보상을 최대화하는 행동을 학습합니다.

### **1.1 클러스터링 (Clustering)**
**1. K-Means**
-   데이터를 k개의 클러스터로 그룹화하며, 각 클러스터의 중심점(centroid)과 데이터 포인트 간의 거리를 최소화합니다.

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 데이터 생성
X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]

# 모델 학습
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# 클러스터 할당
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 시각화
plt.scatter([x[0] for x in X], [x[1] for x in X], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
plt.show()
```

### **1.2 차원 축소 (Dimensionality Reduction)**

**1. Principal Component Analysis (PCA)**
-   데이터의 분산을 최대한 보존하면서 차원을 줄이는 선형 변환 기법입니다.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 데이터 생성
X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]

# PCA 적용
pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)

# 시각화
plt.scatter(X_reduced, [0] * len(X_reduced))
plt.show()
```

**2. Kernel PCA**
-   커널 함수를 사용하여 비선형 데이터를 고차원 공간으로 매핑한 후 PCA를 적용합니다.

```python
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

# 데이터 생성
X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]

# Kernel PCA 적용
kpca = KernelPCA(n_components=1, kernel='rbf')
X_reduced = kpca.fit_transform(X)

# 시각화
plt.scatter(X_reduced, [0] * len(X_reduced))
plt.show()
```

**3. Linear Discriminant Analysis (LDA)**
-   클래스 간 분산을 최대화하고 클래스 내 분산을 최소화하여 차원을 줄이는 기법입니다. 지도학습

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

# 데이터 생성
X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]
y = [0, 0, 1, 1, 0, 1]

# LDA 적용
lda = LDA(n_components=1)
X_reduced = lda.fit_transform(X, y)

# 시각화
plt.scatter(X_reduced, [0] * len(X_reduced), c=y, cmap='viridis')
plt.show()
```
