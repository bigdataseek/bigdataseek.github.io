---
title: 1차시 11(빅데이터 분석):Machine Learning 개요
layout: single
classes: wide
categories:
  - Machine Learning
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

# 1. 머신러닝 개요

## **1. 머신러닝 개요**

### **1.1 머신러닝이란?**
- **머신러닝의 정의와 중요성**:
  - 데이터에서 패턴을 학습하고 이를 활용해 예측하거나 결정하는 기술.
  - 전통적인 프로그래밍은 명확한 규칙을 기반으로 동작하지만, 머신러닝은 데이터를 통해 규칙을 자동으로 학습.
- **전통적인 프로그래밍과 머신러닝의 차이점**:
  - 전통적 프로그래밍: 입력 + 규칙 → 출력
  - 머신러닝: 입력 + 출력 → 규칙
- **머신러닝의 역사적 발전과 현재 동향**:
  - 1950년대: Alan Turing의 "컴퓨터가 생각할 수 있을까?" 질문.
  - 2000년대 이후: 빅데이터와 딥러닝의 발전.
- **실생활 응용 사례 소개**:
  - 의료 진단(암 분류), 추천 시스템(넷플릭스), 자연어 처리(번역), 자율주행 등.

---

### **1.2 머신러닝의 유형**
- **지도학습 (Supervised Learning)**:
  - **분류(Classification)**: 스팸 메일 분류, 이미지 분류.
  - **회귀(Regression)**: 집값 예측, 주식 가격 예측.
- **비지도학습 (Unsupervised Learning)**:
  - **군집화(Clustering)**: 고객 세그먼테이션, 문서 그룹화.
  - **차원 축소(Dimensionality Reduction)**: 데이터 시각화, 노이즈 제거.
- **강화학습 (Reinforcement Learning)**:
  - 게임 AI, 로봇 제어.
- **준지도학습 (Semi-supervised Learning)**:
  - 일부 레이블이 있는 데이터를 활용.


### **1.3 머신러닝 워크플로우**
1. **문제 정의**: 해결하려는 문제를 명확히 정의.
2. **데이터 수집 및 준비**: 데이터 수집, 결측치 처리, 이상치 제거.
3. **특성 추출 및 선택**: 데이터에서 중요한 특성 추출.
4. **모델 선택 및 학습**: 알고리즘 선택 및 하이퍼파라미터 설정.
5. **모델 평가 및 최적화**: 성능 지표를 사용해 모델 평가.
6. **모델 배포 및 모니터링**: 실제 환경에서 모델 적용.

## **2.주요 머신러닝 알고리즘**

### **2.1 지도학습 알고리즘**
- **선형 회귀 (Linear Regression)**:
  - 기본 원리: 입력 변수와 출력 변수 간 선형 관계를 모델링.
  - 손실 함수: MSE(Mean Squared Error).
  - 최적화: 경사하강법(Gradient Descent).
- **로지스틱 회귀 (Logistic Regression)**:
  - 이진 분류와 다중 분류 가능.
  - 활성화 함수: 시그모이드(Sigmoid).
- **결정 트리 (Decision Trees)**:
  - 정보 이득, 엔트로피, 지니 계수를 사용해 분할.
- **앙상블 기법**:
  - **랜덤 포레스트**: 여러 결정 트리를 조합.
  - **그래디언트 부스팅**: 순차적으로 모델을 강화.



### **2.2 비지도학습 알고리즘**
- **K-평균 군집화 (K-means Clustering)**:
  - 알고리즘 원리: 데이터를 K개의 클러스터로 나눔.
  - 최적의 K 선택: Elbow Method.
- **주성분 분석 (PCA)**:
  - 차원 축소의 필요성: 데이터 시각화, 연산 효율성.
  - PCA의 수학적 직관: 공분산 행렬의 고유값 분해.


### **2.3 모델 평가 방법**
- **분류 평가 지표**:
  - 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1 점수.
  - 혼동 행렬(Confusion Matrix).
  - ROC 곡선과 AUC.
- **회귀 평가 지표**:
  - MAE(Mean Absolute Error), MSE(Mean Squared Error), RMSE, R-squared.
- **교차 검증 (Cross-validation)**:
  - K-Fold Cross Validation.

## **3.머신러닝 실무 고려사항**

### **3.1 특성 공학 (Feature Engineering)**
- **특성 선택의 중요성**: 관련 없는 특성 제거.
- **범주형 변수 처리**: One-Hot Encoding, Label Encoding.
- **스케일링과 정규화**: Min-Max Scaling, Standardization.
- **결측치와 이상치 처리**: 평균 대체, 이상치 제거.

### **3.2 과적합과 과소적합**
- **과적합/과소적합의 이해**:
  - 과적합: 학습 데이터에 너무 맞춰짐.
  - 과소적합: 데이터 패턴을 충분히 학습하지 못함.
- **편향-분산 트레이드오프**:
  - 편향(Bias): 모델의 단순성.
  - 분산(Variance): 모델의 복잡성.
- **정규화 기법**: L1(Lasso), L2(Ridge).
- **조기 종료(Early Stopping)**: 학습 중단.


### **3.3 하이퍼파라미터 튜닝**
- **그리드 서치(Grid Search)**: 모든 가능한 조합 탐색.
- **랜덤 서치(Random Search)**: 무작위로 조합 탐색.
- **베이지안 최적화(Bayesian Optimization)**: 확률적 접근.


## **4.실습 - Python을 활용한 머신러닝**

### **4.1 환경 설정 및 라이브러리 소개**

```python
# 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
```

### **4.2 지도학습 실습**

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

print(f"X.shape(): {X.shape}, y.shape(): {y.shape}")
print(f"X.head() : \n {X[:10]}")
print(f"feature_names: \n{iris.feature_names}")
print(f"target_name: \n{iris.target_names}")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 결정 트리 모델 학습
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# 랜덤 포레스트 모델 학습
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
```


### **4.3 비지도학습 실습**
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# K-Means 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# PCA를 통한 시각화
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.title("K-Means Clustering with PCA")
plt.show()
```

---

# 2. Scikit-learn 

![scikit-learn logo](/assets/images/Scikit_learn_logo_small.svg)
[Scikit-learn의 Getting_Started 페이지](https://scikit-learn.org/stable/getting_started.html)

## **1. Estimator Basics (모델 학습 및 예측)**
- **핵심 개념**: 
  - Scikit-learn의 모든 모델은 `Estimator` 객체로 구현됩니다.(추정기)
  - `fit()` 메서드를 사용해 데이터에 모델을 학습시키고, `predict()` 메서드로 새로운 데이터를 예측합니다.
- **예제 코드**:
  ```python
  from sklearn.ensemble import RandomForestClassifier

  clf = RandomForestClassifier(random_state=0)
  X = [[1, 2, 3], [11, 12, 13]]  # 샘플 데이터
  y = [0, 1]  # 타겟 레이블
  clf.fit(X, y)  # 모델 학습
  clf.predict([[4, 5, 6], [14, 15, 16]])  # 새로운 데이터 예측
  ```
- **설명 포인트**:
  - `X`: 입력 데이터(특성 행렬), `(n_samples, n_features)` 형태.
  - `y`: 출력 데이터(타겟 값), 분류 문제에서는 클래스 레이블, 회귀 문제에서는 실수 값.
  - 학습된 모델은 새로운 데이터를 예측할 때 다시 학습할 필요가 없습니다.


## **2. Transformers and Pre-processors (데이터 전처리)**
### **2.1. 핵심 개념**:
  - 데이터 전처리는 머신러닝 파이프라인에서 중요한 단계입니다.
  - Scikit-learn은 데이터 변환을 위한 `Transformer` 객체를 제공하며, `fit()`과 `transform()` 메서드를 사용합니다.
  - **예제 코드**:
    ```python
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = [[0, 15], [1, -10]]
    scaled_X = scaler.fit_transform(X)  # 데이터 표준화
    print(scaled_X)
    ```
  - **설명 포인트**:
    - `StandardScaler`: 데이터를 평균 0, 분산 1로 표준화합니다.
    - 전처리는 데이터의 스케일을 맞추거나 결측치를 처리하는 데 사용됩니다.

### **2.2. fit(), transform(), fit_transform()**:
- fit(): 입력 데이터(X)를 분석하여 필요한 통계적 정보(예: 평균, 표준편차)를 계산
  ```python
  from sklearn.preprocessing import StandardScaler

  scaler = StandardScaler()
  X = [[0, 15], [1, -10]]
  scaler.fit(X)  # 데이터의 평균과 표준편차를 계산
  print(scaler.mean_)  # 각 열의 평균
  print(scaler.scale_)  # 각 열의 표준편차
  ```

- transform(): `fit()`으로 계산된 통계적 정보를 기반으로 데이터를 변환.
  ```python
  scaled_X = scaler.transform(X)  # 데이터를 표준화
  print(scaled_X)
  ```

- fit_transform(): 데이터의 통계적 특성을 계산(`fit`)하고, 이를 즉시 데이터에 적용(`transform`), 주로 훈련데이터에 사용

  ```python
  scaled_X = scaler.fit_transform(X)  # fit()과 transform()을 동시에 수행
  print(scaled_X)
  ```

- fit()은 데이터를 분석하고, transform()은 데이터를 변환합니다. fit_transform()은 이 두 과정을 한 번에 수행합니다. 훈련 데이터에서는 fit_transform()을 사용하고, 테스트 데이터에서는 transform()만 사용해야 한다.

### **2.3. ColumnTransformer()**:
- 머신러닝 작업에서는 데이터의 각 특성(컬럼)이 서로 다른 형태와 스케일을 가질 수 있습니다. 
  - 수치형 데이터 : 표준화(Standardization)나 정규화(Normalization)가 필요.
  - 범주형 데이터 : 원-핫 인코딩(One-Hot Encoding)이나 레이블 인코딩(Label Encoding)이 필요.
  - 텍스트 데이터 : 토큰화(Tokenization), 벡터화(Vectorization) 등이 필요.
- 서로 다른 특성에 대해 다른 전처리를 적용하려면 `ColumnTransformer`를 사용합니다.

  ```python
  import pandas as pd
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import StandardScaler, OneHotEncoder
  from sklearn.pipeline import Pipeline
  from sklearn.linear_model import LogisticRegression

  # 샘플 데이터 생성
  data = {
      'age': [25, 45, 35, 50],
      'salary': [50000, 100000, 70000, 120000],
      'gender': ['male', 'female', 'female', 'male'],
      'city': ['New York', 'Paris', 'London', 'Tokyo']
  }
  df = pd.DataFrame(data)
  X = df.drop('city', axis=1)  # 입력 데이터
  y = df['city']              # 타겟 데이터

  # ColumnTransformer 정의
  # ColumnTransformer는 서로 다른 유형의 피처(수치형, 범주형 등)에 대해 각기 다른 전처리 방법을 적용할 수 있도록 해줍니
  preprocessor = ColumnTransformer( #각 튜플은 (이름, 변환기, 적용할 컬럼)으로 정의
      transformers=[
          ('num', StandardScaler(), ['age', 'salary']),  # 수치형 데이터: 표준화
          ('cat', OneHotEncoder(), ['gender'])           # 범주형 데이터: 원-핫 인코딩
      ]
  )

  # 파이프라인 생성
  # Pipeline은 전처리 단계(예: StandardScaler, OneHotEncoder)와 모델 학습(예: LogisticRegression)을 하나의 객체로 통합
  # Pipeline은 학습 데이터와 테스트 데이터에 대해 전처리 과정을 동일하게 적용하므로, 데이터 누수(data leakage)를 방지
  # Pipeline을 사용하면 전처리와 모델 파라미터를 함께 GridSearchCV나 RandomizedSearchCV로 튜닝
  # 학습된 Pipeline 객체는 전처리와 모델을 하나로 묶어 저장할 수 있어, 프로덕션 환경에서 예측 시 전처리 과정을 별도로 관리할 필요가 없습니다. 예를 들어, joblib로 저장 후 바로 배포 가능

  pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('classifier', LogisticRegression())
  ])

  # 모델 학습
  pipeline.fit(X, y)

  # 새로운 데이터 예측
  new_data = pd.DataFrame({
      'age': [30],
      'salary': [80000],
      'gender': ['female']
  })
  print(pipeline.predict(new_data))
  ```


## **3. Pipelines: Chaining Pre-processors and Estimators (파이프라인)**
### **3.1. 핵심 개념**:
  - 여러 단계(전처리 + 모델 학습)를 하나의 파이프라인으로 연결하여 간단히 관리할 수 있습니다.
  - 파이프라인은 데이터 누출(Data Leakage)을 방지하는 데 유용합니다.
  - **예제 코드**:
    ```python
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) #default로 test_size=0.25
    pipe.fit(X_train, y_train)  # 파이프라인 학습
    print(accuracy_score(pipe.predict(X_test), y_test))  # 정확도 평가
    ```
  - **설명 포인트**:
    - 파이프라인은 전처리와 모델 학습을 하나의 객체로 묶어줍니다.
    - 데이터 누출(Data Leakage): 테스트 데이터 정보가 훈련 데이터에 유출되는 것을 방지합니다.

### **3.2. Pipeline()과 make_pipeline()**
**1. `Pipeline()`**
- **특징**:
  - 명시적으로 각 단계의 이름을 지정해야 합니다.
  - 각 단계는 `(이름, 객체)` 형태의 튜플로 정의됩니다.
- **사용 예시**:
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression

  pipeline = Pipeline(steps=[
      ('scaler', StandardScaler()),          # 첫 번째 단계: 표준화
      ('classifier', LogisticRegression())  # 두 번째 단계: 분류 모델
  ])
  ```
  - 여기서 `'scaler'`와 `'classifier'`는 각 단계의 이름입니다.
  - 이름은 임의로 지정할 수 있지만, 중복되지 않아야 합니다.

**2. `make_pipeline()`**
  - **특징**:
    - 단계의 이름을 자동으로 생성합니다.
    - 단순히 객체를 순서대로 나열하면 됩니다.
  - **사용 예시**:
    ```python
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    pipeline = make_pipeline(
        StandardScaler(),          # 첫 번째 단계: 표준화
        LogisticRegression()      # 두 번째 단계: 분류 모델
    )
    ```
    - 여기서 각 단계의 이름은 자동으로 생성됩니다. 예를 들어:
      - `StandardScaler` → `'standardscaler'`
      - `LogisticRegression` → `'logisticregression'`

**3.차이점** 
- `Pipeline()`은 각 단계의 이름을 직접 지정할 수 있어 가독성과 참조가 쉽지만, 코드가 조금 더 복잡합니다. 반면에 `make_pipeline()`은 이름을 자동으로 생성해 주기 때문에 간단하고 빠르게 파이프라인을 만들 수 있습니다. 간단한 작업에는 `make_pipeline()`을, 복잡한 작업이나 디버깅이 필요한 경우에는 `Pipeline()`을 사용


## **4. Model Evaluation (모델 평가)**
- **핵심 개념**:
  - 모델의 성능은 반드시 테스트 데이터를 통해 평가해야 합니다.
  - 교차 검증(Cross-validation)은 모델의 일반화 성능을 평가하는 데 유용합니다.
- **예제 코드**:
  ```python
  from sklearn.datasets import make_regression
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import cross_validate

  X, y = make_regression(n_samples=1000, random_state=0) 
  # X는 독립변수로 100개의 feature를 갖는다,2D 배열 형태
  # y는 종속변수로 1D 배열 형태

  lr = LinearRegression()
  result = cross_validate(lr, X, y)  # 5-fold 교차 검증
  print(result['test_score'])  # 각 폴드의 점수 출력
  ```
- **설명 포인트**:
  - `make_regression()`은  회귀(Regression) 문제를 위한 가상의 데이터셋을 생성
  - `cross_validate()`은 기본값이 5-fold
  - 교차 검증은 데이터를 여러 개의 폴드로 나누고, 각 폴드를 번갈아가며 테스트 세트로 사용합니다.
  - 이를 통해 모델의 안정성을 평가할 수 있습니다.
  - 회귀 모델의 경우, scoring 파라미터가 지정되지 않으면 기본적으로 R² 점수 (결정 계수, coefficient of determination)
  - train_test_split을 하지 않은 이유:
    - train_test_split은 데이터를 한 번만 나누므로, 특정 분할에 따라 모델 성능이 달라질 수 있습니다(데이터 분포의 편향 가능성).
    - 반면, 교차 검증은 데이터를 여러 폴드로 나누어 평가하므로, 모델의 일반화 성능(새로운 데이터에 대한 성능)을 더 안정적으로 측정할 수 있습니다.


## **5. Automatic Parameter Searches (자동 하이퍼파라미터 탐색)**
### **5.1. 핵심 개념**:
  - 데이터 전처리 → 모델 학습 → 모델 평가 → 하이퍼파라미터 튜닝.
  - 모델의 성능은 하이퍼파라미터에 크게 의존합니다.
  - Scikit-learn은 자동으로 최적의 하이퍼파라미터를 찾는 도구를 제공합니다.
  - **RandomizedSearchCV**:
    ```python
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from scipy.stats import randint

    param_distributions = {'n_estimators': randint(1, 5), 'max_depth': randint(5, 10)}
    #randint()는 분포 객체로, 동적으로 무작위 샘플링을 수행하며 RandomizedSearchCV와 자연스럽게 호환.
    search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
                                n_iter=5,
                                param_distributions=param_distributions,
                                random_state=0)
    search.fit(X_train, y_train)  # 최적 파라미터 탐색
    print(search.best_params_)  # 최적 파라미터 출력
    # max_depth=9와 n_estimators=4가 가장 높은 교차 검증 R² 점수를 기록.
    print(search.score(X_test, y_test))  # 테스트 데이터 점수
    ```
  - **설명 포인트**:
    - `RandomizedSearchCV`: 무작위로 하이퍼파라미터 조합을 탐색합니다.
    - 최적의 파라미터를 찾으면 해당 설정으로 모델이 학습됩니다.

### **5.2. GridSearchCV**
- **특징**:
  - 모든 가능한 하이퍼파라미터 조합을 체계적으로 탐색합니다.
  - "완전 탐색(Exhaustive Search)" 방식으로, 지정된 파라미터 그리드의 모든 경우를 시도합니다.
  - 따라서 매우 정확한 결과를 제공하지만, 계산 비용이 매우 큽니다.

- **사용 예시**:
  ```python
  from sklearn.model_selection import GridSearchCV
  from sklearn.ensemble import RandomForestClassifier

  # 하이퍼파라미터 그리드 정의
  param_grid = {
      'n_estimators': [10, 50, 100],
      'max_depth': [None, 10, 20]
  }

  # GridSearchCV 객체 생성
  grid_search = GridSearchCV(
      estimator=RandomForestClassifier(random_state=42),
      param_grid=param_grid,
      cv=5,  # 5-fold 교차 검증
      scoring='accuracy'
  )

  # 데이터 로드 및 학습
  X, y = ...  # 데이터셋 로드
  grid_search.fit(X, y)

  # 최적의 파라미터와 점수 출력
  print("Best Parameters:", grid_search.best_params_)
  # Best Parameters: {'max_depth': None, 'n_estimators': 100}
  print("Best Score:", grid_search.best_score_)
  ```

### **5.3. 어떤 경우에 사용할까?**
1. **`GridSearchCV`**:
   - 하이퍼파라미터 공간이 작고, 모든 조합을 체계적으로 탐색하고 싶을 때.
   - 예: `n_estimators=[10, 50, 100]`, `max_depth=[5, 10, 15]`처럼 제한된 범위

2. **`RandomizedSearchCV`**:
   - 하이퍼파라미터 공간이 크거나 연속적인 값(예: 실수 범위)을 포함할 때.
   - 예: `n_estimators=10~200`, `max_depth=None 또는 10~50`처럼 넓은 범위
