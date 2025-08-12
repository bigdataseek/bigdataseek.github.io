---
title: 2차시 5:머신러닝 From Scratch 2
layout: single
classes: wide
categories:
  - ML
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

## 4. 의사결정나무(Decision Tree)

### 4.1 의사결정나무(Decision Tree)란?
**의사결정나무**는 머신러닝에서 분류(Classification)나 회귀(Regression) 문제를 해결하는 데 사용되는 모델입니다. 마치 플로우차트처럼 작동합니다. 데이터를 여러 기준(질문)에 따라 나누어 최종적으로 결과를 예측합니다. 예를 들어, "이 동물은 고양이인가 강아지인가?"를 예측하려면 다음과 같은 질문을 던질 수 있습니다:
- "체중이 10kg 이상인가?" → 예/아니오
- "털이 긴가?" → 예/아니오

이 질문들은 데이터를 점점 더 작은 그룹으로 나누며, 최종적으로 한 가지 결론(예: "고양이")에 도달합니다. 이 과정은 나무 구조처럼 보이며, 각 질문은 "노드(Node)"이고, 분기(Branches)는 질문의 답변(예/아니오)을 나타냅니다.

* 코드 개요
    - **`DecisionTree.py`**: 의사결정나무 알고리즘을 처음부터 구현한 코드입니다. 데이터를 분할하고, 나무를 만들고, 예측하는 모든 로직이 포함되어 있습니다.
    - **`train.py`**: 의사결정나무를 실제 데이터(유방암 데이터셋)에 적용해 학습시키고, 예측 정확도를 계산하는 코드입니다.

### 4.2 `DecisionTree.py` 코드 해석
Decision Tree가 어떻게 "질문과 답"의 구조로 학습하고 예측하는지 이해
*   특히 주목할 점들:
    - 재귀적 구조: _grow_tree 함수가 자기 자신을 호출하며 트리를 만드는 방식
    - 정보 이득 계산: 엔트로피를 이용해 최적의 분할점을 찾는 방법
    - 중단 조건: 과적합을 방지하기 위한 다양한 조건들
    - 예측 과정: 트리를 따라 내려가며 최종 답을 찾는 방식

1.`Node` 클래스
```python
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature      # 어떤 특성(Feature)을 기준으로 나눌지
        self.threshold = threshold  # 그 특성의 기준값(Threshold)
        self.left = left           # 기준값 이하일 때의 하위 노드
        self.right = right         # 기준값 초과일 때의 하위 노드
        self.value = value         # 리프 노드일 경우 예측값

    def is_leaf_node(self):
        return self.value is not None  # 값이 있으면 리프 노드
```

- **역할**: `Node` 클래스는 의사결정나무의 각 노드를 나타냅니다. 나무는 여러 노드로 구성되며, 각 노드는 데이터를 분할하거나 최종 예측값을 가집니다.
- **속성**:
  - `feature`: 데이터를 나눌 때 사용할 특성(예: "체중").
  - `threshold`: 그 특성의 기준값(예: "10kg").
  - `left`와 `right`: 기준값에 따라 데이터가 나뉘었을 때의 하위 노드(왼쪽/오른쪽 가지).
  - `value`: 이 노드가 최종 예측값을 가지는 "리프 노드"라면 그 값을 저장(예: "고양이").
- **메서드**:
  - `is_leaf_node()`: 이 노드가 리프 노드인지 확인합니다. 리프 노드는 더 이상 분할하지 않고 최종 예측값을 반환합니다.

- 예시:
    - 만약 데이터가 "체중 <= 10kg"로 나뉜다면: `feature = "체중"`, `threshold = 10`, `left`는 체중 10kg 이하 데이터로 가는 노드, `right`는 체중 10kg 초과 데이터로 가는 노드.
    - 리프 노드라면 `value`가 "고양이" 또는 "강아지" 같은 최종 예측값을 가짐.

2.`DecisionTree` 클래스

이 클래스는 의사결정나무의 전체 구조를 관리합니다. 주요 메서드들을 하나씩 살펴보겠습니다.


*   2.1 `__init__` (초기화)

    ```python
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split  # 분할을 멈추는 최소 샘플 수
        self.max_depth = max_depth                  # 나무의 최대 깊이
        self.n_features = n_features                # 사용할 특성 수
        self.root = None                           # 나무의 루트 노드
    ```
    - **역할**: 의사결정나무 객체를 초기화합니다. 나무의 설정을 정의합니다.
    - `min_samples_split`: 노드를 나눌 때 필요한 최소 샘플 수(기본값: 2). 데이터가 너무 적으면 더 이상 분할하지 않음.
    - `max_depth`: 나무의 최대 깊이(기본값: 100). 너무 깊으면 과적합(Overfitting) 위험이 있음.
    - `n_features`: 사용할 특성의 수. 지정하지 않으면 데이터의 모든 특성을 사용.
    - `root`: 나무의 시작점(루트 노드). 처음에는 `None`으로 설정.
    - 예시: `max_depth=10`은 나무가 최대 10단계까지만 깊어질 수 있다는 뜻입니다. 이는 모델이 너무 복잡해지는 것을 방지합니다.


*   2.2 `fit` (학습)

    ```python
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)
    ```
    - **역할**: 학습 데이터(`X`, `y`)를 사용해 의사결정나무를 만듭니다.
    - **입력**:
        - `X`: 특성 데이터(2D 배열, 각 행은 샘플, 각 열은 특성).
        - `y`: 타겟 데이터(각 샘플의 클래스 레이블).
    - **동작**:
        - `n_features`를 설정(지정된 값이 없으면 데이터의 모든 특성을 사용).
        - `_grow_tree` 메서드를 호출해 나무를 생성하고, 루트 노드를 설정.
    - 예시: 유방암 데이터셋에서 `X`는 각 환자의 특성(종양 크기, 모양 등), `y`는 진단 결과(양성/악성)를 나타냅니다. 이 메서드는 데이터를 기반으로 나무를 만듭니다.


*   2.3. `_grow_tree` (나무 생성)

    ```python
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)
    ```

    - **역할**: 재귀적으로 나무를 생성합니다. 데이터를 분할하고, 각 분할에 대해 하위 노드를 만듭니다.
    - **중단 조건**:
        - 나무의 깊이가 `max_depth`에 도달.
        - 모든 샘플이 같은 클래스(`n_labels == 1`).
        - 샘플 수가 `min_samples_split`보다 적음.
        - 이 경우, 리프 노드를 만들고 가장 흔한 클래스(`_most_common_label`)를 예측값으로 설정.
    - **분할 과정**:
        - 무작위로 선택한 특성들(`feat_idxs`) 중에서 최적의 분할 기준(`best_feature`, `best_thresh`)을 찾음.
        - 데이터를 두 그룹(`left_idxs`, `right_idxs`)으로 나누고, 각 그룹에 대해 재귀적으로 나무를 생성.
    - **출력**: 분할 노드(`Node`) 또는 리프 노드.
    -   예시: 만약 "종양 크기 <= 2cm"가 최적의 분할 기준이라면, 데이터는 두 그룹(크기 ≤ 2cm, 크기 > 2cm)으로 나뉘고, 각 그룹에 대해 다시 분할을 시도합니다.


*   2.4. `_best_split` (최적 분할 찾기)

    ```python
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold
    ```
    - **역할**: 데이터를 가장 잘 나누는 특성과 기준값을 찾습니다.
    - **정보 이득(Information Gain)**: 분할 전후의 엔트로피(불확실성) 차이를 계산해 가장 정보 이득이 큰 분할을 선택합니다. 값이 클수록 더 깔끔하게 분할된 것
    - **동작**:
        - 각 특성(`feat_idx`)과 그 특성의 고유한 값(`thresholds`)을 순회.
        - 각 분할에 대해 정보 이득(`_information_gain`)을 계산.
        - 가장 높은 정보 이득을 주는 특성과 기준값을 반환.
    -  예시:"종양 크기" 특성의 값들(예: 1.5cm, 2cm, 3cm 등)을 하나씩 테스트해, 어떤 기준값(예: 2cm)이 데이터를 가장 잘 나누는지 확인합니다.


*   2.5. `_information_gain` (정보 이득 계산)

    ```python
    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        information_gain = parent_entropy - child_entropy
        return information_gain
    ```
    - **역할**: 분할의 품질을 평가하기 위해 정보 이득을 계산합니다.
    - **엔트로피(Entropy)**: 데이터의 불확실성을 측정하는 값. 엔트로피가 낮을수록 데이터가 더 "순수"함(모두 같은 클래스). 0에 가까울수록 데이터가 깔끔하게 분류
    - **정보 이득**:
        - 분할 전 엔트로피(`parent_entropy`)에서 분할 후 자식 노드의 가중 평균 엔트로피(`child_entropy`)를 뺀 값.
        - 정보 이득이 크면 분할이 데이터를 더 잘 나눴다는 뜻.
    - **동작**:
        - 데이터를 두 그룹으로 나눔(`_split`).
        - 각 그룹의 엔트로피를 계산(`_entropy`).
        - 가중 평균 엔트로피를 계산해 정보 이득을 반환.
    - 예시: 만약 "종양 크기 <= 2cm"로 나눴을 때 한 그룹은 모두 양성, 다른 그룹은 모두 악성이라면 엔트로피가 0에 가까워지고 정보 이득이 커집니다.

*   2.6. `_entropy` (엔트로피 계산)

    ```python
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])
    ```

    - **역할**: 데이터의 불확실성을 측정하는 엔트로피를 계산합니다.
    - **엔트로피 공식**: `-Σ(p * log(p))`, 여기서 `p`는 각 클래스의 비율.
    - **동작**:
        - `np.bincount(y)`로 각 클래스의 빈도를 계산.결과 배열의 길이는 arr.max() + 1입니다.
        - 클래스 비율(`ps`)을 구하고, 엔트로피 공식을 적용.
        - 0인 비율은 로그 계산에서 제외(`p > 0`).
    - 예시:데이터에 양성(60%)과 악성(40%)이 있다면:
        - 엔트로피 = `-(0.6 * log(0.6) + 0.4 * log(0.4))` ≈ 0.971.


*   2.7. `_split` (데이터 분할)

    ```python
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    ```

    - **역할**: 특정 특성과 기준값을 기준으로 데이터를 두 그룹으로 나눕니다.
    - **동작**:
        - `X_column`의 값이 `split_thresh` 이하인 샘플의 인덱스를 `left_idxs`에 저장.
        - 초과인 샘플의 인덱스를 `right_idxs`에 저장.
    - **출력**: 두 그룹의 인덱스 배열.
    - np.argwhere():조건을 만족하는 배열 요소의 인덱스(위치)를 반환하는 함수
    - 예시: "종양 크기 <= 2cm"로 나누면, 2cm 이하 샘플은 `left_idxs`, 초과 샘플은 `right_idxs`에 속함.

*   2.8. `_most_common_label` (가장 흔한 클래스)

    ```python
    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    ```

    - **역할**: 리프 노드에서 가장 흔한 클래스를 예측값으로 반환.
    - **동작**:
        - `Counter`를 사용해 클래스 빈도를 계산.
        - 가장 빈도가 높은 클래스를 반환.
    - 예시: 만약 데이터에 양성(10개), 악성(5개)이 있다면, `value`는 "양성"이 됩니다.

*   2.9. `predict` (예측)

    ```python
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    ```

    - **역할**: 새로운 데이터(`X`)에 대해 예측을 수행.
    - **동작**:
        - 각 샘플(`x`)에 대해 `_traverse_tree`를 호출해 나무를 탐색하고 예측값을 반환.
        - 결과는 배열 형태로 반환.


*   2.10. `_traverse_tree` (나무 탐색)

    ```python
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    ```

    - **역할**: 한 샘플(`x`)을 나무를 따라 탐색해 예측값을 반환.
    - **동작**:
        - 리프 노드면 `value`를 반환.
        - 그렇지 않으면, 샘플의 특성 값(`x[node.feature]`)과 노드의 기준값(`node.threshold`)을 비교.
        - 기준값 이하면 왼쪽 가지(`left`), 초과면 오른쪽 가지(`right`)로 이동.
        - 재귀적으로 탐색을 반복.
    - 예시:
        - 샘플의 "종양 크기"가 1.5cm이고, 노드의 기준값이 2cm라면 왼쪽 가지로 이동해 탐색을 계속합니다.

### 4.3 `train.py` 코드 해석
1.이 파일은 의사결정나무를 실제 데이터에 적용하고 평가합니다.

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree

# 데이터 로드
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# 모델 학습
clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)

# 예측
predictions = clf.predict(X_test)

# 정확도 계산
def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(acc)
```

- **데이터 로드**:
  - `datasets.load_breast_cancer()`: 유방암 데이터셋을 로드. `X`는 특성(종양 크기, 모양 등), `y`는 타겟(0: 악성, 1: 양성).
- **데이터 분할**:
  - `train_test_split`: 데이터를 학습용(80%)과 테스트용(20%)으로 나눔.
  - `random_state=1234`: 무작위 분할의 재현성을 보장.
- **모델 학습**:
  - `DecisionTree(max_depth=10)`: 최대 깊이 10인 의사결정나무 객체 생성.
  - `fit(X_train, y_train)`: 학습 데이터로 나무를 생성.
- **예측**:
  - `predict(X_test)`: 테스트 데이터에 대해 예측 수행.
- **정확도 계산**:
  - `accuracy`: 예측값과 실제값을 비교해 정확도를 계산(맞은 예측의 비율).
  - 결과는 `print(acc)`로 출력.
- 예시:
    - 유방암 데이터셋은 약 569개의 샘플과 30개의 특성을 가집니다. 이 코드는 80%의 데이터를 사용해 나무를 학습시키고, 나머지 20%로 모델의 정확도를 평가합니다.

2.전체적인 이해
*   1.코드의 흐름
    1. **데이터 준비** (`train.py`):
        - 유방암 데이터셋을 로드하고 학습/테스트 데이터로 나눕니다.
    2. **모델 학습** (`DecisionTree.py`):
        - `fit` 메서드를 호출해 데이터를 기반으로 나무를 생성.
        - `_grow_tree`가 재귀적으로 나무를 만들며, 각 단계에서 최적의 분할을 찾음(`_best_split`, `_information_gain`).
    3. **예측**:
        - 테스트 데이터를 나무에 통과시켜 예측값을 얻음(`predict`, `_traverse_tree`).
    4. **평가**:
        - 예측값과 실제값을 비교해 정확도를 계산.

*   2.의사결정나무의 동작 원리
    - 데이터를 특성과 기준값으로 나누어 나무를 만듭니다.
    - 각 분할은 정보 이득을 최대화하는 방향으로 이루어집니다.
    - 예측 시, 새 데이터를 나무의 루트에서 리프까지 탐색해 최종 클래스를 반환합니다.

*   3.팁
    - **엔트로피와 정보 이득**:
        - 엔트로피는 데이터의 "혼란스러움"을 나타냅니다. 클래스 분포가 균등할수록 엔트로피가 높음.
        - 정보 이득은 분할 후 엔트로피가 얼마나 줄어드는지를 측정. 좋은 분할은 정보 이득이 큼.
    - **과적합(Overfitting)**:
        - `max_depth`를 제한하지 않으면 나무가 너무 깊어져 학습 데이터에만 잘 맞는 모델이 될 수 있음.
        - 이 코드에서는 `max_depth=10`으로 과적합을 방지.
    - **무작위 특성 선택**:
        - `np.random.choice`로 특성을 무작위로 선택해 계산 효율성을 높이고, 과적합을 줄임.

### 4.4 scikit-learn을 사용해 구현

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 데이터 로드
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# 데이터 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# 의사결정나무 모델 생성 및 학습
clf = DecisionTreeClassifier(max_depth=10, random_state=1234)
clf.fit(X_train, y_train)

# 예측
predictions = clf.predict(X_test)

# 정확도 계산
accuracy = np.mean(y_test == predictions)
print(f"정확도: {accuracy:.4f}")
```

## 5. 랜덤포레스트(Random Forest)

### 5.1 RandomForest란 무엇인가요?

1.개념:
- RandomForest는 **"여러 명의 전문가가 투표해서 결정하는 것"**과 같습니다.
- 한 명의 의사보다 여러 명의 의사가 진단하면 더 정확하듯이
- 하나의 결정트리보다 여러 개의 결정트리가 예측하면 더 정확합니다
- 이를 **앙상블(Ensemble)** 기법이라고 합니다

2.핵심 원리
-   배깅(Bagging)
    - Bootstrap + Aggregating
    - 서로 다른 샘플로 여러 모델 학습 후 결합
- 다수결 투표
    - 분류: 가장 많이 선택된 클래스
    - 회귀: 모든 예측의 평균값
- 무작위성
    - 데이터 샘플링의 무작위성
    - 특성 선택의 무작위성 (코드에는 구현되지 않음)

3.장점
- **과적합 방지**: 여러 트리의 평균으로 일반화 성능 향상
- **노이즈에 강함**: 개별 트리의 실수를 다른 트리들이 보정
- **특성 중요도**: 어떤 특성이 중요한지 자동으로 계산 가능
- **결측값 처리**: 일부 트리에서 실수해도 다른 트리가 보완


### 5.2 RandomForest.py - 메인 클래스
1.초기화(__init__)

```python
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees              # 만들 나무의 개수
        self.max_depth = max_depth           # 나무의 최대 깊이
        self.min_samples_split = min_samples_split  # 분할에 필요한 최소 샘플 수
        self.n_features = n_feature          # 각 분할에서 고려할 특성 수
        self.trees = []                      # 나무들을 저장할 리스트,실용적으로 유용
```

- `n_trees`: 숲에 몇 그루의 나무를 심을지 결정 (많을수록 정확하지만 느려짐)
- `max_depth`: 나무가 얼마나 깊게 자랄 수 있는지 (깊을수록 복잡한 패턴 학습 가능)
- `min_samples_split`: 가지를 더 나누려면 최소 몇 개의 데이터가 필요한지

2.학습(fit) 과정

```python
def fit(self, X, y):
    self.trees = [] # 모든 추정기(estimator)는 fit() 호출 시 이전 학습 결과를 초기화
    for _ in range(self.n_trees):  # n_trees만큼 반복
        tree = DecisionTree(...)   # 새로운 결정트리 생성
        X_sample, y_sample = self._bootstrap_samples(X, y)  # 부트스트랩 샘플링
        tree.fit(X_sample, y_sample)  # 샘플로 트리 학습
        self.trees.append(tree)       # 학습된 트리를 숲에 추가
```

- 설정한 개수만큼 결정트리를 만듭니다
- 각 트리마다 다른 데이터 샘플을 사용해 학습합니다 (부트스트랩)
- 모든 트리를 숲(trees 리스트)에 저장합니다

3.부트스트랩 샘플링 - RandomForest의 핵심

```python
def _bootstrap_samples(self, X, y):
    n_samples = X.shape[0]  # 전체 데이터 개수
    idxs = np.random.choice(n_samples, n_samples, replace=True)  # 복원추출
    return X[idxs], y[idxs]
```

**부트스트랩이란?**
- 원본 데이터에서 **복원추출**로 같은 크기의 샘플을 만드는 것
- 예: 원본 데이터가 $\[1,2,3,4,5\]$라면, 샘플은 $\[1,1,3,5,2\]$ 같이 중복 허용
- 각 트리가 조금씩 다른 데이터로 학습하여 **다양성**을 확보

4.예측(predict) 과정 - 민주주의 투표!

```python
def predict(self, X):
    predictions = np.array([tree.predict(X) for tree in self.trees])  # 모든 트리가 예측
    tree_preds = np.swapaxes(predictions, 0, 1)  # 데이터별로 재정렬
    predictions = np.array([self._most_common_label(pred) for pred in tree_preds])  # 다수결
    return predictions
```

**예측 과정:**
- 숲의 모든 나무가 각자 예측을 합니다
- 각 데이터 포인트별로 모든 나무의 예측을 모읍니다
- **다수결 투표**로 최종 예측을 결정합니다

### 5.3 train.py - 실제 사용 예시

```python
# 1. 데이터 준비
data = datasets.load_breast_cancer()  # 유방암 데이터셋
X = data.data    # 특성 (종양 크기, 모양 등)
y = data.target  # 레이블 (악성/양성)

# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. 모델 생성 및 학습
clf = RandomForest(n_trees=20)  # 20그루 나무로 숲 만들기
clf.fit(X_train, y_train)       # 훈련 데이터로 학습

# 4. 예측 및 평가
predictions = clf.predict(X_test)  # 테스트 데이터 예측
acc = accuracy(y_test, predictions)  # 정확도 계산
```

### 5.4 scikit-learn으로 Random Forest 구현


```python 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# 1. 데이터 로드
data = load_breast_cancer()
X = data.data
y = data.target

# 2. 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. RandomForest 모델 생성 및 학습
# 기본 설정으로 간단하게
rf_basic = RandomForestClassifier(random_state=42)
rf_basic.fit(X_train, y_train)

# 4. 예측
y_pred_basic = rf_basic.predict(X_test)
accuracy_basic = accuracy_score(y_test, y_pred_basic)

print("=== 기본 RandomForest 결과 ===")
print(f"정확도: {accuracy_basic:.4f}")
print()

# 5. 하이퍼파라미터를 조정한 모델
rf_tuned = RandomForestClassifier(
    n_estimators=100,        # 트리 개수 (기본값도 100)
    max_depth=10,           # 최대 깊이
    min_samples_split=5,    # 분할 최소 샘플 수
    min_samples_leaf=2,     # 리프 노드 최소 샘플 수
    max_features='sqrt',    # 각 분할에서 고려할 특성 수
    bootstrap=True,         # 부트스트랩 사용 여부
    random_state=42
)

rf_tuned.fit(X_train, y_train)
y_pred_tuned = rf_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print("=== 튜닝된 RandomForest 결과 ===")
print(f"정확도: {accuracy_tuned:.4f}")
print()

# 6. 상세한 성능 평가
print("=== 상세 분류 보고서 ===")
print(classification_report(y_test, y_pred_tuned, 
                          target_names=['악성', '양성']))

# 7. 혼동 행렬
print("=== 혼동 행렬 ===")
cm = confusion_matrix(y_test, y_pred_tuned)
print(cm)
print()

# 8. 특성 중요도 확인
print("=== 상위 10개 중요 특성 ===")
feature_importance = rf_tuned.feature_importances_
feature_names = data.feature_names

# 중요도 순으로 정렬
indices = np.argsort(feature_importance)[::-1]

for i in range(10):
    idx = indices[i]
    print(f"{i+1:2d}. {feature_names[idx]:25s}: {feature_importance[idx]:.4f}")

# 9. 예측 확률 확인 (처음 5개 샘플)
print("\n=== 예측 확률 (처음 5개 샘플) ===")
y_proba = rf_tuned.predict_proba(X_test[:5])
for i, (prob, pred, actual) in enumerate(zip(y_proba, y_pred_tuned[:5], y_test[:5])):
    print(f"샘플 {i+1}: 악성={prob[0]:.3f}, 양성={prob[1]:.3f} "
          f"| 예측={pred} | 실제={actual}")

# 10. 교차 검증으로 성능 검증
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rf_tuned, X_train, y_train, cv=5, scoring='accuracy')
print(f"\n=== 5-fold 교차검증 결과 ===")
print(f"평균 정확도: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 11. 그리드 서치로 최적 하이퍼파라미터 찾기
from sklearn.model_selection import GridSearchCV

print("\n=== 그리드 서치 실행 중... ===")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1  # 모든 CPU 코어 사용
)

grid_search.fit(X_train, y_train)

print("=== 최적 하이퍼파라미터 ===")
print(grid_search.best_params_)
print(f"최고 교차검증 점수: {grid_search.best_score_:.4f}")

# 최적 모델로 최종 예측
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"테스트 세트 정확도: {accuracy_best:.4f}")
```


## 6. 나이브 베이즈(Naive Bayes)

### 6.1 나이브 베이즈란?

#### (1) 개념
* **Bayes 정리**를 기반으로 하는 분류 알고리즘입니다.
* 특징(feature)들이 서로 **독립(independent)** 라는 가정(나이브, 즉 '순진한')을 합니다.
* 주어진 데이터 **X**가 어떤 클래스 **C**에 속할 확률 $P(C\|X)$을 계산해 가장 확률이 높은 클래스를 선택합니다.

공식: $P(C\|X) = \frac{P(X\|C) \cdot P(C)}{P(X)}$

여기서:

* $P(C)$: 사전확률 (prior) → 해당 클래스가 전체에서 차지하는 비율
* $P(X\|C)$: 우도(likelihood) → 해당 클래스에서 X라는 특징이 나타날 확률
* $P(C\|X)$: 사후확률 (posterior) → X라는 데이터가 주어졌을 때 C일 확률

특징들이 서로 독립이라고 가정하면:

$P(x \mid C_k) = \prod_{i=1}^n P(x_i \mid C_k)$

-   즉, 전체 특징 확률은 각 특징 확률의 곱으로 표현됩니다.

곱셈을 로그로 변환하면 덧셈이 되어 **수치 안정성**이 좋아집니다:

$\log P(C_k \mid x) \propto \log P(C_k) + \sum_{i=1}^n \log P(x_i \mid C_k)$

- 여기서 $\propto$는 “비례한다”는 뜻이며, $P(x)$는 클래스 비교 시 상관 없으니 생략됩니다.

#### (2) 쉬운 비교

나이브 베이즈를 **탐정**에 비유하면:

* **사전확률(P(C))**: 용의자가 등장할 확률(예: A가 범인일 확률 40%, B는 60%)
* **우도($P(X\|C)$)**: 단서가 주어졌을 때 그 용의자가 범인일 가능성 (예: A는 키가 크다, B는 키가 작다)
* **사후확률($P(C\|X)$)**: 단서를 바탕으로 최종적으로 판단하는 범인 확률
* "나이브"란? 모든 단서가 서로 독립적이라고 가정하는 것 (현실에선 완전히 독립적이지 않음)

#### (3) 핵심 포인트

* 나이브 베이즈는 **빠르고 간단**하면서도 의외로 성능이 좋음
* 텍스트 분류(스팸메일, 감정분석)에 특히 효과적
* 이 코드에서는 **정규분포**를 가정한 **가우시안 나이브 베이즈**를 직접 구현

### 6.2 코드 구조

#### (1) `fit()` — 학습 단계

```python
def fit(self, X, y):
    n_samples, n_features = X.shape
    self._classes = np.unique(y)
    n_classes = len(self._classes)

    # 각 클래스의 평균(mean), 분산(variance), 사전확률(prior) 저장
    self._mean = np.zeros((n_classes, n_features))
    self._var = np.zeros((n_classes, n_features))
    self._priors = np.zeros(n_classes)

    for idx, c in enumerate(self._classes):
        X_c = X[y == c]
        self._mean[idx, :] = X_c.mean(axis=0)
        self._var[idx, :] = X_c.var(axis=0)
        self._priors[idx] = X_c.shape[0] / float(n_samples)
```

* **클래스별 평균, 분산**을 구해 저장
* **사전확률** $P(C)$ = 해당 클래스의 데이터 개수 ÷ 전체 데이터 개수

#### (2) `predict()` — 예측 단계

```python
def predict(self, X):
    return np.array([self._predict(x) for x in X])
```

* 각 데이터 샘플 **x**에 대해 `_predict()`를 호출

#### (3) `_predict()` — 한 샘플의 예측

```python
def _predict(self, x):
    posteriors = []

    for idx, c in enumerate(self._classes):
        prior = np.log(self._priors[idx])   # log P(C)
        posterior = np.sum(np.log(self._pdf(idx, x))) + prior
        posteriors.append(posterior)

    return self._classes[np.argmax(posteriors)] #사후확률이 가장 큰 클래스 선택
```
* 이 `_predict()`는 다음 수식을 계산하는 코드입니다:
    * $\hat{C} = \arg\max_{C_k} \left[ \log P(C_k) + \sum_{i=1}^n \log P(x_i \mid C_k) \right]$

* 클래스별 **사후확률**을 계산
* 로그(log)를 쓰는 이유:
  곱셈 연산이 많으면 값이 너무 작아져서 underflow 발생 → 로그 변환으로 안정성 확보
* 가장 큰 posterior를 갖는 클래스를 반환

#### (4) `_pdf()` — 가우시안 확률 밀도 함수
가우시안(정규분포)의 PDF 공식은 다음과 같습니다.

$f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \cdot \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$

- 여기서:
    * $\mu$ = 평균(mean)
    * $\sigma^2$ = 분산(variance)
    * $\sigma$ = 표준편차(std)

```python
def _pdf(self, class_idx, x):
    mean = self._mean[class_idx]
    var = self._var[class_idx]
    numerator = np.exp(-((x - mean) ** 2) / (2 * var))
    denominator = np.sqrt(2 * np.pi * var)
    return numerator / denominator
```

* 각 특징이 **정규분포**를 따른다고 가정하고, 해당 클래스의 mean과 var로 확률을 계산

### 6.3 실행 예제

```python
from sklearn.model_selection import train_test_split
from sklearn import datasets

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print("Accuracy:", accuracy(y_test, predictions))
```

* `make_classification`으로 가상의 데이터 생성
* 학습 후 정확도 출력

### 6.4 scikit-learn을 활용한 나이브 베이즈 구현

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 1. 데이터 생성
#가상의 분류 데이터 생성
X, y = make_classification( 
    n_samples=1000, n_features=10, n_classes=2, random_state=123
)

# 2. 학습용 / 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# 3. 모델 생성 및 학습
model = GaussianNB()
model.fit(X_train, y_train)

# 4. 예측
y_pred = model.predict(X_test)

# 5. 정확도 평가
print("Gaussian Naive Bayes accuracy:", accuracy_score(y_test, y_pred))
```
