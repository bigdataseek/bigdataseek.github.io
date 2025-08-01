---
title: 5차시 1:탐색적 데이터 분석(EDA)
layout: single
classes: wide
categories:
  - 탐색적 데이터 분석(EDA)
tags:
  - EDA
toc: true
---

## **1. EDA란 무엇인가?**

### **1.1 정의**  
EDA(Exploratory Data Analysis, 탐색적 데이터 분석)는 데이터를 처음 접했을 때 데이터의 기본적인 특성과 패턴을 이해하고, 데이터 내에 숨겨진 정보를 발견하는 과정입니다. 간단히 말해, "데이터를 탐험하며 데이터가 우리에게 무엇을 말하고 있는지 듣는" 작업이라고 할 수 있습니다.

### **1.2 핵심 개념**  
- EDA는 데이터 분석의 첫걸음으로, 데이터를 이해하고 문제를 정의하는 데 필수적입니다.  
- 데이터를 단순히 처리하거나 모델링하는 것이 아니라, 데이터 자체를 깊이 들여다보는 과정입니다.  
- 통계적 방법과 시각화 도구를 활용하여 데이터의 특성을 파악합니다.

### **1.3 예시 설명**  
예를 들어, Titanic 데이터셋을 분석한다고 가정해봅시다. EDA를 통해 승객들의 나이, 성별, 좌석 등급 등의 변수를 살펴보고, 이들이 생존 여부와 어떤 관계가 있는지 알아볼 수 있습니다. 이러한 초기 분석은 이후 머신러닝 모델을 만들거나 비즈니스 의사결정을 내리는 데 중요한 기초 자료가 됩니다.


## **2. 왜 EDA가 중요한가?**
EDA는 데이터 분석 프로젝트의 성공 여부를 결정짓는 핵심 단계입니다. 이유는 다음과 같습니다:

1. **데이터의 기본 특성 이해**  
   - 데이터를 분석하기 전에 데이터의 구조, 결측값, 이상치 등을 파악해야 합니다.  
   - 예를 들어, 특정 변수에 결측값이 많다면 이를 어떻게 처리할지 미리 고민할 수 있습니다.

2. **문제 정의 및 가설 설정**  
   - 데이터를 탐색하면서 분석 목표를 명확히 하고, 연구 질문을 세울 수 있습니다.  
   - 예: "나이가 많은 승객일수록 생존 확률이 높을까?" 같은 가설을 세울 수 있습니다.

3. **데이터 품질 확인**  
   - 잘못된 데이터나 일관성 없는 데이터를 조기에 발견하여 분석의 신뢰도를 높일 수 있습니다.  
   - 예: "좌석 등급이 4등급인데 데이터에는 5등급이 포함되어 있다" 같은 오류를 발견할 수 있습니다.

4. **4.효율적인 분석 계획 수립**  
   - EDA를 통해 어떤 변수가 중요한지, 어떤 관계가 있는지 파악하면 이후 분석 방향을 효율적으로 설정할 수 있습니다.



## **3. 데이터 분석 프로세스에서 EDA의 위치**
### **3.1데이터 분석 프로세스 개요:**  
데이터 분석 프로젝트는 일반적으로 다음 단계로 진행됩니다:  

- **1.문제 정의:** 해결하고자 하는 문제를 명확히 정의합니다.  
- **2.데이터 수집:** 필요한 데이터를 수집합니다.  
- **3.데이터 전처리:** 데이터를 정리하고 준비합니다.  
- **4.EDA(탐색적 데이터 분석):** 데이터를 탐색하고 이해합니다.  
- **5.모델링:** 머신러닝 모델을 구축하거나 통계적 분석을 수행합니다.  
- **6.결과 해석 및 시각화:** 분석 결과를 해석하고 시각적으로 표현합니다.  
- **7.결론 도출 및 의사결정:** 분석 결과를 바탕으로 의사결정을 내립니다.

### **3.2 EDA의 위치:**  
EDA는 데이터 전처리 이후, 본격적인 모델링이나 분석을 시작하기 전에 수행됩니다.  
- 데이터 전처리 단계에서는 데이터를 정리하고 결측값을 처리하는 등 기본적인 작업을 합니다.  
- EDA는 전처리된 데이터를 기반으로 데이터의 특성을 깊이 이해하고, 분석 방향을 설정하는 단계입니다.  


## **4. EDA의 주요 단계 소개**

1. **데이터 요약: 데이터의 기본 속성 파악**
- **목적:** 데이터의 기본적인 속성과 특성을 이해합니다.  
- **주요 작업:**  
  - 데이터의 구조(변수 이름, 데이터 타입 등) 확인  
  - 데이터의 크기(행/열 수), 결측값 여부 파악  
  - 주요 통계량(평균, 중앙값, 표준편차 등) 계산  
- **예시:**  
  - Titanic 데이터셋에서 승객의 평균 나이, 최대/최소 나이 등을 확인  
  - 결측값이 있는 변수를 파악하고, 이를 어떻게 처리할지 고민  

2. **데이터 시각화: 데이터 분포와 관계 확인**
- **목적:** 데이터를 시각적으로 표현하여 분포와 관계를 직관적으로 이해합니다.  
- **주요 작업:**  
  - 히스토그램, 산점도, 상관관계 히트맵 등 다양한 시각화 도구 활용  
  - 변수 간의 관계를 탐색하고, 데이터의 패턴을 발견  
- **예시:**  
  - 나이와 생존 여부의 관계를 산점도로 시각화  
  - 좌석 등급과 생존 확률의 관계를 박스플롯으로 확인  

3. **데이터 탐색: 데이터 내 숨겨진 패턴과 이상치 발견**
- **목적:** 데이터 내 숨겨진 패턴, 이상치, 특이점을 발견합니다.  
- **주요 작업:**  
  - 이상치 탐지(Z-score, IQR 등 활용)  
  - 그룹별 데이터 비교(예: 성별에 따른 생존율 차이)  
  - 시간 시리즈 데이터의 추세나 계절성 분석(해당 데이터셋이 있는 경우)  
- **예시:**  
  - 나이가 0살인 승객 데이터가 이상치인지 확인  
  - 특정 좌석 등급에서 생존율이 유독 높거나 낮은 이유 탐색  


## **5.데이터 요약**

1.**수치형 데이터 요약**
- **평균(Mean):** 데이터 값들의 합을 데이터 개수로 나눈 값. 데이터의 중심 경향을 나타냅니다.  
- **중앙값(Median):** 데이터를 정렬했을 때 중간에 위치한 값. 이상치에 덜 민감합니다.  
- **최빈값(Mode):** 데이터에서 가장 자주 등장하는 값.  
- **분산(Variance):** 데이터 값들이 평균으로부터 얼마나 퍼져 있는지를 나타냅니다.  
- **표준편차(Standard Deviation):** 분산의 제곱근. 데이터의 퍼짐 정도를 나타냅니다.  
- **사분위수(Quartiles):** 데이터를 4등분한 값(Q1, Q2=중앙값, Q3). 데이터의 분포를 파악하는 데 유용.

  ```python

    import pandas as pd

    # 데이터 로드
    df = pd.read_csv('titanic.csv')

    # Age 변수 요약
    age = df['age'].dropna()  # 결측값 제거

    # 통계량 계산
    mean_age = age.mean()
    median_age = age.median()
    mode_age = age.mode()[0]
    variance_age = age.var()
    std_age = age.std()
    quartiles_age = age.quantile([0.25, 0.5, 0.75])
    
    print(f"평균 나이: {mean_age}")
    print(f"중앙값 나이: {median_age}")
    print(f"최빈값 나이: {mode_age}")
    print(f"분산: {variance_age}")
    print(f"표준편차: {std_age}")
    print(f"사분위수: {quartiles_age}")    
  ```

- 왜도(Skewness)와 첨도(Kurtosis)**  
  - **왜도(Skewness):** 데이터 분포의 비대칭성을 나타내는 지표입니다.  
    - 왜도 = 0: 좌우 대칭인 정규분포  
    - 왜도 > 0: 오른쪽 꼬리가 긴 분포 (양의 왜도)  
    - 왜도 < 0: 왼쪽 꼬리가 긴 분포 (음의 왜도)  
  - **첨도(Kurtosis):** 데이터 분포의 뾰족함과 꼬리 두께를 나타내는 지표입니다.  
    - 첨도 = 3: 정규분포 (메소쿠르틱)  
    - 첨도 > 3: 뾰족하고 두꺼운 꼬리를 가진 분포 (렙토크루틱)  
    - 첨도 < 3: 납작하고 얇은 꼬리를 가진 분포 (플라티쿠르틱)  

  ```python
  from scipy.stats import skew, kurtosis

  # 왜도와 첨도 계산
  skewness_age = skew(age)
  kurtosis_age = kurtosis(age, fisher=False)  # 실제 첨도 값 (정규분포 기준 3)

  print(f"왜도: {skewness_age}")
  print(f"첨도: {kurtosis_age}")
  ```

  - **활용:**  
    - 왜도와 첨도를 통해 데이터의 분포 형태를 파악하고, 이상치나 비대칭성을 확인할 수 있습니다.  
    - 예: 왜도가 크게 치우쳐 있다면 데이터 변환이 필요할 수 있습니다. 첨도가 높다면 극단값(outlier)에 주의해야 합니다.



2.**범주형 데이터 요약**:
범주형 데이터는 특정 카테고리로 분류된 데이터입니다. 이를 요약하기 위해 빈도수와 상대 빈도수를 계산.  
- **빈도수(Frequency):** 각 카테고리가 몇 번 등장하는지 세는 값.  
- **상대 빈도수(Relative Frequency):** 전체 데이터 대비 해당 카테고리의 비율.  



```python
# 성별 빈도수 계산
sex_counts = df['sex'].value_counts()
sex_relative_freq = df['sex'].value_counts(normalize=True)

print("성별 빈도수:")
print(sex_counts)
print("\n성별 상대 빈도수:")
print(sex_relative_freq)
```

## **6.시각화 도구**
### 6.1 Matplotlib (plt)

*   **기본 구조**: Figure (전체 도화지)와 Axes (개별 그래프 영역)로 구성됩니다.
*   **역할**: 그래프의 기본 틀을 만들고, Axes 객체를 통해 개별 그래프를 그립니다.

    ```python
    import matplotlib.pyplot as plt

    # Figure 생성 (도화지 준비)
    fig = plt.figure(figsize=(6, 4))

    # Axes 생성 (스케치북 준비, 1개 행 1개 열을 가진 첫번째 subplot)
    ax = fig.add_subplot(111)

    # 그래프 그리기 (스케치북에 그림 그리기)
    ax.plot([1, 2, 3], [4, 5, 6])

    # 그래프 표시
    plt.show()
    ```

* add_subplot() vs plt.subplots()
  - `add_subplot()` 함수 외에 `plt.subplots()` 함수를 사용하여 Figure와 Axes 객체를 동시에 생성할 수도 있습니다. `plt.subplots()` 함수는 여러 개의 subplot을 한 번에 생성하고, 각 subplot에 대한 Axes 객체를 배열 형태로 반환합니다.

    ```python
    fig, axes = plt.subplots(2, 2) # 2행 2열의 subplot 생성

    axes[0, 0].plot([1, 2, 3], [4, 5, 6]) # 첫 번째 subplot에 그래프 그리기
    axes[0, 1].plot([1, 2, 3], [7, 8, 9]) # 두 번째 subplot에 그래프 그리기


    plt.show()
    ```

### 6.2 Pandas plot()

*   **역할**: Series나 DataFrame 객체에 내장된 함수로, Matplotlib의 pyplot 함수들을 wrapping하여 더 쉽고 간결하게 그래프를 그릴 수 있습니다.
*   **특징**:
    *   데이터를 자동으로 x, y축에 매핑하고, 필요한 경우 데이터 변환을 수행합니다.
    *   `kind` 매개변수로 그래프 종류를 쉽게 선택할 수 있습니다.
    *   Matplotlib의 기능을 그대로 활용할 수 있습니다.
*   **예시**:
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt

    # Series 객체 생성
    s = pd.Series([1, 2, 3, 4, 5])

    # Series 객체로 선 그래프 그리기
    s.plot(kind='line', title='Series Line Graph')
    plt.show()

    # DataFrame 객체 생성
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                       'B': [2, 4, 6, 8, 10]})

    # DataFrame 객체로 선 그래프 그리기
    df.plot(kind='line', title='DataFrame Line Graph')
    plt.show()
    ```

### 6.3 Seaborn (sns)

*   **역할**: Matplotlib을 wrapping하여 더 쉽고 직관적인 인터페이스를 제공하며, 통계적인 정보를 시각적으로 표현하는 데 특화된 기능을 제공합니다.
*   **특징**:
    *   Matplotlib 기반으로 만들어졌기 때문에 Matplotlib의 Figure와 Axes 객체를 그대로 사용.
    *   함수 기반 접근 방식과 Axes-level 접근 방식을 제공합니다.
*   **예시**:
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 데이터 준비
    data = {'x': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]}
    df = pd.DataFrame(data)

    # 함수 기반 접근 방식
    sns.countplot(x='x', data=df)
    plt.show()

    # Axes-level 접근 방식
    fig, ax = plt.subplots()
    sns.countplot(x='x', data=df, ax=ax)
    plt.show()
    ```

### 6.4 요약

| 기능 | Matplotlib (plt) | Pandas plot() | Seaborn (sns) |
|---|---|---|---|
| 역할 | 기본 틀 제공 | Matplotlib wrapping | Matplotlib wrapping, 통계적 시각화 |
| 사용 | 복잡한 코드 | 간결한 코드 | 더 쉽고 직관적인 코드 |
| 특징 | Figure, Axes 객체 사용 | 데이터 자동 처리 | 다양한 그래프 스타일 제공 |


## **7.데이터 시각화**
### 7.1 **수치형 데이터 시각화**
데이터의 분포를 시각적으로 확인하면 데이터의 형태와 특징을 더 잘 이해할 수 있습니다. 주요 시각화 도구는 다음과 같습니다:

- **히스토그램(Histogram):** 데이터의 빈도 분포를 막대로 표현합니다. 

  ```
  import matplotlib.pyplot as plt
  import seaborn as sns

  # 히스토그램
  plt.figure(figsize=(8, 4))
  sns.histplot(age, kde=True, bins=20)
  plt.title("Age Distribution")
  plt.xlabel("Age")
  plt.ylabel("Frequency")
  plt.show()
  ```

- **박스 플롯(Box Plot):** 데이터의 사분위수와 이상치를 한눈에 보여줍니다.  

  ```
  # 박스 플롯
  plt.figure(figsize=(8, 4))
  sns.boxplot(x=age)
  plt.title("Age Box Plot")
  plt.xlabel("Age")
  plt.show()
  ```

- **밀도 추정(Kernel Density Estimate, KDE):** 데이터의 확률 밀도를 부드럽게 표현합니다.  

  ```
  # 밀도 추정
  plt.figure(figsize=(8, 4))
  sns.kdeplot(age, shade=True)
  plt.title("Age Density Plot")
  plt.xlabel("Age")
  plt.ylabel("Density")
  plt.show()
  ```

- **산점도(Scatter Plot):** 산점도는 두 변수 간의 관계를 시각적으로 확인하는 데 유용합니다

  ```python
  # 산점도 그리기
  plt.figure(figsize=(8, 6))
  sns.scatterplot(data=df, x='age', y='fare', hue='survived', alpha=0.7)
  plt.title("Scatter Plot of Age vs Fare")
  plt.xlabel("Age")
  plt.ylabel("Fare")
  plt.legend(title="Legend", loc="upper right")
  plt.show()
  ```

### 7.2 **범주형 데이터 시각화**
범주형 데이터의 비율을 시각적으로 표현하기 위해 파이 차트와 막대 그래프를 사용합니다.  

- **파이 차트(Pie Chart):** 각 카테고리의 비율을 원형으로 표현합니다. 

  ```
  # 파이 차트
  plt.figure(figsize=(6, 6))
  sex_counts.plot.pie(autopct='%1.1f%%', startangle=90)
  plt.title("Sex Distribution (Pie Chart)")
  plt.ylabel("")
  plt.show()
  ```

- **막대 그래프(Bar Chart):** 각 카테고리의 빈도수를 막대로 표현합니다.  

  ```python
  # 막대 그래프
  plt.figure(figsize=(6, 4))
  sns.countplot(x='sex', data=df)
  plt.title("Sex Distribution (Bar Chart)")
  plt.xlabel("Sex")
  plt.ylabel("Count")
  plt.show()
  ```

## **8. 데이터 탐색**

### **8.1 이상치 분석**
- 데이터에서 일반적인 패턴과 크게 벗어난 값으로, 데이터의 정확성을 해칠 수 있거나 중요한 정보를 제공할 수도 있습니다.
- 이상치는 다음과 같은 이유로 발생할 수 있습니다:
  - 측정 오류
  - 입력 오류
  - 자연스러운 변동성
- 이상치를 탐지하는 주요 방법:
  - **박스 플롯(Box Plot)**: 사분위수와 IQR(Interquartile Range)을 사용하여 이상치를 시각적으로 확인.
  - **Z-Score**: 평균에서 표준편차 몇 배 이상 떨어진 데이터를 이상치로 간주 (z의 기준점 3)
    - 정규분포의 경험적 규칙에 따라, z의 절대값 > 3 큰 데이터는 전체 데이터의 0.3% 미만으로 매우 드물다.
    - 이상치를 식별하기 위한 실용적이고 균형 잡힌 기준이다.
    - 너무 낮거나 높은 기준 대비 적절한 타협점을 제공한다.

  - **IQR 기반 필터링**: Q1 - 1.5 * IQR ~ Q3 + 1.5 * IQR 범위를 벗어나는 값을 이상치로 정의.

  - 실습

  ```python
  import seaborn as sns
  import matplotlib.pyplot as plt
  import pandas as pd
  import numpy as np

  # Titanic 데이터셋 로드
  titanic = sns.load_dataset('titanic')

  # 박스 플롯으로 이상치 확인
  plt.figure(figsize=(8, 6))
  sns.boxplot(x=titanic['age'])
  plt.title("Box Plot of Age")
  plt.show()

  # Z-Score 계산으로 이상치 탐지
  from scipy.stats import zscore

  titanic['age_zscore'] = zscore(titanic['age'].dropna())
  outliers = titanic[(titanic['age_zscore'] > 3) | (titanic['age_zscore'] < -3)]
  print("이상치 데이터:")
  print(outliers[['age', 'age_zscore']])

  # IQR 기반 이상치 탐지
  Q1 = titanic['age'].quantile(0.25)
  Q3 = titanic['age'].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  outliers_iqr = titanic[(titanic['age'] < lower_bound) | (titanic['age'] > upper_bound)]
  print("IQR 기반 이상치 데이터:")
  print(outliers_iqr[['age']])
  ```


### **8.2 결측값 분석**
데이터가 누락된 상태를 의미합니다.
- 결측값의 원인:
  - 데이터 수집 과정에서의 문제
  - 설문 응답 미제출
  - 시스템 오류
- 결측값 처리 방법:
  - **삭제**: 결측값이 있는 행 또는 열 제거.
  - **대체**: 평균, 중앙값, 최빈값 등으로 결측값 대체.
  - **보간법**: 시간 순서 데이터에서 이전/다음 값으로 보간.

- **실습**

  ```python
  # 결측값 비율 확인
  missing_values = titanic.isnull().sum() / len(titanic) * 100
  print("결측값 비율(%):")
  print(missing_values)

  # 결측값 시각화
  plt.figure(figsize=(8, 6))
  sns.heatmap(titanic.isnull(), cbar=False, cmap='viridis')
  plt.title("Missing Values Heatmap")
  plt.show()

  # 결측값 처리: 나이(Age) 평균으로 대체
  titanic['age'].fillna(titanic['age'].mean(), inplace=True)

  # 결측값 처리 후 확인
  print("결측값 처리 후:")
  print(titanic.isnull().sum())
  ```


### **8.3 변수 간 관계 분석**
- 변수 간 관계를 분석하면 데이터 내 숨겨진 패턴이나 상관관계를 발견할 수 있습니다.
- 주요 분석 방법:
  - **산점도(Scatter Plot)**: 두 변수 간 선형 또는 비선형 관계 확인.
  - **상관관계 행렬(Correlation Matrix)**: 수치형 변수 간 상관관계 계산.
  - **카테고리별 비교**: 범주형 변수와 수치형 변수 간 관계 확인.

- **실습**

  ```python
  # 산점도로 관계 확인
  plt.figure(figsize=(8, 6))
  sns.scatterplot(data=titanic, x='age', y='fare', hue='survived')
  plt.title("Age vs Fare with Survival")
  plt.show()

  # 상관관계 행렬
  numeric_columns = ['age', 'fare', 'pclass', 'survived']
  correlation_matrix = titanic[numeric_columns].corr() #corr(): 피어슨 상관계수(-1 ~ +1)

  plt.figure(figsize=(8, 6))
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
  plt.title("Correlation Matrix")
  plt.show()

  # 카테고리별 비교: 성별(Sex)에 따른 생존률
  sns.barplot(data=titanic, x='sex', y='survived')
  plt.title("Survival Rate by Sex")
  plt.show()
  ```

### **8.4 패턴/트렌드 발견**
- 데이터에서 반복되는 패턴이나 시간에 따른 트렌드를 발견하면 미래를 예측하거나 인사이트를 도출하는 데 유용합니다.
- 주요 분석 방법:
  - **시계열 분석(Time Series Analysis)**: 시간에 따른 변화 추세 확인.
  - **그룹별 집계(Aggregation by Group)**: 특정 그룹(예: 성별, 클래스) 내에서의 패턴 확인.
  - **클러스터링(Clustering)**: 데이터 포인트를 유사한 그룹으로 묶어 패턴 발견.

- **실습**

  ```python
  # 클래스(Class)별 생존률 비교
  sns.countplot(data=titanic, x='pclass', hue='survived')
  plt.title("Survival Count by Class")
  plt.show()

  # 나이(Age) 그룹별 생존률
  titanic['age_group'] = pd.cut(titanic['age'], bins=[0, 18, 35, 50, 100], labels=['Child', 'Young', 'Adult', 'Senior'])
  sns.barplot(data=titanic, x='age_group', y='survived') #신뢰구간
  plt.title("Survival Rate by Age Group")
  plt.show()

  # 가족 크기(Family Size) 생성 및 생존률 분석
  titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1
  sns.lineplot(data=titanic, x='family_size', y='survived') #신뢰구간
  plt.title("Survival Rate by Family Size")
  plt.show()
  ```

