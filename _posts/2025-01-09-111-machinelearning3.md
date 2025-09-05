---
title: 1차시 11(빅데이터 분석):Machine Learning 3(실습 예제)
layout: single
classes: wide
categories:
  - Casino Analysis
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

## 1. 카지노 고객을 비슷한 특성을 가진 그룹으로 나누는(클러스터링) 과정
- 출처: [02_first_kmeans.ipynb](https://github.com/giraffa-analytics/YT_casino_ml_project/blob/master/02_first_kmeans.ipynb)

### 1.1 필요한 라이브러리 가져오기
```python
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
```

  
이 코드는 데이터 분석과 클러스터링(군집화)을 위해 필요한 라이브러리들을 불러옵니다.  

- `pandas`: 데이터를 표 형태로 다루기 위한 도구.  
- `numpy`: 수치 계산을 쉽게 하기 위한 도구.  
- `datetime`: 날짜와 시간을 다루는 데 사용.  
- `StandardScaler`: 데이터를 표준화(정규화)하는 데 사용.  
- `seaborn`, `matplotlib`: 데이터를 시각화(그래프)하기 위한 도구.  
- `KMeans`: K-평균 클러스터링 알고리즘을 사용하기 위한 도구.  
- `silhouette_samples`, `silhouette_score`: 클러스터링 결과의 품질을 평가하기 위한 도구.  
- `SilhouetteVisualizer`: 클러스터링 결과를 시각적으로 보여주는 도구.

### 1.2 데이터 불러오기 및 전처리
```python
casino = pd.read_csv("Online_casino_DIB.csv")
# Convert the timestamp column to datetime
casino.ReqTimeUTC = pd.to_datetime(casino.ReqTimeUTC)

# Remove timestamps outside the range
casino = casino[casino.ReqTimeUTC <='2020-02-29 00:00:00+00:00'].copy()
# Give new values to df column
casino.TransactionType = casino.TransactionType.map({'LOYALTYCARDDEBIT':'L2D', 'LOYALTYCARDCREDITCL':'L1D', 'LOYALTYCARDCREDIT':'L2W'})
# Filter df by condition 
casino = casino[(casino.TransactionType == "L2D") & (casino.Status=="APPROVED")].reset_index(drop=True)
# and remove single value columns
casino = casino[['AccountIdentifier', 'ReqTimeUTC', 'TransactionAmount']]
# Sort df by column values
casino = casino.sort_values(["AccountIdentifier", "ReqTimeUTC"]).reset_index(drop=True)
# Rename columns
casino.rename(columns = {'AccountIdentifier':'customer', 'ReqTimeUTC':'timest',  'TransactionAmount':'amount'}, inplace=True)
casino.head()
```

  
이 코드는 카지노 데이터를 불러와 전처리하는 과정입니다.  
1. `Online_casino_DIB.csv` 파일을 읽어 `casino`라는 데이터프레임에 저장합니다.  
2. `ReqTimeUTC` 열(시간 정보)을 날짜 형식으로 변환합니다.  
3. 2020년 2월 29일 이후의 데이터는 제외합니다.  
4. `TransactionType` 열의 값을 간단히 변환(예: 'LOYALTYCARDDEBIT' → 'L2D').  
5. 거래 유형이 'L2D'이고 상태가 'APPROVED'인 데이터만 남깁니다.  
6. 필요한 열만 선택(`AccountIdentifier`, `ReqTimeUTC`, `TransactionAmount`)하고, 열 이름을 `customer`, `timest`, `amount`로 변경합니다.  
7. 데이터를 고객 ID와 시간 순으로 정렬합니다.  
8. `casino.head()`는 데이터의 처음 5행을 보여줍니다.

### 1.3 데이터 크기 확인
```python
print(len(casino))
print(len(casino.customer.unique()))
```

  
이 코드는 데이터의 크기를 확인합니다.  

- `len(casino)`: 전체 거래 수(행 수)를 출력합니다.  
- `len(casino.customer.unique())`: 고유한 고객 수를 출력합니다.  
이를 통해 데이터에 몇 개의 거래와 몇 명의 고객이 있는지 알 수 있습니다.

### 1.4 클러스터 분석 설명 (Markdown)
```markdown
# Cluster analysis

Task:\
Based on the features of the customers, create GROUPS such that:
 - similar customers are close together: Within-group variability is small
 - different customers are far apart: Between-group variablity is large

Input:\
Customer features.

Output:\
One labels for each customer, mapping to the group it was assigned to.

... but we don't have any customer features!
So let's build them. The process is called 
#### FEATURE ENGINEERING.
```
  
이 Markdown 셀은 클러스터링의 목표와 과정을 설명합니다.  
- **목표**: 비슷한 고객을 같은 그룹으로 묶고, 다른 고객은 다른 그룹으로 나누는 것.  
- **입력**: 고객의 특징(예: 소비 금액, 구매 횟수 등).  
- **출력**: 각 고객에게 그룹 번호(레이블)를 부여.  
- 하지만 현재 데이터에는 고객의 특징이 없으므로, 이를 만들어내는 **특징 공학(Feature Engineering)** 과정을 진행해야 한다고 설명합니다.


### 1.5 고객 활동 기간 및 구매 횟수 계산
```python
base_timestamp = pd.to_datetime("2020-03-01 00:00:00+00:00")

retention_ = casino.groupby("customer").agg(
    first_active_in_days = ('timest', lambda x: (base_timestamp - x.dt.floor("d").min()).days),
    last_active_in_days = ('timest', lambda x: (base_timestamp - x.dt.floor("d").max()).days),
    nr_purchases = ('timest', 'count')
)
retention_ = retention_.reset_index()
retention_.head(3)
```

  
이 코드는 고객별로 활동 기간과 구매 횟수를 계산합니다.  

1. 기준 날짜(`2020-03-01`)를 설정합니다.  
2. `groupby("customer")`: 고객별로 데이터를 묶습니다.  
3. 각 고객의:  
   - `first_active_in_days`: 첫 거래로부터 기준 날짜까지의 일수.  
   - `last_active_in_days`: 마지막 거래로부터 기준 날짜까지의 일수.  
   - `nr_purchases`: 총 구매 횟수.  
4. 결과를 `retention_` 데이터프레임에 저장하고, 처음 3행을 보여줍니다.

### 1.6 고객별 총 소비 금액 및 변동성 계산
```python
overall_spent = casino.groupby("customer").agg({'amount': ['sum', 'std']}).reset_index()
overall_spent.columns = overall_spent.columns.droplevel(0) #멀티레벨 컬럼에서 첫 번째 레벨(레벨 0)을 제거
overall_spent.rename(columns={"": "customer"}, inplace=True)
overall_spent.head(3)
```

  
이 코드는 고객별로 총 소비 금액과 금액의 변동성을 계산합니다.  
1. `groupby("customer")`: 고객별로 데이터를 묶습니다.  
2. `amount` 열에 대해:  
   - `sum`: 총 소비 금액.  
   - `std`: 소비 금액의 표준편차(변동성).  
3. 열 이름을 정리하고, `customer` 열 이름을 명확히 설정합니다.  
4. 결과를 `overall_spent` 데이터프레임에 저장하고, 처음 3행을 보여줍니다.

### 1.7 고객별 월평균 소비 금액 계산
```python
casino['y_month'] = casino.timest.dt.to_period('M')
monthly_expenditure = casino.groupby(['customer', 'y_month'])['amount'].sum().reset_index()

monthly_average_spent = monthly_expenditure.groupby('customer')['amount'].mean().reset_index()
monthly_average_spent.rename(columns={'amount': 'm_avg_spent'}, inplace=True)
monthly_average_spent
```

  
이 코드는 고객별 월평균 소비 금액을 계산합니다.  
1. `casino['y_month']`: 거래 날짜를 연도-월 형식으로 변환합니다.  
2. `groupby(['customer', 'y_month'])`: 고객과 월별로 데이터를 묶고, 각 월의 총 소비 금액을 계산합니다.  
3. `groupby('customer')`: 고객별로 월 소비 금액의 평균을 계산합니다.  
4. 결과를 `monthly_average_spent` 데이터프레임에 저장하고, 열 이름을 `m_avg_spent`로 변경합니다.

### 1.8 데이터프레임 인덱스 설정
```python
overall_spent = overall_spent.set_index("customer")
monthly_average_spent = monthly_average_spent.set_index("customer")
retention_= retention_.set_index("customer")
```

  
이 코드는 데이터프레임의 인덱스를 `customer` 열로 설정합니다.  
- 인덱스를 설정하면 나중에 데이터프레임을 합칠 때 고객 ID를 기준으로 쉽게 정렬할 수 있습니다.

### 1.9 고객 데이터 통합
```python
customer_data = overall_spent.join([monthly_average_spent, retention_]).reset_index() 
# join() 메서드는 기본적으로 인덱스를 키(Key)로 사용
customer_data
```

  
이 코드는 이전에 만든 데이터프레임(`overall_spent`, `monthly_average_spent`, `retention_`)을 하나로 결합.  
- `join`: 고객 ID를 기준으로 데이터를 결합합니다.  
- `reset_index`: 합친 후 인덱스를 초기화하여 `customer` 열을 다시 일반 열로 만든다.  
- 결과적으로 각 고객의 특징(총 소비, 변동성, 월평균 소비, 활동 기간, 구매 횟수)이 포함된 `customer_data` 데이터프레임이 생성됩니다.

### 1.10 결측값 처리
```python
customer_data.isna().sum()
# Fill with 0
customer_data.loc[customer_data["std"].isna(), "std"] = 0.001
customer_data.loc[customer_data["std"] == 0, "std"] = 0.001
customer_data
```

  
이 코드는 데이터에 결측값(누락된 값)이 있는지 확인하고 처리합니다.  
1. `isna().sum()`: 각 열의 결측값 개수를 확인합니다.  
2. `std` 열(소비 금액의 표준편차)에 결측값이나 0이 있는 경우, 0.001로 채웁니다.  
   - 이는 나중에 로그 변환 시 문제가 없도록 작은 값을 넣는 과정입니다.  
3. 처리된 데이터를 `customer_data`에 저장합니다.

### 1.11 데이터 시각화 (Pairplot)
```python
sns.pairplot(X_std)
```

  
이 코드는 데이터의 특징들 간의 관계를 시각화합니다.  
- `sns.pairplot`: 모든 특징 쌍에 대해 산점도와 히스토그램을 그립니다.  
- `X_std`: 로그 변환된 데이터를 사용합니다(아직 생성되지 않음, 이후 셀에서 정의).  
- 이를 통해 특징 간 상관관계를 눈으로 확인할 수 있습니다.

### 1.12 데이터 선택 및 로그 변환
```python
X = customer_data[['sum', 'std', 'm_avg_spent', 'first_active_in_days','last_active_in_days', 'nr_purchases']]
X_std = np.log(X)
```

  
이 코드는 클러스터링에 사용할 특징을 선택하고 로그 변환을 적용합니다.  
1. `X`: 고객 데이터에서 6개의 특징(총 소비, 변동성, 월평균 소비, 첫 활동 일수, 마지막 활동 일수, 구매 횟수)을 선택합니다.  
2. `np.log(X)`: 데이터의 분포를 정규화하기 위해 로그 변환을 적용합니다.  
   - 로그 변환은 데이터의 값이 너무 크거나 작을 때 스케일을 조정해줍니다.  
3. 변환된 데이터는 `X_std`에 저장됩니다.


### 1.13 Elbow Method로 적절한 클러스터 수 찾기
```python
cluster = [1,2,3,4,5,6]
wcss_ls = []

for i in cluster:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(X_std)
    wcss = kmeans.inertia_
    wcss_ls.append(wcss)
print(wcss_ls)
plt.plot(cluster, wcss_ls)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within Cluster Sum of Squares')
plt.show()
```

  
이 코드는 K-평균 클러스터링에서 적절한 클러스터 수를 찾기 위해 **Elbow Method**를 사용합니다.  
1. `cluster = [1,2,3,4,5,6]`: 클러스터 수를 1부터 6까지 시도합니다.  
2. 각 클러스터 수에 대해:  
   - `KMeans(n_clusters=i)`: K-평균 모델을 생성합니다.  
   - `fit(X_std)`: 로그 변환된 데이터로 클러스터링을 수행합니다.  
   - `inertia_`: 클러스터 내 제곱합(WCSS, Within Cluster Sum of Squares)을 계산합니다.  
3. `wcss_ls`: 각 클러스터 수에 대한 WCSS 값을 저장합니다.  
4. `plt.plot`: 클러스터 수와 WCSS를 그래프로 그려 "궁극점"(Elbow)을 찾습니다.  
   - 그래프에서 꺾이는 지점(급격히 감소가 멈추는 곳)이 적절한 클러스터 수입니다.


### 1.14 Silhouette Visualizer로 클러스터 평가
```python
fig, ax = plt.subplots(4, 2, figsize=(15,15))

for i in [2, 3, 4, 5, 6,7,8,9]:
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(X_std)
    ax[q-1][mod].set_title(f"{i} clusters")
```

  
이 코드는 클러스터링 결과를 **Silhouette Visualizer**로 시각화합니다.  
1. `plt.subplots(4, 2)`: 4x2 격자 형태로 그래프를 준비합니다.  
2. 클러스터 수 2~9에 대해 반복:  
   - `KMeans(n_clusters=i)`: K-평균 모델을 설정합니다(`k-means++`로 초기화, 10번 초기화 시도, 최대 100번 반복).  
   - `SilhouetteVisualizer`: 각 클러스터의 실루엣 점수를 시각화합니다.  
   - 실루엣 점수는 클러스터링의 품질을 평가하며, 값이 높을수록 클러스터가 잘 형성된 것입니다.  
3. 각 그래프는 클러스터 수(2~9)에 따른 실루엣 점수 분포를 보여줍니다.
4. 스코어가 약간 낮더라도 해석력 + 데이터 특성과 맞으면 선택 가능

### 1.15 최종 클러스터링 및 실루엣 점수 계산
```python
cluster_model = KMeans(n_clusters=3)
kmeans = cluster_model.fit(X_std)
# kmeans.labels_는 KMeans가 학습한 후 각 데이터 포인트가 속한 클러스터 번호.(넘파이 배열)
silhouette_score(X_std, labels = kmeans.labels_)
```

  
이 코드는 클러스터 수를 3으로 설정하고 최종 클러스터링을 수행합니다.  
1. `KMeans(n_clusters=3)`: 클러스터 수를 3으로 설정한 K-평균 모델을 만듭니다.  
2. `fit(X_std)`: 로그 변환된 데이터로 클러스터링을 수행합니다.  
3. `silhouette_score`: 클러스터링 결과의 품질을 평가합니다(0~1 사이, 높을수록 좋음).  
   - 결과는 실루엣 점수(예: 0.704...)를 반환합니다.

### 1.16 클러스터 레이블 추가
```python
customer_data['group'] = kmeans.labels_
customer_data
```

  
이 코드는 각 고객에게 클러스터 레이블(그룹 번호)을 추가합니다.  
- `kmeans.labels_`: 각 고객이 속한 클러스터 번호(0, 1, 2)를 반환합니다.  
- `customer_data['group']`: `customer_data`에 `group` 열을 추가해 클러스터 번호를 저장합니다.  
- 최종 데이터프레임에는 각 고객의 특징과 클러스터 번호가 포함됩니다.

### 1.17 클러스터 시각화 (산점도)
```python
fig, ax = plt.subplots()
sns.scatterplot(X_std, x = "std", y = "m_avg_spent", hue=customer_data['group'], alpha=0.7)
plt.show()
```

  
이 코드는 클러스터링 결과를 산점도로 시각화합니다.  
- `sns.scatterplot`: `std`(소비 변동성)와 `m_avg_spent`(월평균 소비)를 축으로, 클러스터별로 색상을 다르게 표시합니다.  
- `hue=customer_data['group']`: 클러스터 번호에 따라 점의 색상을 구분합니다.  
- `alpha=0.7`: 점의 투명도를 조정해 겹치는 부분을 보기 쉽게 합니다.  
- 이를 통해 각 클러스터가 어떻게 분포되어 있는지 확인할 수 있습니다.

### 1.18 또 다른 클러스터 시각화 및 축 조정
```python
fig, ax = plt.subplots()
sns.scatterplot(X_std, x = "std", y = "sum", hue=customer_data['group'], alpha=0.8)

y = [0]
for i in np.arange(2,14,2):
    nr_int = int(np.exp(i))
    y.append(nr_int)
print(y)

x = [0]
for i in range(-4,7,2):
    nr_int = round(np.exp(i),2)
    x.append(nr_int)
print(x)
ax.set_xticklabels(x)
ax.set_yticklabels(y)
plt.show()
```

  
이 코드는 또 다른 산점도를 그리고, 축의 눈금을 조정합니다.  
1. `sns.scatterplot`: `std`(소비 변동성)와 `sum`(총 소비)을 축으로 클러스터를 시각화합니다.  
2. `y`와 `x`: 로그 변환된 값을 원래 스케일로 되돌리기 위해 지수 함수(`np.exp`)를 사용해 눈금 값을 계산합니다.  
   - `y`: y축(총 소비) 눈금을 설정합니다.  
   - `x`: x축(소비 변동성) 눈금을 설정합니다.  
3. `set_xticklabels`, `set_yticklabels`: 계산된 값을 축 눈금으로 설정해 그래프를 더 읽기 쉽게 만든다.  
4. 최종적으로 클러스터링 결과를 시각적으로 확인할 수 있습니다.

### 1.19 요약
이 노트북은 카지노 데이터를 분석하여 고객을 비슷한 특성을 가진 그룹으로 나누는(클러스터링) 과정을 다룹니다. 주요 단계는 다음과 같습니다:  
1. **데이터 전처리**: 카지노 데이터를 불러와 필요한 열만 선택하고 정리합니다.  
2. **특징 공학**: 고객별 특징(총 소비, 변동성, 월평균 소비, 활동 기간, 구매 횟수)을 만듭니다.  
3. **데이터 변환**: 로그 변환으로 데이터 스케일을 조정합니다.  
4. **클러스터링**: K-평균 알고리즘을 사용해 고객을 그룹화하고, Elbow Method와 Silhouette Visualizer로 최적의 클러스터 수를 찾습니다.  
5. **결과 시각화**: 클러스터링 결과를 산점도로 확인합니다.

---

## 2. 카지노 데이터를 분석하여 고객이 앞으로 3일 안에 거래를 할지 예측
- 출처: [03_binary_classification.ipynb)](https://github.com/giraffa-analytics/YT_casino_ml_project/blob/master/03_binary_classification.ipynb)

### 2.1 라이브러리 임포트

```python
import pandas as pd
import numpy as np
import datetime as dt
from collections import Counter
import seaborn as sns
# Modeling
from sklearn.preprocessing import LabelEncoder
# !pip install xgboost
import xgboost as xgb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
```

  
이 코드는 데이터 분석과 머신러닝 모델을 만들기 위해 필요한 파이썬 라이브러리를 불러옵니다.  
- `pandas`: 데이터를 표 형태로 다루기 위한 도구입니다. 엑셀처럼 데이터를 정리하고 분석할 수 있습니다.  
- `numpy`: 수치 계산을 위한 도구입니다. 배열이나 행렬 연산에 유용합니다.  
- `datetime`: 날짜와 시간을 다루기 위한 도구입니다.  
- `Counter`: 데이터의 빈도를 세는 데 사용됩니다.  
- `seaborn`, `matplotlib`: 데이터를 시각화(그래프 그리기)하기 위한 도구입니다.  
- `LabelEncoder`: 범주형 데이터를 숫자로 변환하는 데 사용됩니다.  
- `xgboost`: 강력한 머신러닝 알고리즘으로, 데이터를 기반으로 예측 모델을 만듭니다.  
- `confusion_matrix`: 모델의 예측 성능을 평가하기 위해 사용됩니다.  

### 2.2 데이터 불러오기 및 전처리

```python
casino = pd.read_csv("Online_casino_DIB.csv")
# Convert the timestamp column to datetime
casino.ReqTimeUTC = pd.to_datetime(casino.ReqTimeUTC)

# Remove timestamps outside the range
casino = casino[casino.ReqTimeUTC <='2020-02-29 00:00:00+00:00'].copy()
# Give new values to df column
casino.TransactionType = casino.TransactionType.map({'LOYALTYCARDDEBIT':'L2D', 'LOYALTYCARDCREDITCL':'L1D', 'LOYALTYCARDCREDIT':'L2W'})
# Filter df by condition 
casino = casino[(casino.TransactionType == "L2D") & (casino.Status=="APPROVED")].reset_index(drop=True)
# and remove single value columns
casino = casino[['AccountIdentifier', 'ReqTimeUTC', 'TransactionAmount']]
# Sort df by column values
casino = casino.sort_values(["AccountIdentifier", "ReqTimeUTC"]).reset_index(drop=True)
# Rename columns
casino.rename(columns = {'AccountIdentifier':'customer', 'ReqTimeUTC':'timest',  'TransactionAmount':'amount'}, inplace=True)
casino.head()
```

  
이 코드는 카지노 데이터를 불러오고, 분석에 필요한 형태로 데이터를 정리하는 과정입니다.  
1. `pd.read_csv`: "Online_casino_DIB.csv" 파일을 불러와 `casino`라는 이름의 데이터프레임(표)으로 저장합니다.  
2. `pd.to_datetime`: 시간 데이터(`ReqTimeUTC`)를 파이썬이 이해할 수 있는 날짜/시간 형식으로 변환합니다.  
3. `casino[casino.ReqTimeUTC <= '2020-02-29']`: 2020년 2월 29일 이후의 데이터는 제외합니다.  
4. `TransactionType.map`: 거래 유형을 간단한 이름(`L2D`, `L1D`, `L2W`)으로 변경합니다.  
5. `casino[(casino.TransactionType == "L2D") & (casino.Status=="APPROVED")]`: 거래 유형이 `L2D`(입금)이고 상태가 `APPROVED`(승인)인 데이터만 남깁니다.  
6. `casino[['AccountIdentifier', 'ReqTimeUTC', 'TransactionAmount']]`: 필요한 열만 선택합니다(고객 ID, 시간, 거래 금액).  
7. `sort_values`: 고객 ID와 시간 순으로 데이터를 정렬합니다.  
8. `rename`: 열 이름을 더 직관적으로 변경합니다(`customer`, `timest`, `amount`).  
9. `casino.head()`: 데이터의 첫 5행을 확인하여 올바르게 처리되었는지 확인합니다.  

### 2.3 Problem framing

  
이 부분은 문제를 정의하고 해결 방법을 계획하는 단계입니다.  
1. **질문**: "고객이 앞으로 3일 안에 구매를 할 것인가?" 답은 "예"(1) 또는 "아니오"(0)로, 이진 분류 문제입니다.  
2. **문제 유형**: 이는 **지도 학습(supervised learning)** 문제입니다. 과거 데이터를 바탕으로 미래를 예측하며, 시간 데이터를 다루므로 **시계열(time series)** 요소도 포함됩니다.  
3. **알고리즘**: 이진 분류에 적합한 알고리즘으로는 XGBoost, 로지스틱 회귀, 랜덤 포레스트 등이 있습니다. 이 코드에서는 XGBoost를 사용합니다.  
4. **필요한 데이터**: 고객별로 일별 거래 데이터를 사용하며, 시간 단위는 "일(day)"로 설정합니다.


### 2.4 Feature engineering from transactional data

  
**특성 공학(feature engineering)**은 데이터를 머신러닝 모델이 학습할 수 있는 형태로 변환하는 과정입니다. 여기서는 원래의 거래 데이터를 분석에 유용한 새로운 변수(특성)로 변환합니다. 예를 들어, 고객이 지난 14일 동안 얼마나 자주 거래했는지, 거래 금액은 얼마인지 등을 계산합니다.

### 2.5 일별 데이터 집계

```python
casino["day"] = casino.timest.dt.floor('D')
daily_activity = casino.groupby(["customer", "day"]).agg({"amount": ["sum", "count"]}).reset_index()
daily_activity = daily_activity.droplevel(axis=1, level=1)
daily_activity.columns = ["customer", "day", "daily_amount", "nr_trans"]
daily_activity
```

  
이 코드는 거래 데이터를 고객별, 일별로 요약합니다.  
1. `casino["day"] = casino.timest.dt.floor('D')`: 거래 시간을 일 단위로 변환하여 `day` 열을 추가합니다(예: 2020-01-01 14:30 → 2020-01-01).  
2. `groupby(["customer", "day"]).agg({"amount": ["sum", "count"]})`: 고객별, 날짜별로 거래 금액의 합(`sum`)과 거래 횟수(`count`)를 계산합니다.  
3. `droplevel`: 열 이름의 불필요한 계층을 제거합니다.  
4. `columns = ["customer", "day", "daily_amount", "nr_trans"]`: 열 이름을 직관적으로 변경합니다(`daily_amount`: 일별 총 거래 금액, `nr_trans`: 일별 거래 횟수).  
5. `daily_activity`: 결과를 출력하여 확인합니다.  

**결과**: 고객이 하루에 얼마나 자주, 얼마를 썼는지 요약된 데이터프레임이 생성됩니다.

### 2.6 고객별 전체 날짜 데이터 생성

```python
full_customer_df = pd.DataFrame()
customer_ids = daily_activity.customer.unique()
len(customer_ids)

for customer in customer_ids:
    customer_df = daily_activity[daily_activity.customer == customer]
    customer_full_date_range = pd.date_range(
    customer_df.day.min(),
    customer_df.day.max(),
    freq = "D")
    customer_df = customer_df.set_index(keys = "day").copy()
    
    customer_df = customer_df.reindex(list(customer_full_date_range), fill_value=0)
    # reindex: DataFrame(또는 Series)의 인덱스를 새로운 인덱스 집합으로 맞추는 기능.
    #새로 추가된 인덱스(즉, 기존 DataFrame에는 없었던 날짜)에 대해 결측값이 생기는데, 그걸 0으로 채움
    # reindex()는 숫자형이 아닌 데이터에 fill_value=0를 적용할 수 없어 NaN 값으로 채우게 됩니다.
    customer_df["customer"] = [customer]*len(customer_df)
    customer_df = customer_df.reset_index()

    full_customer_df = pd.concat([full_customer_df, customer_df])
full_customer_df = full_customer_df.reset_index(drop=True)
print(len(full_customer_df))
```

  
이 코드는 각 고객의 거래가 없는 날도 포함하여 연속적인 날짜 데이터를 만듭니다.  
1. `customer_ids`: 데이터에 있는 모든 고객 ID를 가져옵니다.  
2. `for customer in customer_ids`: 각 고객에 대해 반복합니다.  
3. `daily_activity[daily_activity.customer == customer]`: 해당 고객의 데이터를 추출합니다.  
4. `pd.date_range`: 고객의 첫 거래 날짜부터 마지막 거래 날짜까지 모든 날짜를 생성합니다.  
5. `reindex`: 거래가 없는 날에는 거래 금액과 횟수를 0으로 채웁니다.  
6. `pd.concat`: 모든 고객의 데이터를 하나로 합칩니다.  
7. `print(len(full_customer_df))`: 최종 데이터의 행 수를 출력합니다.  

**결과**: 거래가 없는 날도 포함된, 고객별 연속적인 일별 데이터가 생성됩니다.


### 2.7 특성 및 응답 변수 생성

```python
ml_df = pd.DataFrame()
for customer in customer_ids:
    customer_df = full_customer_df[full_customer_df.customer==customer]
    customer_df = customer_df.reset_index(drop=True)

    features = []
    responses = []
    # Initial cutoff indices will be incremented
    x1, x14, y = 0, 14, 17

    while y<=len(customer_df):
        # Base
        feat_x1_x14 = customer_df.nr_trans[x1:x14].values.tolist()
        trans_next_3_days = np.count_nonzero(customer_df.nr_trans[x14:y])
        response = [1 if trans_next_3_days!=0 else 0][0]
      #   response = [0 if trans_next_3_days!=0 else 1][0]
        responses.append(response)

        # Additional features
        ## -- 지난 14일 동안 입금한 일수 
        x15 = np.count_nonzero(customer_df.nr_trans[x1:x14])
        ## -- Nr of days with money deposits in the last 7 days
        x16 = np.count_nonzero(customer_df.nr_trans[x1+7:x14])
        ## -- Nr of days with money deposits in the last 3 days
        x17 = np.count_nonzero(customer_df.nr_trans[x14-2:x14+1])

         # 지난 14일 동안의 평균 입금 수
        x18 = customer_df.nr_trans[x1:x14].mean().tolist()
        # 지난 14일 동안의 일일 최대 입금 횟수
        x19 = customer_df.nr_trans[x1:x14].max().tolist()

        # 지난 7일간 입금된 금액
        x20 = customer_df.daily_amount[x1+7:x14].sum().tolist()
        # 지난 7일간 입금된 평균 일일 금액
        x21 = customer_df.daily_amount[x1+7:x14].mean().tolist()

        feat_x1_x14.extend([x15, x16, x17, x18, x19, x20, x21, customer])
        features.append(feat_x1_x14)

        # increment
        x1+=1
        x14+=1
        y+=1
    df = pd.DataFrame(features)
    df["response"]=responses 
    ml_df  = pd.concat([ml_df , df])
```

  
이 코드는 머신러닝 모델에 사용할 특성(입력 변수)과 응답 변수(예측 대상)를 생성합니다.  
1. **목표**: 지난 14일의 데이터를 바탕으로 다음 3일 동안 거래가 있을지 예측합니다.  
2. `for customer in customer_ids`: 각 고객에 대해 데이터를 처리합니다.  
3. `x1, x14, y = 0, 14, 17`: 14일간의 데이터를 보고(0~13일), 그 다음 3일(14~16일)을 예측합니다.  
4. **응답 변수(response)**: `trans_next_3_days`는 다음 3일 동안의 거래 횟수를 계산합니다. 거래가 있으면 0, 없으면 1로 설정합니다.  
5. **특성 생성**:  
   - `feat_x1_x14`: 지난 14일 동안의 거래 횟수 리스트.  
   - `x15`: 지난 14일 동안 거래가 있었던 날의 수.  
   - `x16`: 지난 7일 동안 거래가 있었던 날의 수.  
   - `x17`: 최근 3일 동안 거래가 있었던 날의 수.  
   - `x18`: 지난 14일 동안의 평균 거래 횟수.  
   - `x19`: 지난 14일 동안의 최대 일일 거래 횟수.  
   - `x20`: 지난 7일 동안의 총 거래 금액.  
   - `x21`: 지난 7일 동안의 평균 일일 거래 금액.  
6. `pd.concat`: 모든 고객의 특성과 응답 변수를 하나로 합칩니다.  

**결과**: 머신러닝 모델에 입력할 수 있는 데이터셋(`ml_df`)이 생성됩니다.


### 2.8 Train-Test Split

  
모델을 학습시키기 위해 데이터를 **훈련 데이터(train)**와 **테스트 데이터(test)**로 나눕니다. 두 가지 방법이 제안됩니다:  
1. **Option 1**: 특정 기간(예: 몇 달)을 테스트 데이터로 사용하고 나머지를 훈련 데이터로 사용.  
2. **Option 2**: 고객별로 데이터를 나누어, 일부 고객은 훈련용, 나머지 고객은 테스트용으로 사용.  
이 코드에서는 고객별로 나누는 **Option 2**를 선택하여 데이터 누출(data leakage)을 방지합니다.


### 2.9 중복 제거

```python
ml_df = ml_df.drop_duplicates(subset=list(range(20))).reset_index(drop=True)
print(len(ml_df))
```

  
중복된 데이터를 제거하여 모델이 잘못 학습하는 것을 방지합니다.  
- `drop_duplicates`: 특성 열(0~19번 열)을 기준으로 중복된 행을 제거합니다.  
- `print(len(ml_df))`: 중복 제거 후 데이터의 행 수를 확인합니다.  

**결과**: 데이터가 깨끗해지고, 중복으로 인한 편향이 줄어듭니다.

### 2.10 응답 변수 인코딩

```python
le = LabelEncoder()
ml_df.response = le.fit_transform(ml_df.response)
```

  
응답 변수(`response`)를 숫자 형태로 변환합니다. 머신러닝 모델은 숫자 데이터를 선호하므로, `LabelEncoder`를 사용해 0과 1로 변환합니다.  
- 예: "예"(1) → 1, "아니오"(0) → 0.  

**결과**: 모델이 이해할 수 있는 숫자형 응답 변수가 준비됩니다.


### 2.11 클래스 균형 확인

```python
print(Counter(ml_df.response))
balanced_df = ml_df.copy()
```

  
`Counter`를 사용해 응답 변수의 클래스 분포(0과 1의 개수)를 확인합니다. 클래스 불균형이 없으면 `balanced_df`에 데이터를 그대로 저장합니다.  
- 클래스 불균형: 한 클래스가 다른 클래스보다 훨씬 많으면 모델이 편향될 수 있습니다.  
- 여기서는 데이터가 이미 균형 잡혀 있다고 가정합니다.  

**결과**: 데이터의 클래스 분포를 확인하고, 추가 작업 없이 진행합니다.

### 2.12 테스트 고객 선택

```python
split_df = balanced_df.iloc[:,[21, 22]].groupby(21).count().reset_index()
# "balanced_df에서 21번째와 22번째 컬럼을 선택하고, 선택된 DataFrame에서 이름이 '21'인 컬럼으로 그룹화하여 개수를 센다"
test_customers = list(split_df.sample(frac=0.2)[21])# 열 이름이 숫자로 되어 있음
test_customers[0:3]
```

  
고객 ID를 기준으로 데이터를 훈련용과 테스트용으로 나눕니다.  
1. `balanced_df.iloc[:,[21, 22]]`: 고객 ID(21번 열)와 응답 변수(22번 열)를 선택합니다.  
2. `groupby(21).count()`: 고객별 데이터 개수를 계산합니다.  
3. `sample(frac=0.2)`: 고객의 20%를 테스트 데이터로 무작위 선택합니다.  
4. `test_customers[0:3]`: 선택된 테스트 고객 ID의 첫 3개를 확인합니다.  

**결과**: 테스트용 고객 ID 리스트가 생성됩니다.


### 2.13 훈련/테스트 데이터 분리

```python
test_df = balanced_df[balanced_df[21].isin(test_customers)]
train_df = balanced_df[~balanced_df[21].isin(test_customers)]
print(len(train_df))
print(Counter(train_df.response))
print(len(test_df))
print(Counter(test_df.response))
```

  
데이터를 훈련 데이터(`train_df`)와 테스트 데이터(`test_df`)로 나눕니다.  
1. `test_df`: 테스트 고객 ID에 해당하는 데이터를 추출합니다.  
2. `train_df`: 테스트 고객을 제외한 나머지 데이터를 추출합니다.  
3. `print` 문으로 훈련/테스트 데이터의 크기와 클래스 분포를 확인합니다.  

**결과**: 모델 학습용 훈련 데이터와 평가용 테스트 데이터가 준비됩니다.


### 2.14 특성과 응답 변수 분리

```python
x_train = train_df.iloc[:, :21]
y_train = train_df.response
x_test = test_df.iloc[:, :21]
y_test = test_df.response
```

  
훈련/테스트 데이터를 입력 특성(`x`)과 응답 변수(`y`)로 나눕니다.  
- `x_train`, `x_test`: 특성(0~20번 열, 즉 21개 특성).  
- `y_train`, `y_test`: 응답 변수(0 또는 1).  

**결과**: 모델 학습에 필요한 입력 데이터(`x_train`, `x_test`)와 정답 데이터(`y_train`, `y_test`)가 준비.


### 2.15 XGBoost 모델 학습

```python
xgb_classifier = xgb.XGBClassifier(eta = 0.05)
xgb_classifier.fit(x_train, y_train)
print(xgb_classifier.score(x_train, y_train))
print(xgb_classifier.score(x_test, y_test))
```

  
XGBoost 모델을 학습시키고 성능을 평가합니다.  
1. `xgb.XGBClassifier(eta = 0.05)`: XGBoost 분류 모델을 생성합니다. `eta`는 학습 속도를 조절하는 파라미터입니다.  
2. `fit(x_train, y_train)`: 훈련 데이터를 사용해 모델을 학습시킵니다.  
3. `score`: 모델의 정확도를 계산합니다.  
   - 훈련 데이터 정확도: `x_train`, `y_train`에 대한 예측 정확도.  
   - 테스트 데이터 정확도: `x_test`, `y_test`에 대한 예측 정확도.  

**결과**: 모델이 학습되고, 훈련/테스트 데이터에 대한 정확도가 출력됩니다.

### 2.16 예측 및 혼동 행렬

```python
pred_labels = xgb_classifier.predict(x_test)
confusion_matrix(y_true = y_test, y_pred = pred_labels)
```

  
테스트 데이터에 대해 예측을 수행하고, 결과를 혼동 행렬(confusion matrix)로 확인합니다.  
1. `predict(x_test)`: 테스트 데이터의 특성을 사용해 예측값(`pred_labels`)을 생성합니다.  
2. `confusion_matrix`: 실제 값(`y_test`)과 예측값(`pred_labels`)을 비교해 혼동 행렬을 생성합니다.  
   - 혼동 행렬은 모델이 얼마나 잘 예측했는지(맞춘 경우와 틀린 경우)를 보여줍니다.  

**결과**: 모델의 예측 성능을 평가할 수 있는 혼동 행렬이 생성됩니다.


### 2.17 성능 지표 계산

```python
tn, fp, fn, tp = confusion_matrix(y_true = y_test, y_pred = pred_labels).ravel()
recall = tp/(tp+fn)
precision = tp/(tp+fp)
false_negative_rate = fn/(tp+fn)
print(f"accuracy:{(tp+tn)/(len(y_test))}")
print(f"precision: {precision}")
print(f"recall:{recall}")
print(f"miss rate:{false_negative_rate}")
```

  
혼동 행렬을 사용해 모델의 성능 지표를 계산합니다.  
1. `tn, fp, fn, tp`: 혼동 행렬을 풀어서 참 음성(True Negative), 거짓 양성(False Positive), 거짓 음성(False Negative), 참 양성(True Positive)을 추출합니다.  
2. **성능 지표**:  
   - `accuracy`: 전체 예측 중 맞춘 비율.  
   - `precision`: 양성으로 예측한 것 중 실제 양성인 비율.  
   - `recall`: 실제 양성 중 모델이 양성으로 맞춘 비율.  
   - `false_negative_rate`: 실제 양성을 음성으로 잘못 예측한 비율(누락률).  

**결과**: 모델의 성능을 정확도, 정밀도, 재현율, 누락률로 확인합니다.


### 2.18 K-폴드 교차 검증 설정

```python
split_df = split_df.sample(frac=1).reset_index(drop=True)
K = 5
test_fold_size = int(len(split_df)/5)
folds = {
    0: {"train": [], "test": []},
    1: {"train": [], "test": []},
    2: {"train": [], "test": []},
    3: {"train": [], "test": []},
    4: {"train": [], "test": []}
}
end_idx = 0
for fold in range(K):
    start_idx = end_idx
    end_idx = end_idx + test_fold_size
    test_fold_df = split_df[start_idx: end_idx]
    test_fold_customers = list(set(test_fold_df[21]))
    train_fold_customers = list(set(split_df[21]) - set(test_fold_df[21]))
    folds[fold]["test"].append(test_fold_customers)
    folds[fold]["train"].append(train_fold_customers)
    print(len(test_fold_customers), len(train_fold_customers))
```

  
K-폴드 교차 검증을 위해 데이터를 5개의 폴드(fold)로 나눕니다.  
1. `split_df.sample(frac=1)`: 고객 데이터를 무작위로 섞습니다.  
2. `K = 5`: 데이터를 5등분합니다.  
3. `test_fold_size`: 각 폴드의 테스트 데이터 크기를 계산합니다.  
4. `folds`: 각 폴드에 대해 훈련 고객과 테스트 고객을 저장할 딕셔너리를 만듭니다.  
5. `for fold in range(K)`: 5번 반복하며, 각 폴드의 테스트 고객과 훈련 고객을 나눕니다.  
6. `print`: 각 폴드의 테스트/훈련 고객 수를 출력합니다.  

**결과**: 5개의 폴드로 데이터를 나누어 교차 검증을 준비합니다.


### 2.19 K-폴드 교차 검증 수행

```python
accuracies = []
precisions = []
recalls = []
miss_rates = []

for fold_id in folds.keys():
    print(fold_id)
    test_customers = folds[fold_id]["test"][0]
    train_customers = folds[fold_id]["train"][0]
    test_df = balanced_df[balanced_df[21].isin(test_customers)]
    train_df = balanced_df[balanced_df[21].isin(train_customers)]

    x_train = train_df.iloc[:, :21]
    y_train = train_df.response
    x_test = test_df.iloc[:, :21]
    y_test = test_df.response
    # Train the model
    xgb_classifier = xgb.XGBClassifier(eta = 0.05)
    xgb_classifier.fit(x_train, y_train)

    # Predict
    pred_labels = xgb_classifier.predict(x_test)
    confusion_matrix(y_true = y_test, y_pred = pred_labels)
    tn, fp, fn, tp = confusion_matrix(y_true = y_test, y_pred = pred_labels).ravel()

    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    false_negative_rate = fn/(tp+fn)
    accuracy = (tp+tn)/(len(y_test))

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    miss_rates.append(false_negative_rate)
```

  
5개의 폴드에 대해 각각 모델을 학습시키고 성능을 평가합니다.  
1. 각 폴드에 대해:  
   - 훈련/테스트 데이터를 나눕니다.  
   - XGBoost 모델을 학습시킵니다.  
   - 테스트 데이터로 예측을 수행하고 혼동 행렬을 생성합니다.  
2. 각 폴드에서 정확도, 정밀도, 재현율, 누락률을 계산하여 리스트에 저장합니다.  

**결과**: 각 폴드의 성능 지표가 저장됩니다.

### 2.20 성능 지표 시각화

```python
metrics = pd.DataFrame.from_dict({
    "acc": accuracies,
    "prec": precisions,
    "rec": recalls,
    "miss": miss_rates
})
sns.lineplot(metrics)
```

  
교차 검증 결과를 시각화합니다.  
1. `pd.DataFrame.from_dict`: 각 폴드의 성능 지표(정확도, 정밀도, 재현율, 누락률)를 데이터프레임으로 만듭니다.  
2. `sns.lineplot`: 각 지표를 선 그래프로 그려 비교합니다.  

**결과**: 모델의 성능을 시각적으로 확인할 수 있는 그래프가 생성됩니다.


### 2.21 요약
카지노 데이터를 분석하여 고객이 앞으로 3일 안에 거래를 할지 예측하는 머신러닝 모델을 만드는 과정 
1. 데이터를 불러오고 정리합니다.  
2. 일별 데이터로 변환하고, 특성을 생성합니다.  
3. 데이터를 훈련/테스트로 나누고, XGBoost 모델을 학습시킵니다.  
4. 모델의 성능을 평가하고, 교차 검증으로 안정성을 확인합니다.  
5. 결과를 시각화하여 모델 성능을 비교합니다.  

이 과정은 데이터 전처리, 특성 공학, 모델 학습, 성능 평가, 시각화까지 머신러닝 프로젝트의 전형적인 흐름을 보여줍니다.