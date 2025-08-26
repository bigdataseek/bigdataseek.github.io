---
title: 1차시 7(빅데이터 분석):데이터 전처리 및 클리닝 
layout: single
classes: wide
categories:
  - Pandas
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---


## 1. 결측치 처리 (Missing Value Handling)

### 1.1 이론적 배경

- 결측치는 데이터 수집 과정에서 발생하는 불완전한 정보를 의미합니다. 결측치 패턴에 따라 처리 방법이 달라집니다.

**결측치 유형:**

| 구분                                                      | 정의                                  | 예시                       | 분석 시 문제점                  | 처리 전략                                                               |
| ------------------------------------------------------- | ----------------------------------- | ------------------------ | ------------------------- | ------------------------------------------------------------------- |
| **MCAR**<br>(Missing Completely At Random)<br>완전 무작위 결측 | 결측 발생이 데이터와 무관하게 완전히 랜덤             | 설문지 작성 중 잉크 번짐으로 응답이 지워짐 | 데이터 제거·단순 대체해도 편향 없음      | - 단순 제거(Listwise Deletion)<br>- 평균/중앙값 대체                           |
| **MAR**<br>(Missing At Random)<br>무작위 결측                | 결측이 다른 변수와는 관련 있지만, 해당 변수 값 자체와는 무관 | 고연령자가 소득 질문을 생략          | 관련 변수 고려 없이 단순 제거 시 편향 발생 | - 다중 대체(Multiple Imputation)<br>- 회귀 대체                             |
| **MNAR**<br>(Missing Not At Random)<br>비무작위 결측          | 결측 발생이 해당 변수 값 자체와 관련               | 고소득자가 소득을 숨김             | 결측 자체가 중요한 정보, 단순 대체 불가   | - "결측 여부" 변수 추가<br>- 특수한 통계 기법(Selection model 등)<br>- 데이터 수집 설계 보완 |

*   **주요 처리 방법:**
    1. **완전 삭제 (Listwise Deletion)**
    - 결측치가 있는 행 전체를 제거
    - 장점: 간단하고 편향 없음 (MCAR인 경우)
    - 단점: 데이터 손실, 표본 크기 감소

    2. **부분 삭제 (Pairwise Deletion)**
    - 분석에 필요한 변수에서만 결측치 제거
    - 분석마다 다른 표본 크기 사용

    3. **평균/중위수/최빈값 대체**
    - 연속형: 평균 또는 중위수
    - 범주형: 최빈값
    - 단점: 분산 과소추정

    4. **보간법 (Interpolation)**
    - 선형보간: 시계열 데이터의 추세 활용
    - 다항식 보간: 복잡한 패턴 모델링

    5. **다중 대체 (Multiple Imputation)**
    - 소득이 결측일 때, 나이·직업·학력 등을 활용해 "가능한 소득값"을 5가지 정도 만들어서 5개의 데이터셋 생성
    - 불확실성을 반영한 통계적 추론

### 1.2 실습 예제: 온라인 쇼핑몰 고객 데이터

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

# 샘플 데이터 생성
np.random.seed(42)
data = {
    'customer_id': range(1, 1001),
    'age': np.random.normal(35, 12, 1000).astype(int),
    'income': np.random.normal(50000, 15000, 1000),
    'purchase_amount': np.random.normal(200, 100, 1000),
    'website_visits': np.random.poisson(5, 1000),
    'gender': np.random.choice(['M', 'F'], 1000),
    'region': np.random.choice(['Seoul', 'Busan', 'Daegu', 'Incheon'], 1000)
}

df = pd.DataFrame(data)

# 인위적으로 결측치 생성
missing_indices_age = np.random.choice(df.index, 80, replace=False)
missing_indices_income = np.random.choice(df.index, 120, replace=False)
missing_indices_purchase = np.random.choice(df.index, 60, replace=False)

df.loc[missing_indices_age, 'age'] = np.nan
df.loc[missing_indices_income, 'income'] = np.nan
df.loc[missing_indices_purchase, 'purchase_amount'] = np.nan

# 결측치 현황 파악
print("결측치 현황:")
print(df.isnull().sum())
print(f"\n전체 결측치 비율: {df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%")

# 결측치 패턴 시각화
import missingno as msno
msno.matrix(df)
plt.title('결측치 패턴')
plt.show()

# 방법 1: 완전 삭제
df_complete = df.dropna()
print(f"완전 삭제 후 데이터 크기: {df_complete.shape[0]} (원본: {df.shape[0]})")

# 방법 2: 평균/중위수 대체
df_mean_imputed = df.copy()
df_mean_imputed['age'].fillna(df['age'].median(), inplace=True)
df_mean_imputed['income'].fillna(df['income'].mean(), inplace=True)
df_mean_imputed['purchase_amount'].fillna(df['purchase_amount'].median(), inplace=True)

# 방법 3: KNN 대체
numeric_cols = ['age', 'income', 'purchase_amount', 'website_visits']
knn_imputer = KNNImputer(n_neighbors=5)
df_knn_imputed = df.copy()
df_knn_imputed[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
```

## 2. 이상치 탐지 및 처리 (Outlier Detection and Treatment)

### 2.1 이론적 배경

-   이상치는 데이터의 일반적인 패턴에서 크게 벗어난 값으로, 데이터 입력 오류, 측정 오류, 실제 극값일 수 있다.

-   **이상치 탐지 방법:**
    1. **IQR (Interquartile Range) 방법**
    - Q1 - 1.5×IQR 미만 또는 Q3 + 1.5×IQR 초과인 값
    - 장점: 분포에 무관하게 적용 가능
    - 단점: 경직적인 기준

    2. **Z-Score 방법**
    - |Z-score| > 2 또는 3인 값
    - 가정: 데이터가 정규분포를 따름
    - 공식: Z = (X - μ) / σ

    3. **Modified Z-Score**
    - 중위수와 MAD(Median Absolute Deviation) 사용
    - 이상치에 더 강건함

### 2.2 실습 예제: 전자상거래 거래 데이터

```python
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 이상치가 포함된 거래 데이터 생성
np.random.seed(42)
normal_transactions = np.random.normal(100, 30, 950)
outlier_transactions = np.random.uniform(500, 1000, 50)  # 이상치
transaction_amounts = np.concatenate([normal_transactions, outlier_transactions])

df_transactions = pd.DataFrame({
    'transaction_id': range(1, 1001),
    'amount': transaction_amounts,
    'customer_type': np.random.choice(['Bronze', 'Silver', 'Gold'], 1000)
})

# 1. IQR 방법으로 이상치 탐지
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

outliers_iqr = detect_outliers_iqr(df_transactions['amount'])
print(f"IQR 방법으로 탐지된 이상치 개수: {outliers_iqr.sum()}")

# 2. Z-Score 방법으로 이상치 탐지
z_scores = np.abs(stats.zscore(df_transactions['amount']))
outliers_zscore = z_scores > 3
print(f"Z-Score 방법으로 탐지된 이상치 개수: {outliers_zscore.sum()}")

# 이상치 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 박스플롯
axes[0,0].boxplot(df_transactions['amount'])
axes[0,0].set_title('Box Plot - 이상치 탐지')
axes[0,0].set_ylabel('거래금액')

# 히스토그램
axes[0,1].hist(df_transactions['amount'], bins=50, alpha=0.7)
axes[0,1].set_title('히스토그램')
axes[0,1].set_xlabel('거래금액')

# 산점도 (인덱스 vs 금액)
axes[1,0].scatter(df_transactions.index, df_transactions['amount'], alpha=0.5)
axes[1,0].scatter(df_transactions.index[outliers_iqr], 
                 df_transactions['amount'][outliers_iqr], color='red', label='IQR 이상치')
axes[1,0].set_title('IQR 이상치 표시')
axes[1,0].legend()

# Z-score 산점도
axes[1,1].scatter(df_transactions.index, df_transactions['amount'], alpha=0.5)
axes[1,1].scatter(df_transactions.index[outliers_zscore], 
                 df_transactions['amount'][outliers_zscore], color='orange', label='Z-score 이상치')
axes[1,1].set_title('Z-Score 이상치 표시')
axes[1,1].legend()

plt.tight_layout()
plt.show()

# 이상치 처리 방법들
# 방법 1: 제거
df_no_outliers = df_transactions[~outliers_iqr]

# 방법 2: 변환 (로그 변환)
df_log_transformed = df_transactions.copy()
df_log_transformed['log_amount'] = np.log1p(df_transactions['amount'])

# 방법 3: 캡핑 (상한/하한 설정)
df_capped = df_transactions.copy()
Q1 = df_transactions['amount'].quantile(0.25)
Q3 = df_transactions['amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_capped['amount'] = df_capped['amount'].clip(lower=lower_bound, upper=upper_bound)

print(f"원본 데이터 통계:")
print(df_transactions['amount'].describe())
print(f"\n이상치 제거 후 통계:")
print(df_no_outliers['amount'].describe())
print(f"\n캡핑 후 통계:")
print(df_capped['amount'].describe())
```

## 3. 데이터 타입 변환 및 정규화 (Data Type Conversion and Normalization)

### 3.1 이론적 배경

-   **데이터 타입 변환:**
    - 분석 목적에 맞는 적절한 데이터 타입 설정
    - 메모리 효율성 향상
    - 연산 성능 개선

- **정규화 기법:**
    1. **Min-Max 정규화**
    - 공식: (x - min) / (max - min)
    - 범위: $\[0, 1\]$
    - 이상치에 민감함

    2. **Z-Score 표준화**
    - 공식: (x - μ) / σ
    - 평균: 0, 표준편차: 1
    - 정규분포 가정

### 3.2 실습 예제: 다양한 타입의 고객 데이터

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 다양한 타입의 데이터 생성
np.random.seed(42)
customer_data = {
    'customer_id': range(1, 1001),
    'registration_date': pd.date_range('2020-01-01', periods=1000, freq='D')[:1000],
    'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], 1000),
    'income': np.random.normal(60000, 20000, 1000),
    'credit_score': np.random.normal(700, 100, 1000).astype(int),
    'num_purchases': np.random.poisson(10, 1000),
    'satisfaction': np.random.choice(['매우불만', '불만', '보통', '만족', '매우만족'], 1000),
    'is_premium': np.random.choice([True, False], 1000),
    'spending_category': np.random.choice(['Low', 'Medium', 'High'], 1000)
}

df_customers = pd.DataFrame(customer_data)

print("원본 데이터 타입:")
print(df_customers.dtypes)
print("\n데이터 미리보기:")
print(df_customers.head())

# 1. 데이터 타입 변환
df_converted = df_customers.copy()

# 날짜 타입 변환 및 파생 변수 생성
df_converted['registration_date'] = pd.to_datetime(df_converted['registration_date'])
df_converted['days_since_registration'] = (pd.Timestamp.now() - df_converted['registration_date']).dt.days
df_converted['registration_year'] = df_converted['registration_date'].dt.year
df_converted['registration_month'] = df_converted['registration_date'].dt.month

# 범주형 데이터 순서 설정
age_order = ['18-25', '26-35', '36-45', '46-55', '55+']
satisfaction_order = ['매우불만', '불만', '보통', '만족', '매우만족']
spending_order = ['Low', 'Medium', 'High']

df_converted['age_group'] = pd.Categorical(df_converted['age_group'], categories=age_order, ordered=True)
df_converted['satisfaction'] = pd.Categorical(df_converted['satisfaction'], categories=satisfaction_order, ordered=True)
df_converted['spending_category'] = pd.Categorical(df_converted['spending_category'], categories=spending_order, ordered=True)

# 불린 타입 변환
df_converted['is_premium'] = df_converted['is_premium'].astype('bool')

print("\n변환 후 데이터 타입:")
print(df_converted.dtypes)

# 2. 범주형 데이터 인코딩
# One-Hot Encoding
df_encoded = pd.get_dummies(df_converted, columns=['age_group', 'spending_category'], prefix=['age', 'spending'])

# Label Encoding (순서가 있는 범주형)
le = LabelEncoder()
df_encoded['satisfaction_encoded'] = le.fit_transform(df_converted['satisfaction'])

# 3. 수치형 데이터 정규화
numeric_columns = ['income', 'credit_score', 'num_purchases', 'days_since_registration']

# Min-Max 정규화
scaler_minmax = MinMaxScaler()
df_minmax = df_encoded.copy()
df_minmax[numeric_columns] = scaler_minmax.fit_transform(df_encoded[numeric_columns])

# 표준화 (Z-Score)
scaler_standard = StandardScaler()
df_standard = df_encoded.copy()
df_standard[numeric_columns] = scaler_standard.fit_transform(df_encoded[numeric_columns])


# 정규화 결과 비교 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 원본 데이터
axes[0,0].hist(df_encoded['income'], bins=50, alpha=0.7)
axes[0,0].set_title('원본 Income 분포')

# Min-Max 정규화
axes[0,1].hist(df_minmax['income'], bins=50, alpha=0.7, color='orange')
axes[0,1].set_title('Min-Max 정규화 후 Income 분포')

# 표준화
axes[1,0].hist(df_standard['income'], bins=50, alpha=0.7, color='green')
axes[1,0].set_title('표준화 후 Income 분포')

plt.tight_layout()
plt.show()

# 통계 요약
print("\n정규화 방법별 통계 요약 (Income):")
print("원본:", df_encoded['income'].describe().round(2))
print("Min-Max:", df_minmax['income'].describe().round(2))
print("표준화:", df_standard['income'].describe().round(2))
```

## 4. 중복 데이터 처리 (Duplicate Data Handling)

### 4.1 이론적 배경
-   중복 데이터는 동일한 관측치가 여러 번 기록된 경우를 의미하며, 분석 결과를 왜곡할 수 있습니다.

-   **중복 유형:**
    1. **완전 중복**: 모든 컬럼이 동일
    2. **부분 중복**: 일부 핵심 컬럼이 동일
    3. **유사 중복**: 약간의 차이가 있지만 실질적으로 같은 데이터

-   **처리 전략:**
    - 완전 중복: 제거
    - 부분 중복: 비즈니스 규칙에 따라 처리
    - 유사 중복: 데이터 매칭 및 통합

### 4.2 실습 예제: 고객 주문 데이터

```python
import pandas as pd
import numpy as np

# 1. 예시 데이터 만들기
# 총 1000개의 가상 주문 데이터를 만듭니다.
# 일부러 100개의 완전 중복 데이터를 포함시켰습니다.
orders_data = {
    'order_id': [f'ORD{i:03d}' for i in range(1000)],
    'customer_id': np.random.randint(1, 301, 1000),
    'product_name': np.random.choice(['노트북', '스마트폰', '태블릿'], 1000),
    'quantity': np.random.randint(1, 5, 1000),
}
df = pd.DataFrame(orders_data)

# 완전 중복 데이터 추가
duplicate_rows = df.sample(n=100, random_state=42)
df = pd.concat([df, duplicate_rows], ignore_index=True)

print(f"전체 데이터 건수: {len(df)} 건")
print("---")

# 2. 완전 중복 데이터 찾기 및 제거하기
# `duplicated()` 함수는 중복된 행을 True, 아닌 행을 False로 표시합니다.
# `sum()`을 사용하면 True 값의 개수, 즉 중복된 행의 수를 알 수 있습니다.
print("🔎 완전 중복 데이터 찾기")
complete_duplicates = df.duplicated()
print(f"✔️ 발견된 완전 중복 건수: {complete_duplicates.sum()} 건")
print("---")

# `drop_duplicates()` 함수로 중복 데이터를 제거할 수 있습니다.
# `keep='first'`는 중복된 행 중 첫 번째 행만 남기고 나머지를 제거합니다. (기본값)
# `keep='last'`는 마지막 행만 남기고 나머지를 제거합니다.
# `keep=False`는 중복된 모든 행을 제거합니다.
print("🗑️ 완전 중복 데이터 제거하기")
df_no_duplicates = df.drop_duplicates()
print(f"✔️ 중복 제거 후 데이터 건수: {len(df_no_duplicates)} 건")
print("---")

# 3. 특정 컬럼을 기준으로 중복 찾기
# `subset` 옵션에 중복을 확인할 컬럼 이름을 리스트로 넣어줍니다.
# 여기서는 'customer_id'와 'product_name'이 같으면 중복으로 간주합니다.
print("🔎 특정 컬럼(고객+상품) 기준 중복 데이터 찾기")
subset_duplicates = df.duplicated(subset=['customer_id', 'product_name'])
print(f"✔️ 발견된 부분 중복 건수: {subset_duplicates.sum()} 건")
print("---")

# 4. 중복 데이터에 대한 다양한 처리 전략
# 데이터의 성격에 따라 중복을 처리하는 방식은 달라질 수 있습니다.
# 예를 들어, 한 고객이 같은 상품을 여러 번 구매했을 때,
# 이 구매 내역을 하나로 합치고 싶다면 `groupby()` 함수를 사용할 수 있습니다.

print("📊 중복 데이터 처리 전략 (예시)")
# `groupby()`로 'customer_id'와 'product_name'을 기준으로 묶습니다.
# `agg()`를 사용하여 묶인 데이터의 수량을 모두 더합니다.
df_aggregated = df.groupby(['customer_id', 'product_name']).agg(
    total_quantity=('quantity', 'sum')
).reset_index()

print("✔️ 통합된 데이터 건수:", len(df_aggregated), "건")
print("\n[통합된 데이터 예시]")
print(df_aggregated.head())
```
