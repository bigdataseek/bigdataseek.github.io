---
title: 1차시 7(빅데이터 분석):데이터 전처리 및 클리닝 (실습 예제)
layout: single
classes: wide
categories:
  - Pandas
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---


## 1. Kaggle House Prices 데이터셋을 활용한 실습
- datasets: "House Prices - Advanced Regression Techniques"

### 1.1 데이터 불러오기 및 결측치 확인

- 먼저 필요한 라이브러리를 임포트하고 Kaggle House Prices 데이터를 불러옵니다. `train.csv`와 `test.csv` 두 파일을 사용합니다.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 결측치 확인
total_missing_train = train.isnull().sum().sort_values(ascending=False)
total_missing_test = test.isnull().sum().sort_values(ascending=False)

# 결측치 비율 계산
percent_missing_train = (train.isnull().sum() / len(train)).sort_values(ascending=False)
percent_missing_test = (test.isnull().sum() / len(test)).sort_values(ascending=False)

# 결과 합치기
missing_data_train = pd.concat([total_missing_train, percent_missing_train], axis=1, keys=['Total', 'Percent'])
missing_data_test = pd.concat([total_missing_test, percent_missing_test], axis=1, keys=['Total', 'Percent'])

print("Train 데이터 결측치 현황:\n", missing_data_train[missing_data_train['Total'] > 0])
print("\nTest 데이터 결측치 현황:\n", missing_data_test[missing_data_test['Total'] > 0])
```


### 1.2 결측치 처리 (Imputation)

- 결측치는 데이터의 특징에 따라 다르게 처리합니다. **수치형 데이터**는 평균(mean) 또는 중앙값(median)으로 대체하고, **범주형 데이터**는 최빈값(mode)이나 'None'과 같은 값으로 대체할 수 있습니다. `FireplaceQu`와 같은 특정 칼럼은 결측치가 '해당 없음'을 의미하므로 이를 'None'으로 채우는 것이 적절합니다.

```python
# 수치형 결측치 중앙값(median)으로 대체
# 'LotFrontage' 칼럼 결측치를 Neighborhood별 중앙값으로 대체
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#agg()나 apply()와의 차이점
#agg() / apply(): 그룹별로 요약된 결과 (한 그룹 → 한 행)
#transform(): 그룹별 계산 결과를 원본 행에 모두 부여 (한 그룹 → 여러 행)
#즉, "요약하지 않고 원본 형태를 유지하면서 그룹별 연산 결과를 넣고 싶을 때" transform()을 사용합니다.

# 기타 수치형 결측치 처리
numeric_missing = ['MasVnrArea', 'GarageYrBlt']
for col in numeric_missing:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(test[col].median())

# 범주형 결측치 'None' 또는 최빈값으로 대체
categorical_none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
for col in categorical_none_cols:
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')

# 남은 범주형 결측치는 최빈값으로 대체
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])
test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode()[0])
test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])
test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
test['Functional'] = test['Functional'].fillna(test['Functional'].mode()[0])
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])
```


### 1.3 이상치 제거 (Outlier Removal)

- IQR(사분위 범위)을 사용하여 **이상치**를 탐지하고 제거할 수 있습니다. 이 방법은 데이터의 분포가 **비대칭적**일 때 특히 유용합니다. 다음 코드는 'GrLivArea'와 'SalePrice' 칼럼의 산점도를 그려 이상치를 시각적으로 확인하고, 이를 제거하는 예제입니다.

```python
# 'GrLivArea'와 'SalePrice' 산점도 시각화
plt.figure(figsize=(8, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)
plt.title('GrLivArea vs SalePrice')
plt.show()

# 이상치 제거
# 'GrLivArea'가 4000 이상이거나 'SalePrice'가 600000 이상인 데이터 포인트 제거
train = train.drop(train[(train['GrLivArea'] > 4000) | (train['SalePrice'] > 600000)].index)

# 재시각화하여 이상치 제거 확인
plt.figure(figsize=(8, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)
plt.title('GrLivArea vs SalePrice (Outliers Removed)')
plt.show()

```


### 1.4 데이터 타입 변환 및 Feature Engineering

- 일부 숫자형 칼럼은 실제로는 **범주형 데이터**를 나타낼 수 있습니다. 예를 들어, `MSSubClass`는 주택의 종류를 나타내는 코드이므로 문자열(string)로 변환하는 것이 좋습니다. 또한, `TotalSF`와 같이 여러 칼럼을 합쳐 새로운 특성(feature)을 만드는 **Feature Engineering**을 수행할 수 있습니다.

```python
# 'MSSubClass' 칼럼을 범주형으로 변환
train['MSSubClass'] = train['MSSubClass'].astype('object')
test['MSSubClass'] = test['MSSubClass'].astype('object')

# Feature Engineering: 'TotalSF' 특성 생성
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
```

## 2. 가상의 고객 데이터 활용
### 2.1 실습용 데이터셋 생성

```python
# 필요한 라이브러리를 불러옵니다.
# pandas는 데이터를 다루는 데, numpy는 계산에, matplotlib와 seaborn은 시각화에 사용됩니다.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 실습용 데이터셋 생성하기 ---
# 실제 데이터처럼 결측치, 이상치, 잘못된 데이터 타입, 중복 데이터 등을 포함하는 가상의 데이터를 만듭니다.
# 이 함수는 예시를 위해 미리 만들어져 있으니, 그냥 실행만 하시면 됩니다.
def create_messy_ecommerce_data():
    np.random.seed(42)
    n_records = 2000
    data = {
        'user_id': np.random.randint(1000, 9999, n_records),
        'order_date': pd.date_range('2023-01-01', periods=n_records, freq='H')[:n_records],
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], n_records),
        'price': np.random.exponential(50, n_records) + 10,
        'quantity': np.random.poisson(2, n_records) + 1,
        'customer_age': np.random.normal(35, 12, n_records),
        'customer_income': np.random.normal(55000, 18000, n_records),
        'shipping_cost': np.random.normal(5, 2, n_records),
        'customer_rating': np.random.choice([1, 2, 3, 4, 5], n_records, p=[0.05, 0.1, 0.15, 0.4, 0.3]),
        'payment_method': np.random.choice(['Card', 'PayPal', 'Bank Transfer', 'Cash'], n_records),
        'region': np.random.choice(['Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju'], n_records)
    }
    df = pd.DataFrame(data)
    
    # 인위적으로 문제 있는 데이터 생성
    # 0과 1 사이의 균일분포에서 n_records개의 무작위 숫자를 생성
    # Pandas는 부울 마스크를 인덱스 라벨을 찾는 것과 유사한 방식으로 처리하여 특정 조건에 맞는 행을 효율적으로 선택
    df.loc[np.random.random(n_records) < 0.12, 'customer_age'] = np.nan
    df.loc[np.random.random(n_records) < 0.15, 'customer_income'] = np.nan
    df.loc[np.random.random(n_records) < 0.08, 'customer_rating'] = np.nan
    df.loc[np.random.random(n_records) < 0.05, 'shipping_cost'] = np.nan
    # 무작위로 선택된 50개의 행에서 가격(price) 값을 1000에서 5000 사이의 무작위 값으로 덮어쓰는 것
    df.loc[np.random.choice(df.index, 50, replace=False), 'price'] = np.random.uniform(1000, 5000, 50)
    df.loc[np.random.choice(df.index, 30, replace=False), 'quantity'] = np.random.randint(20, 100, 30)
    df['customer_age'] = df['customer_age'].astype('object')
    df['customer_rating'] = df['customer_rating'].astype('object')
    
    duplicate_indices = np.random.choice(df.index, 100, replace=True)
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    return df

# 함수를 호출하여 더러운 데이터셋을 생성하고, 변수에 저장합니다.
messy_data = create_messy_ecommerce_data()
print("=== 1단계: 원본 데이터 현황 확인 ===")
print("데이터 크기:", messy_data.shape)
print("\n데이터 타입 정보:")
print(messy_data.info())
print("\n결측치 수:")
print(messy_data.isnull().sum())
print("-" * 50)
```

### 2.2 중복 데이터 처리하기

```python
# 'drop_duplicates()' 함수를 사용하여 완전히 똑같은 행을 제거합니다.
# `inplace=True`를 사용하면 원본 데이터프레임을 바로 수정합니다.
print("=== 2단계: 중복 데이터 제거 ===")
initial_rows = len(messy_data)
messy_data.drop_duplicates(inplace=True)
removed_rows = initial_rows - len(messy_data)
print(f"제거된 중복 행 수: {removed_rows}")
print(f"중복 제거 후 데이터 크기: {messy_data.shape}")
print("-" * 50)
```

### 2.3 데이터 타입 변환하기
```python
# 'customer_age'와 'customer_rating'은 숫자로 변환해야 합니다.
# 'pd.to_numeric()' 함수를 사용하여 숫자로 바꾸고,
# 'errors='coerce'' 옵션으로 변환이 불가능한 값은 결측치(NaN)로 만듭니다.
print("=== 3단계: 데이터 타입 변환 ===")
messy_data['customer_age'] = pd.to_numeric(messy_data['customer_age'], errors='coerce')
messy_data['customer_rating'] = pd.to_numeric(messy_data['customer_rating'], errors='coerce')
print("데이터 타입 변환 후:")
print(messy_data[['customer_age', 'customer_rating']].dtypes)
print("-" * 50)
```

### 2.4 결측치 처리하기 (Imputation)

```python
# 결측치는 평균, 중앙값, 최빈값 등으로 채울 수 있습니다.
# 여기서는 숫자는 중앙값으로, 평점은 3(보통)으로 채웁니다.
print("=== 4단계: 결측치 채우기 ===")
# 나이, 소득, 배송비의 결측치를 중앙값으로 채웁니다.
messy_data.fillna({'customer_age':data['customer_age'].median()}, inplace=True)
messy_data.fillna({'customer_income':data['customer_income'].median()}, inplace=True)
messy_data.fillna({'shipping_cost':data['shipping_cost'].median()}, inplace=True)


# 고객 평점의 결측치는 '보통'을 의미하는 3으로 채웁니다.
messy_data.fillna({'customer_rating':3}, inplace=True)

print("결측치 처리 후 결측치 수:")
print(messy_data.isnull().sum())
print("-" * 50)
```

### 2.5 이상치 제거하기
```python
# 데이터의 특정 범위를 벗어나는 값들을 제거합니다.
# 'price'와 'quantity'의 매우 크거나 작은 값들을 제거해 보겠습니다.
print("=== 5단계: 이상치 제거 ===")
initial_rows_after_missing_handling = len(messy_data)

# 가격이 0원보다 크고 500원보다 작은 값만 남깁니다.
messy_data = messy_data[(messy_data['price'] > 0) & (messy_data['price'] < 500)]

# 수량은 0보다 크고 10보다 작은 값만 남깁니다.
messy_data = messy_data[(messy_data['quantity'] > 0) & (messy_data['quantity'] < 10)]

removed_rows_outlier = initial_rows_after_missing_handling - len(messy_data)
print(f"이상치로 제거된 행 수: {removed_rows_outlier}")
print("-" * 50)
```

### 2.6 파생 변수 생성하기 (Feature Engineering)
```python
# 기존 컬럼들을 조합하여 새로운 의미를 가지는 변수를 만듭니다.
print("=== 6단계: 파생 변수 생성 ===")
# 총 구매 금액 컬럼을 만듭니다.
messy_data['total_purchase'] = messy_data['price'] * messy_data['quantity']
print("생성된 'total_purchase' 컬럼의 상위 5개 값:")
print(messy_data['total_purchase'].head())
print("-" * 50)
```

### 2.7 정제 전/후 데이터 비교 시각화
```python
# 데이터가 얼마나 깨끗해졌는지 시각적으로 확인합니다.
print("=== 7단계: 데이터 정제 전/후 비교 ===")
# 정제된 데이터의 통계 요약
print("정제된 데이터의 최종 요약:")
print(messy_data.describe())
print("\n정제된 데이터의 결측치 현황:")
print(messy_data.isnull().sum())

# 정제된 데이터의 가격 분포 시각화
plt.figure(figsize=(10, 6))
sns.histplot(messy_data['price'], bins=50, kde=True)
plt.title('정제된 데이터의 가격 분포')
plt.xlabel('가격')
plt.ylabel('빈도')
plt.show()

# 정제된 데이터의 고객 평점 분포 시각화
plt.figure(figsize=(8, 5))
sns.countplot(x='customer_rating', data=messy_data)
plt.title('정제된 데이터의 고객 평점 분포')
plt.xlabel('평점')
plt.ylabel('고객 수')
plt.show()

print("모든 데이터 정제 과정이 완료되었습니다!")
print(f"최종 데이터 크기: {messy_data.shape}")
```