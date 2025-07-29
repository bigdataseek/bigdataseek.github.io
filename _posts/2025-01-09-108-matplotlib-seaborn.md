---
title: 1차시 8(빅데이터 분석):Matplotlib/Seaborn
layout: single
classes: wide
categories:
  - Matplotlib
  - Seaborn
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

## 16. Matplotlib이란?

Matplotlib은 Python에서 가장 널리 사용되는 데이터 시각화 패키지입니다. 데이터 과학 및 머신러닝에서 데이터 탐색, 전처리, 모델 훈련 및 평가 등 다양한 단계에서 중요하게 사용됩니다.

### 16.1 설치 및 기본 설정

```python
# 설치
pip install matplotlib numpy

# 기본 임포트
import matplotlib.pyplot as plt
import numpy as np
```

### 16.2 주요 플롯 유형 및 예제

**1. 선 그래프 (Line Chart)**

연결된 데이터 포인트를 선으로 나타냅니다.

```python
# 기본 선 그래프
years = [2018, 2019, 2020, 2021, 2022]
sales = [100, 120, 80, 150, 180]

plt.plot(years, sales)
plt.title('연도별 매출 추이')
plt.xlabel('년도')
plt.ylabel('매출 (만원)')
plt.show()

# 스타일링된 선 그래프
plt.plot(years, sales, color='blue', linewidth=3, linestyle='--', marker='o')
plt.title('스타일링된 매출 추이')
plt.xlabel('년도')
plt.ylabel('매출 (만원)')
plt.grid(True)  # 격자 추가
plt.show()
```

**2. 막대 그래프 (Bar Chart)**

범주형 데이터의 양을 막대로 나타냅니다.

```python
# 기본 막대 그래프
languages = ['Python', 'Java', 'C++', 'JavaScript', 'Go']
popularity = [30, 25, 15, 20, 10]

plt.bar(languages, popularity)
plt.title('프로그래밍 언어 인기도')
plt.xlabel('언어')
plt.ylabel('인기도 (%)')
plt.show()

# 스타일링된 막대 그래프
colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
plt.bar(languages, popularity, color=colors, edgecolor='black', linewidth=1.2)
plt.title('프로그래밍 언어 인기도')
plt.xlabel('언어')
plt.ylabel('인기도 (%)')
plt.xticks(rotation=45)  # X축 레이블 회전
plt.tight_layout()
plt.show()
```

**3. 히스토그램 (Histogram)**

데이터의 빈도 분포를 보여줍니다.

```python
# 기본 히스토그램
data = np.random.normal(100, 15, 1000)  # 평균 100, 표준편차 15인 정규분포

plt.hist(data, bins=30)
plt.title('성적 분포')
plt.xlabel('점수')
plt.ylabel('빈도')
plt.show()

# 누적 히스토그램
plt.hist(data, bins=30, cumulative=True, alpha=0.7, color='green')
plt.title('성적 누적 분포')
plt.xlabel('점수')
plt.ylabel('누적 빈도')
plt.show()
```

**4. 상자 그림 (Box Plot)**

데이터의 분포와 이상치를 보여줍니다.

```python
# 상자 그림
data1 = np.random.normal(100, 10, 200)
data2 = np.random.normal(110, 15, 200)
data3 = np.random.normal(95, 8, 200)

plt.boxplot([data1, data2, data3], labels=['그룹 A', '그룹 B', '그룹 C'])
plt.title('그룹별 성적 분포')
plt.ylabel('점수')
plt.show()
```

**5. 산점도 (Scatter Plot)**

개별 데이터 포인트를 점으로 나타냅니다.

```python
# 기본 산점도
x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 5, 3]

plt.scatter(x, y)
plt.title('기본 산점도')
plt.xlabel('X 값')
plt.ylabel('Y 값')
plt.show()

# 스타일링된 산점도
x = np.random.randn(100)
y = np.random.randn(100)

plt.scatter(x, y, color='red', marker='*', s=50, alpha=0.7)
plt.title('스타일링된 산점도')
plt.xlabel('X 값')
plt.ylabel('Y 값')
plt.show()
```

**6. 파이 차트 (Pie Chart)**

비율을 원형으로 나타냅니다.

```python
# 기본 파이 차트
labels = ['Python', 'Java', 'C++', 'JavaScript']
sizes = [30, 25, 20, 25]

plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('프로그래밍 언어 사용 비율')
plt.show()

# 스타일링된 파이 차트
explode = (0.1, 0, 0, 0)  # Python 조각을 분리
colors = ['gold', 'lightcoral', 'lightskyblue', 'lightgreen']

plt.pie(sizes, labels=labels, autopct='%1.1f%%', explode=explode, 
        colors=colors, startangle=90, pctdistance=0.85)
plt.title('프로그래밍 언어 사용 비율')
plt.show()
```

**7. 영역차트**

시간의 흐름에 따른 데이터의 변화량을 보여주는 데 유용한 차트

```python
#1. 기본 영역 차트
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
# x축: 시간 또는 순서 (예: 월)
x = np.array([1, 2, 3, 4, 5])
# y축: 데이터 값 (예: 월별 매출)
y = np.array([10, 25, 15, 30, 20])

plt.figure(figsize=(8, 5)) # 차트 크기 설정
plt.fill_between(x, y, color="skyblue", alpha=0.4) # 영역 채우기 (색상, 투명도)
plt.plot(x, y, color="Slateblue", alpha=0.6, linewidth=2) # 데이터 라인 추가

plt.title('월별 매출 변화 (기본 영역 차트)')
plt.xlabel('월')
plt.ylabel('매출액 (단위: 만원)')
plt.grid(True, linestyle='--', alpha=0.7) # 그리드 추가
plt.show()

```

### 16.3 여러 플롯 그리기

**1. 서브플롯 (Subplots)**

```python
# 2x2 서브플롯
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 첫 번째 서브플롯
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sin 함수')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('sin(x)')

# 두 번째 서브플롯
axes[0, 1].plot(x, np.cos(x), color='red')
axes[0, 1].set_title('Cos 함수')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('cos(x)')

# 세 번째 서브플롯
data = np.random.randn(1000)
axes[1, 0].hist(data, bins=30, alpha=0.7)
axes[1, 0].set_title('랜덤 데이터 히스토그램')
axes[1, 0].set_xlabel('값')
axes[1, 0].set_ylabel('빈도')

# 네 번째 서브플롯
languages = ['Python', 'Java', 'C++']
popularity = [40, 35, 25]
axes[1, 1].bar(languages, popularity, color=['blue', 'orange', 'green'])
axes[1, 1].set_title('언어 인기도')
axes[1, 1].set_ylabel('비율 (%)')

plt.suptitle('다양한 플롯 예제', fontsize=16)
plt.tight_layout()
plt.show()
```

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig = plt.figure()                  # Figure 생성
ax = fig.add_subplot(111)           # 2D 서브플롯(1행1열 첫번째)
ax.plot(x, y, color='red', label='sin(x)')
ax.set_title("2D Line Plot")        # 제목
ax.set_xlabel("X-axis")            # 축 레이블
ax.set_ylabel("Y-axis")
ax.legend()                        # 범례
plt.show()
```

### 16.4 스타일링 및 사용자 정의

**1. 스타일 시트 사용**

```python
# 사용 가능한 스타일 확인
print(plt.style.available)

# ggplot 스타일 적용
plt.style.use('ggplot')

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, linewidth=2)
plt.title('ggplot 스타일 적용')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.show()

# 기본 스타일로 복원
plt.style.use('default')
```

**2. 범례 추가**
*   label을 지정했다면 반드시 plt.legend()를 호출해야 범례가 표시
*   별도의 창에 그리려면 plt.figure()를 새로 호출하면 됩니다.

```python
x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x), label='sin(x)', linewidth=2)
plt.plot(x, np.cos(x), label='cos(x)', linewidth=2)
#plt.figure()
plt.plot(x, np.tan(x), label='tan(x)', linewidth=2)

plt.title('삼각함수 비교')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.ylim(-2, 2)  # y축 범위 제한
plt.show()
```

### 16.5 플롯 저장

```python
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, linewidth=2)
plt.title('Sin 함수')
plt.xlabel('x')
plt.ylabel('sin(x)')

# 고해상도로 저장
plt.savefig('sin_function.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
```

### 16.6 3D 플로팅 예제

```python
from mpl_toolkits.mplot3d import Axes3D

# 3D 산점도
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)

ax.scatter(x, y, z, c='red', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D 산점도')
plt.show()

# 3D 표면 플롯
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D 표면 플롯')
plt.show()
```

## 17. Seaborn 기본 가이드 및 예제

### 17.1 Seaborn이란?

Seaborn은 Matplotlib을 기반으로 한 Python의 통계 데이터 시각화 라이브러리입니다. Matplotlib보다 더 아름답고 통계적으로 유용한 시각화를 간단한 코드로 만들 수 있습니다.

*   Seaborn의 주요 장점:
    - **통계적 시각화에 특화**: 상관관계, 분포, 회귀 분석 등을 쉽게 시각화
    - **아름다운 기본 스타일**: 별도 스타일링 없이도 보기 좋은 그래프
    - **Pandas와 완벽 호환**: DataFrame을 직접 사용 가능
    - **내장 데이터셋**: 연습용 데이터셋 제공

### 📌 **Seaborn을 사용하면 좋은 상황**

1. **통계적 시각화가 필요할 때**
   - **분포도**, **회귀선**, **신뢰구간** 등 **통계 분석과 결합된 그래프**를 쉽게 그릴 수 있습니다.  
     - 예: 산점도 + 선형 회귀선 (`sns.regplot`), 분포 비교 (`sns.boxplot`, `sns.violinplot`).  
     ```python
     import seaborn as sns
     sns.regplot(x='total_bill', y='tip', data=tips)  # 회귀선 자동 추가
     ```

2. **다변량 데이터(Multivariate Data) 시각화**
   - **여러 변수의 관계**를 한 번에 표현할 때 유용합니다.  
     - 예: `sns.pairplot` (모든 변수의 산점도 행렬), `sns.heatmap` (상관관계 행렬).  
     ```python
     sns.pairplot(iris, hue='species')  # 종(species)별로 색상 구분
     ```

3. **복잡한 카테고리 데이터 시각화**
   - **범주형 변수**의 비교에 최적화되어 있습니다 (e.g., `barplot`, `countplot`, `swarmplot`).  
     ```python
     sns.boxplot(x='day', y='total_bill', data=tips)  # 요일별 지급 금액 분포
     ```

4. **시각적 미학(테마, 색상)을 자동 개선**
   - **기본 색상 팔레트와 스타일**이 matplotlib보다 세련됩니다.  
   - 한 줄로 테마 변경 가능 (`sns.set_style("darkgrid")`).  

5. **데이터프레임과의 높은 호환성**
   - `data` 매개변수에 **pandas DataFrame을 직접 전달**할 수 있어 코드가 간결합니다.  
     ```python
     sns.lineplot(x='date', y='value', data=df, hue='category')  # hue로 자동 그룹화
     ```


### 17.2 설치 및 기본 설정

```python
# 설치
pip install seaborn pandas numpy matplotlib

# 기본 임포트
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Seaborn 스타일 설정
sns.set_theme()  # 또는 sns.set_style("whitegrid")
```

### 17.3 기본 플롯 유형 및 예제
### Seaborn의 기본적인 플롯 유형

1.  **분포 플롯 (Distribution Plots)**: 단일 변수의 데이터 분포를 보여줍니다.

      * **히스토그램 (Histogram) / 커널 밀도 추정 (KDE)**: `sns.histplot()`, `sns.kdeplot()`
          * **용도**: 단일 숫자 변수의 데이터가 어떤 값에 가장 많이 분포하는지, 전체적인 모양은 어떤지 파악할 때 사용합니다. `histplot`은 막대로 빈도를, `kdeplot`은 부드러운 곡선으로 밀도를 나타냅니다.
          * **예시**: 특정 과목 점수 분포, 키/몸무게 분포

        ```python
        # 사용 가능한 모든 데이터셋 이름 출력
        # print(sns.get_dataset_names())  

        # 내장 데이터셋 사용
        tips = sns.load_dataset("tips")        

        # 히스토그램 + 커널 밀도 추정
        #데이터 포인트 주위에 커널 함수(예: 가우시안)를 놓고, 이를 합쳐 부드러운 확률 밀도 곡선을 생성
        plt.figure(figsize=(8, 6))
        sns.histplot(data=tips, x="total_bill", kde=True)
        plt.title('총 계산서 분포')
        plt.show()

        # 여러 그룹의 분포 비교
        plt.figure(figsize=(10, 6))
        sns.histplot(data=tips, x="total_bill", hue="time", multiple="stack")
        plt.title('시간대별 계산서 분포')
        plt.show()
        ```

        ```python
        # print(tips.head()) 결과
                total_bill   tip     sex smoker  day    time  size
        0       16.99  1.01  Female     No  Sun  Dinner     2
        1       10.34  1.66    Male     No  Sun  Dinner     3
        2       21.01  3.50    Male     No  Sun  Dinner     3
        3       23.68  3.31    Male     No  Sun  Dinner     2
        4       24.59  3.61  Female     No  Sun  Dinner     4
        ```

2.  **관계 플롯 (Relational Plots)**: 두 개 이상의 변수 간의 관계를 보여줍니다.

      * **산점도 (Scatter Plot)**: `sns.scatterplot()`
          * **용도**: 두 숫자 변수 사이에 어떤 경향성(선형 관계, 군집 등)이 있는지 파악할 때 사용합니다. 각 점은 하나의 데이터 포인트를 나타냅니다.
          * **예시**: 공부 시간과 성적의 관계, 광고 비용과 판매량의 관계

        ```python
        # 내장 데이터셋 사용
        tips = sns.load_dataset("tips")

        # 기본 산점도
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=tips, x="total_bill", y="tip")
        plt.title('총 계산서 vs 팁')
        plt.show()

        # 범주별 색상 구분
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", style="smoker")
        plt.title('시간대와 흡연 여부에 따른 팁')
        plt.show()
        ```


        ```python
        # 회귀선이 포함된 산점도 
        # 지정한 Figure 사용   
        plt.figure(figsize=(8, 6))    
        sns.regplot(data=tips, x="total_bill", y="tip")
        plt.title('총 계산서와 팁의 관계 (회귀선 포함)')
        plt.show()

        # 조건별 회귀 플롯
        # 그룹별 비교와 서브플롯 분할이 가능.                
        # lmplot은 내부적으로 FacetGrid를 사용해 서브플롯을 생성,자체 figure생성        
        sns.lmplot(data=tips, x="total_bill", y="tip", hue="smoker", col="time")
        plt.show()

        # 페어 플롯 (모든 변수 간 관계)
        # 대각선에는 분포(히스토그램/KDE)가 표시
        # pairplot도 PairGrid를 사용하므로 자체 figure를 생성        
        iris = sns.load_dataset("iris")
        sns.pairplot(iris, hue="species")
        plt.show()
        ```
      * **선 그래프 (Line Plot)**: `sns.lineplot()`
        ```python
        #다중 선 그래프 (hue로 그룹 분리)
        sns.lineplot(
            data=tips,
            x="total_bill", 
            y="tip", 
            hue="sex",  # 성별에 따라 다른 색상의 선 생성
            style="sex",  # 선 스타일도 분리 (점선, 실선 등)
            markers=True,  # 데이터 포인트에 마커 표시
            palette="pastel"  # 색상 팔레트 지정
        )
        ```
        ```python
        오차 영역 표시 (ci)
        sns.lineplot(
            data=tips, 
            x="day", 
            y="tip", 
            errorbar="sd"  # 표준편차로 오차 영역 표시 (기본값: 95% 신뢰구간)
        )
        ```
        ```python
        #시간 순서 데이터에 최적화
        # 날짜 형식 데이터 자동 인식
        fmri = sns.load_dataset("fmri")
        sns.lineplot(
            data=fmri, 
            x="timepoint", 
            y="signal", 
            hue="event"
        )
        ```

3.  **범주형 플롯 (Categorical Plots)**: 하나 이상의 범주형 변수와 숫자형 변수 간의 관계를 보여줍니다.

      * **막대 그래프 (Bar Plot)**: `sns.barplot()`
          * **용도**: 각 범주별 숫자 변수의 평균, 합계 등의 통계량을 막대로 시각화할 때 사용합니다. 기본적으로 평균을 나타내며, 신뢰 구간도 함께 표시됩니다.
          * **예시**: 요일별 매출 평균, 성별에 따른 만족도 평균

        ```python
        # 막대 그래프
        plt.figure(figsize=(8, 6))
        sns.barplot(data=tips, x="day", y="total_bill")
        plt.title('요일별 평균 계산서')
        plt.show()
        ```
      * **박스 플롯 (Box Plot)**: `sns.boxplot()`
          * **용도**: 각 범주별 숫자 변수의 분포(중앙값, 사분위수, 이상치 등)를 한눈에 비교할 때 사용합니다. 데이터의 퍼짐과 중심 경향을 파악하기 좋습니다.
          * **예시**: 학년별 시험 점수 분포, 지역별 집값 분포
        
        ```python
        # 박스 플롯
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=tips, x="day", y="total_bill")
        plt.title('요일별 계산서 분포')
        plt.show()
        ```
      * **바이올린 플롯 (Violin Plot)**: `sns.violinplot()`
          * **용도**: 박스 플롯과 유사하지만, 데이터의 밀도 분포까지 함께 보여주어 데이터가 특정 구간에 더 밀집되어 있는지 등을 파악하기에 좋습니다. 박스 플롯과 KDE의 장점을 합친 형태입니다.
          * **예시**: 박스 플롯과 동일하지만, 분포의 모양까지 더 자세히 보고 싶을 때

        ```python
        # 바이올린 플롯 (박스 플롯 + 커널 밀도)
        plt.figure(figsize=(8, 6))
        sns.violinplot(data=tips, x="day", y="total_bill", hue="time")
        plt.title('요일과 시간대별 계산서 분포')
        plt.show()
        ```
      
      * **기타 범주형 데이터 시각화**:

        ```python
        # 점 플롯 (평균과 신뢰구간)
        plt.figure(figsize=(8, 6))
        sns.pointplot(data=tips, x="day", y="total_bill", hue="time")
        plt.title('요일과 시간대별 평균 계산서')
        plt.show()

        # 카운트 플롯
        plt.figure(figsize=(8, 6))
        sns.countplot(data=tips, x="day", hue="time")
        plt.title('요일별 방문 횟수')
        plt.show()
        ```

4. **관계 + 범주형 플롯**:히트맵(heatmap)

    ```python
    # 상관관계 히트맵
    
    iris = sns.load_dataset("iris")
    plt.figure(figsize=(8, 6))

    # 수치형 특성만 선택하여 상관 관계 계산
    # DataFrame의 .corr() 메서드를 사용하여 각 수치형 특성 간의 피어슨 상관 계수를 계산합니다. 
    # 이 결과는 상관 관계 행렬(DataFrame)이 됩니다
    # 히트맵은 2D 행렬 형태의 데이터만 입력 가능하므로,
    correlation_matrix = iris.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('붓꽃 데이터 상관관계')
    plt.show()

    ```

### 17.4 고급 시각화 예제

**1. Facet Grid를 이용한 다중 플롯**
*  하나의 그래프를 여러 개의 작은 그래프(패싯)로 분할하여 보여주는 그리드
*  각 작은 그래프(패싯)는 데이터의 특정 부분집합을 나타냅니다.
* FacetGrid 객체 생성 후 플롯 매핑 (map 또는 map_dataframe)

```python
# FacetGrid 사용
g = sns.FacetGrid(tips, col="time", row="smoker",hue='sex', margin_titles=True)
g.map(sns.scatterplot, "total_bill", "tip", alpha=0.7)
g.add_legend()
plt.show()
```

**2. 조인트 플롯 (Joint Plot)**
*   두 변수 간의 **관계(relationship)**와 각 변수의 **개별 분포(individual distributions)**를 동시에 보여주는 강력한 통계 데이터 시각화
*   상단(Top) 플롯: X 변수의 **단변량 분포, 오른쪽(Right) 플롯: Y 변수의 단변량 분포

```python
# 산점도 + 각 축의 분포
sns.jointplot(data=tips, x="total_bill", y="tip", kind="reg")
plt.show()

# 육각형 빈도 플롯
sns.jointplot(data=tips, x="total_bill", y="tip", kind="hex")
plt.show()
```

**3. 클러스터맵**
*   계층적 클러스터링이 적용된 히트맵
*   좌측 덴드로그램 (행 덴드로그램): 행(데이터 샘플)의 유사성에 기반한 계층적 클러스터링 결과를 제시
*   상단 덴드로그램 (열 덴드로그램): 열(변수)의 유사성에 기반한 계층적 클러스터링 결과를 제시

```python
# 계층적 클러스터링이 적용된 히트맵
flights = sns.load_dataset("flights")

# pivot함수(인덱스, 컬럼, 셀 값)
flights_pivot = flights.pivot(index="month",columns= "year", values="passengers")

plt.figure(figsize=(10, 8))
sns.clustermap(flights_pivot, cmap="YlOrRd", linewidth=0.5)
plt.show()
```


### 17.5 실무 활용 예제

**1. 시계열 데이터 시각화**

```python
# 항공 승객 데이터
flights = sns.load_dataset("flights")

plt.figure(figsize=(12, 6))
sns.lineplot(data=flights, x="year", y="passengers", hue="month")
plt.title('연도별 월간 항공 승객 수')
plt.show()

# 히트맵으로 시계열 패턴 보기
# 히트맵은 2D 행렬 형태의 데이터만 입력 가능하므로,
plt.figure(figsize=(10, 8))
flights_pivot = flights.pivot("month", "year", "passengers")
sns.heatmap(flights_pivot, annot=True, fmt="d", cmap="YlOrRd")
plt.title('연도-월별 항공 승객 수')
plt.show()
```

**2. 다변량 분석**

```python
# 붓꽃 데이터로 다변량 분석
iris = sns.load_dataset("iris")

# 1. 전체 변수 관계 보기
# pairplot은 자체 Figure를 갖음
sns.pairplot(iris, hue="species", diag_kind="kde")
plt.show()

# 2. 특정 변수들만 선택
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris, x="sepal_length", y="sepal_width", 
                size="petal_length", hue="species", sizes=(50, 200))
plt.title('꽃받침 길이 vs 너비 (꽃잎 길이를 크기로 표현)')
plt.show()
```

**3. 통계적 테스트 결과 시각화**
*   Matplotlib은 명시적으로 새로운 Figure를 생성하거나 서브플롯을 지정하지 않는 한, 모든 플롯을 현재 활성화된 Figure에 추가
*   plt.figure()로 생성한 Figure는 plt.show() 전까지 유지되므로, 이후의 모든 플롯은 해당 창에 그려집니다.

```python
# 그룹 간 차이 비교
# swarmplot 함수를 사용하여 군집 분산 플롯(swarm plot)을 생성하는 것
# 데이터 포인트를 중복 없이 (겹치지 않게) 분산시켜 표시
plt.figure(figsize=(10, 6))
sns.boxplot(data=tips, x="day", y="total_bill")
sns.swarmplot(data=tips, x="day", y="total_bill", color="black", alpha=0.5)
plt.title('요일별 계산서 분포 (개별 데이터 포인트 포함)')
plt.show()
```

**4. 사용자 정의 함수와 Seaborn**

```python
# 사용자 정의 데이터로 시각화
def create_sample_data():
    """샘플 데이터 생성"""
    np.random.seed(42)
    data = {
        'group': ['A'] * 100 + ['B'] * 100 + ['C'] * 100,
        'value1': np.concatenate([
            np.random.normal(10, 2, 100),
            np.random.normal(12, 3, 100),
            np.random.normal(8, 1.5, 100)
        ]),
        'value2': np.concatenate([
            np.random.normal(5, 1, 100),
            np.random.normal(7, 2, 100),
            np.random.normal(4, 0.8, 100)
        ])
    }
    return pd.DataFrame(data)

# 데이터 생성 및 시각화
df = create_sample_data()

# 다중 비교 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 박스 플롯
sns.boxplot(data=df, x="group", y="value1", ax=axes[0,0])
axes[0,0].set_title('그룹별 Value1 분포')

# 바이올린 플롯
sns.violinplot(data=df, x="group", y="value2", ax=axes[0,1])
axes[0,1].set_title('그룹별 Value2 분포')

# 산점도
sns.scatterplot(data=df, x="value1", y="value2", hue="group", ax=axes[1,0])
axes[1,0].set_title('Value1 vs Value2')

# 히스토그램
sns.histplot(data=df, x="value1", hue="group", multiple="stack", ax=axes[1,1])
axes[1,1].set_title('그룹별 Value1 히스토그램')

plt.tight_layout()
plt.show()
```


### 17.6 주요 팁과 모범 사례

```python
# 1. 데이터 전처리와 함께 사용
tips_clean = tips.copy()
tips_clean['tip_rate'] = tips_clean['tip'] / tips_clean['total_bill'] * 100

plt.figure(figsize=(10, 6))
sns.boxplot(data=tips_clean, x="day", y="tip_rate", hue="time")
plt.title('요일과 시간대별 팁 비율')
plt.ylabel('팁 비율 (%)')
plt.show()

# 2. 통계 정보 추가
# Seaborn의 핵심 단일 플롯(barplot, lineplot, scatterplot 등)은 대부분 Axes 객체를 반환하며,
# 그리드 기반 플롯(catplot, relplot 등)은 FacetGrid나 JointGrid 같은 별도의 객체를 반환합니다.

plt.figure(figsize=(8, 6))
ax = sns.barplot(data=tips, x="day", y="total_bill", estimator=np.mean, errorbar=('ci',95))
plt.title('요일별 평균 계산서 (95% 신뢰구간)')

# 막대 위에 값 표시
# ax.patches : 막대 그래프의 각 막대 객체(Rectangle)를 리스트로 반환
for i, bar in enumerate(ax.patches):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
           f'{height:.1f}', ha='center', va='bottom')

plt.show()
```
