---
title: 1차시 17(빅데이터 분석):R Programming Basic 1
layout: single
classes: wide
categories:
  - R programming
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

# 1. R 프로그래밍 소개

## 1. R의 구성 
### 1.1 **R과 RStudio 설치**
- **R 설치**:  
  - [CRAN 웹사이트](https://cran.r-project.org/)에 접속하여 운영체제에 맞는 버전(R for Windows, macOS, Linux)을 다운로드 및 설치합니다.
  - 설치 중 특별한 설정 변경 없이 기본 옵션으로 진행하면 됩니다.

- **RStudio 설치**:  
  - [RStudio 다운로드 페이지](https://posit.co/download/rstudio-desktop/)에서 무료 버전(RStudio Desktop Free)을 다운로드합니다.
  - 설치 역시 기본 설정으로 진행합니다.


### 1.2 **RStudio의 주요 메뉴 및 창 설명**
1. **좌측 상단: 스크립트/콘솔 창**  
   - **스크립트 창**: 코드를 작성하고 저장할 수 있는 공간입니다. `.R` 파일로 저장됩니다.
   - **콘솔 창**: 직접 코드를 입력하고 즉시 실행 결과를 확인할 수 있습니다.

2. **우측 상단: 환경(Workspace) 및 히스토리**  
   - **Environment 탭**: 현재 작업 중인 변수, 데이터 프레임 등을 확인할 수 있습니다.
   - **History 탭**: 이전에 실행한 명령어 기록을 볼 수 있습니다.

3. **좌측 하단: 콘솔 및 출력 창**  
   - **Console**: 코드 실행 결과가 표시됩니다.

4. **우측 하단: 파일, 플롯, 패키지, 도움말**  
   - **Files 탭**: 작업 디렉토리 내 파일 목록을 확인할 수 있습니다.
   - **Packages 탭**: 설치된 패키지 목록과 로드 상태를 확인할 수 있습니다.
   - **Help 탭**: 함수 또는 패키지에 대한 도움말을 검색할 수 있습니다.
   - **Plots 탭**: 그래프나 시각화 결과가 표시됩니다.

### 1.3 **Working Directory 설정**
1. **현재 작업 디렉토리 확인**  
   ```R
   getwd()
   ```
   - 현재 작업 디렉토리 경로를 확인합니다.

2. **작업 디렉토리 변경**  
   - 코드로 변경:
     ```R
     setwd("C:/사용자/폴더명")
     ```
   - 또는 RStudio 메뉴를 사용:
     - `Session > Set Working Directory > Choose Directory...`를 통해 GUI로 선택합니다.

3. **파일 저장 위치 확인**  
   - 작업 디렉토리에 저장된 파일은 `Files 탭`에서 확인 가능합니다.

### 1.4 **간단한 실습**

```R
# 간단한 계산
print(2 + 3)

# 변수 생성 및 출력
x <- 10
print(x)

# 간단한 벡터 생성
my_vector <- c(1, 2, 3, 4, 5)
print(my_vector)

# 작업 디렉토리에 파일 저장
write.csv(my_vector, "my_vector.csv")
```


### 1.5 **추가 팁**
- **패키지 설치 및 로드**:  새로운 패키지를 설치하고 사용하는 방법을 간단히 소개합니다.
  ```R
  install.packages("dplyr")  # 패키지 설치
  library(dplyr)             # 패키지 로드
  ```

- **도움말 활용**:  특정 함수에 대한 도움말을 보는 방법을 알려줍니다.
  ```R
  ?mean  # mean 함수에 대한 도움말 보기
  ```


## 2. R의 기초

### **2.1 R의 기본 구조 및 특징**
- **인터프리터 언어**: 코드를 한 줄씩 실행하며 즉시 결과를 확인할 수 있습니다.
- **대소문자 구분**: R은 대소문자를 엄격히 구분합니다(예: `myVar`와 `myvar`는 서로 다른 변수).
- **주석 사용**: `#` 기호를 사용하여 주석을 작성합니다. 코드를 읽기 쉽게 만드는 데 중요합니다.

```R
# 이건 주석입니다. R은 이를 무시합니다.
print("Hello, World!")  # 출력 함수
```

### **2.2 변수와 할당 연산자**
변수와 값을 연결하는 방법을 설명합니다. 특히 R에서 사용되는 `<-` 연산자가 독특
- **할당 연산자**:
  - `<-`: 가장 일반적으로 사용됩니다.
  - `=`: 특정 상황에서 사용 가능하지만, `<-`를 권장합니다.
- **변수 이름 규칙**:
  - 알파벳으로 시작해야 하며, 숫자와 밑줄(`_`)을 포함할 수 있습니다.
  - 공백이나 특수문자는 사용할 수 없습니다.

```R
# 변수에 값 할당하기
x <- 10
y = 20  # 가능하지만 비추천
z <- x + y

# 결과 출력
print(z)  # 30
```

### **2.3 데이터 타입**
R에서 자주 사용되는 기본 데이터 타입을 소개
- **숫자형(Numeric)**: 정수 또는 실수.
- **문자형(Character)**: 텍스트 데이터.
- **논리형(Logical)**: `TRUE` 또는 `FALSE`.

```R
num <- 42          # 숫자형
text <- "Hello"    # 문자형
is_true <- TRUE    # 논리형

# 데이터 타입 확인
class(num)         # "numeric"
class(text)        # "character"
class(is_true)     # "logical"
```


### **2.4 벡터(Vector)**
R에서 가장 기본적인 데이터 구조인 벡터를 설명합니다. 벡터는 같은 타입의 데이터를 담는 1차원 배열입니다.
- **벡터 생성**: `c()` 함수를 사용합니다.
- **연산**: 벡터는 요소별로 연산이 가능합니다.

```R
vec <- c(1, 2, 3, 4)  # 숫자형 벡터
print(vec)

# 벡터 연산
vec_times_two <- vec * 2
print(vec_times_two)  # [2, 4, 6, 8]

# 인덱싱
# 파이썬과 달리 인덱스 1부터 시작하고, 슬라이싱할 때 마지막 요소도 inclusive
print(vec[1])  # 첫 번째 요소: 1
print(vec[2:4])  # 두 번째부터 네 번째 요소: [2, 3, 4]
```


### **2.5 조건문과 반복문**
조건문과 반복문은 프로그래밍의 기본 도구입니다. R에서도 유사한 방식으로 작동합니다.
- **조건문**: `if`, `else`를 사용합니다.
- **반복문**: `for`, `while`을 사용합니다.

```R
# 조건문
x <- 10
if (x > 5) {
  print("x는 5보다 큽니다.")
} else {
  print("x는 5보다 작거나 같습니다.")
}

# 반복문
for (i in 1:5) {
  print(paste("현재 숫자:", i))
}
```


### **2.6 함수(Function)**
함수는 재사용 가능한 코드 블록입니다. R에는 많은 내장 함수가 있으며, 사용자 정의 함수도 만들 수 있습니다.
- **내장 함수**: `sum()`, `mean()`, `length()` 등.
- **사용자 정의 함수**: `function()` 키워드를 사용합니다.

```R
# 내장 함수 사용
numbers <- c(1, 2, 3, 4, 5)
print(sum(numbers))  # 합계: 15
print(mean(numbers)) # 평균: 3

# 사용자 정의 함수
greet <- function(name) {
  return(paste("안녕하세요,", name, "님!"))
}
print(greet("학생"))  # "안녕하세요, 학생 님!"
```


### **2.7 패키지 설치 및 사용**
- **패키지 설치**: `install.packages("패키지명")`
- **패키지 로드**: `library(패키지명)`

```R
# dplyr 패키지 설치 및 로드
install.packages("dplyr")
library(dplyr)

# 간단한 데이터 조작 예제
# data.frame()은 내장함수로, 함수 내부에서 마침표(.)를 사용하여 가시성을 좋게 한다.
data <- data.frame(a = c(1, 2, 3), b = c(4, 5, 6))
filtered_data <- filter(data, a > 1)
print(filtered_data)
```


### **2.8 연습문제**
1. **1부터 10까지의 숫자를 저장하는 벡터를 만들고, 짝수만 필터링하세요.**
    - **풀이 과정**
        1. `c()` 함수를 사용해 1부터 10까지의 숫자를 포함하는 벡터를 생성합니다.
        2. 벡터에서 짝수만 필터링하기 위해 조건문(`%%` 연산자)을 사용합니다.
        - `%%`는 나머지 연산자를 의미하며, `x %% 2 == 0`은 "x가 2로 나누어 떨어지는가?"를 확인
    - **소스 코드**
        ```R
        # 1부터 10까지의 숫자 벡터 생성
        numbers <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

        # 짝수만 필터링, 조건부 인덱싱
        even_numbers <- numbers[numbers %% 2 == 0]

        # 결과 출력
        print(even_numbers)
        ```
    - **결과**
        ```
        [1]  2  4  6  8 10
        ```

2. **사용자에게 이름을 입력받아 인사말을 출력하는 함수를 작성하세요.**
    - **풀이 과정**
        1. `function()`을 사용해 사용자 정의 함수를 만듭니다.
        2. `readline()` 함수를 사용해 사용자로부터 이름을 입력받습니다.
        3. 입력받은 이름을 문자열과 결합하여 인사말을 출력합니다.
    - **소스 코드**
        ```R
        # 사용자 정의 함수 작성
        greet_user <- function() {
        # 사용자로부터 이름 입력받기
        name <- readline(prompt = "이름을 입력하세요: ")
        
        # 인사말 출력
        message <- paste("안녕하세요,", name, "님!")
        print(message)
        }

        # 함수 실행
        greet_user()
        ```
    - **실행 예시**
        ```
        이름을 입력하세요: 홍길동
        [1] "안녕하세요, 홍길동 님!"
        ```


3. **`mtcars` 데이터셋을 불러와서 `mpg`(연비) 열의 평균을 계산하세요.**
    - **풀이 과정**
        1. `mtcars`는 R에 내장된 데이터셋으로, 자동차 관련 정보를 포함하고 있습니다.
        2. `mean()` 함수를 사용해 `mpg` 열의 평균을 계산합니다.

    - **소스 코드**
        ```R
        # mtcars 데이터셋 로드 (내장 데이터셋이므로 별도 설치 불필요)
        # 기본 데이터셋을 메모리에 로드하는 역할, 반환값 없음 (NULL)
        data(mtcars)

        # mpg 열의 평균 계산
        # mtcars에서 $ 연산자를 사용해 mpg 열(연비, miles per gallon) 만 벡터 형태로 추출
        mpg_mean <- mean(mtcars$mpg)

        # 결과 출력
        print(paste("mpg의 평균:", mpg_mean))
        ```

    - **결과**
        ```
        [1] "mpg의 평균: 20.090625"
        ```

## 3. R의 데이터 구조

### 3.1 **이론**
- **벡터(Vector)**: 동일한 데이터 타입의 1차원 배열.
- **행렬(Matrix)**: 동일한 데이터 타입의 2차원 배열.
- **데이터 프레임(Data Frame)**: 서로 다른 데이터 타입을 포함할 수 있는 2차원 표 형식.
- **리스트(List)**: 다양한 데이터 타입과 구조를 담을 수 있는 유연한 구조.

- **중점 설명**
    - 데이터 프레임은 데이터 분석에서 가장 중요한 구조로, 실제 데이터셋(예: CSV 파일)을 불러오면 일반적으로 데이터 프레임 형태로 저장됩니다.
    - 리스트는 복잡한 데이터를 저장하거나 패키지 함수의 결과를 반환할 때 자주 사용됩니다.

### 3.2 **실습 예제**

1. **벡터와 행렬**
    ```R
    # 벡터 생성 및 연산
    vec <- c(1, 2, 3, 4)
    print(vec * 2)  # 요소별 곱셈

    # 행렬 생성 및 연산
    mat <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)
    print(mat)
    print(mat + 5)  # 요소별 덧셈
    ```

2. **데이터 프레임**
    ```R
    # 데이터 프레임 생성
    df <- data.frame(
    Name = c("Alice", "Bob", "Charlie"),
    Age = c(25, 30, 35),
    Score = c(88, 92, 85)
    )

    # 데이터 프레임 조회
    print(df)
    print(df$Name)  # 특정 열 선택, 항상 벡터로 반환됨 (1차원)
    print(df[1, ])  # 첫 번째 행 선택
    ```

3. **리스트**
    ```R
    # 리스트 생성
    my_list <- list(
    Numbers = c(1, 2, 3),
    Text = "Hello",
    Matrix = matrix(c(1, 2, 3, 4), nrow = 2)
    )

    # 리스트 요소 접근
    print(my_list$Numbers) 
    print(my_list[[2]])  # 두 번째 요소
    ```

## **4. 데이터 조작(dplyr 패키지 활용)**

### 4.1 **이론**
- `dplyr` 패키지는 데이터 프레임을 쉽고 직관적으로 조작할 수 있는 도구
- `filter()`: 조건에 맞는 행 필터링.
- `select()`: 특정 열 선택.
- `mutate()`: 새로운 열 추가 또는 기존 열 수정.
- `summarize()`: 데이터 요약 통계 계산.
- `arrange()`: 데이터 정렬.
- `%>%`(파이프 연산자): `dplyr` 패키지를 로드시 사용, 여러 함수를 연결하여 코드를 간결하게 

### 4.2 **실습 예제**

1. **dplyr 설치 및 로드**
    ```R
    # dplyr 패키지 설치 및 로드
    install.packages("dplyr")
    library(dplyr)
    ```

2. **데이터 필터링 및 선택**
    
    ```R
    # mtcars 데이터셋 사용
    data(mtcars)

    # filter()와 select() 사용
    filtered_data <- mtcars %>%
    filter(cyl == 4) %>%  # 실린더가 4개인 차량만 필터링
    select(mpg, hp)       # mpg와 hp 열만 선택

    print(filtered_data)
    ```

3. **새로운 열 추가 및 요약 통계**
    ```R
    # mutate()와 summarize() 사용
    modified_data <- mtcars %>%
    mutate(kpl = mpg * 0.4251) %>%  # mpg를 km/L로 변환
    summarize(avg_kpl = mean(kpl))  # 평균 kpl 계산
    print(modified_data)

    # group_by()사용
    modified_data <- mtcars %>%
    mutate(kpl = mpg * 0.4251) %>% 
    group_by(cyl) %>%  # 실린더 수별 그룹화
    summarise(avg_kpl = mean(kpl))
    print(modified_data)
    ```

4. **데이터 정렬**
    ```R
    # arrange() 사용
    sorted_data <- mtcars %>%
    arrange(desc(mpg))  # mpg를 내림차순으로 정렬

    print(sorted_data)
    ```

## **5. 데이터 시각화(ggplot2 패키지 활용)**

### 5.1 **이론**
- `ggplot2`는 R에서 데이터 시각화를 위한 강력한 패키지로, 그래프를 층(layer) 단위로 구축
- **aes()**: 미적 매핑(aesthetic mapping). x축, y축, 색상 등을 정의.
- **geom_***(): 그래프 유형(산점도, 막대 그래프 등)을 지정.
- **theme()**: 그래프 스타일(폰트, 배경 등)을 커스터마이징.

### 5.2 **실습 예제**
1. **ggplot2 설치 및 로드**
    ```R
    # ggplot2 패키지 설치 및 로드
    install.packages("ggplot2")
    library(ggplot2)
    ```

2.  **산점도 그리기**
    ```R
    # mtcars 데이터셋 사용
    ggplot(data = mtcars, aes(x = wt, y = mpg)) +
    geom_point(color = "blue") +  # 산점도
    labs(title = "Weight vs MPG", x = "Weight", y = "Miles Per Gallon")
    ```

3. **막대 그래프 그리기**
    ```R
    # iris 데이터셋 사용
    ggplot(data = iris, aes(x = Species, y = Sepal.Length)) +
    geom_bar(stat = "summary", fun = "mean", fill = "orange") +  # 평균 값으로 막대 그래프
    labs(title = "Average Sepal Length by Species")
    ```

4. **히스토그램 그리기**
    ```R
    # mtcars 데이터셋 사용
    ggplot(data = mtcars, aes(x = mpg)) +
    geom_histogram(binwidth = 2, fill = "green", color = "black") +  # 히스토그램
    labs(title = "Distribution of MPG")
    ```

5. **그룹별 시각화**
    ```R
    # mtcars 데이터셋 사용
    ggplot(data = mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
    geom_point(size = 3) +  # 실린더 개수별 색상 구분
    labs(title = "Weight vs MPG by Cylinders", color = "Cylinders")
    ```

# 2. R Programming 실습
## 1. Data Preprocessing

```R
# 필요한 패키지 로드
library(dplyr)
library(caret)
library(ggplot2)

# 1. 데이터 전처리 (Data Preprocessing)
# 샘플 데이터 생성
set.seed(123)
data <- data.frame(
  ID = 1:10,
  Age = c(25, 30, NA, 40, 45, 50, 60, 70, 80, 90),
  Income = c(50000, 60000, 70000, 80000, 90000, 100000, 150000, 200000, 250000, 300000),
  Gender = c("Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"),
  Score = c(85, 90, 95, 100, 105, 110, 115, 120, 125, 130)
)

# 결측값 처리
data_cleaned <- na.omit(data) # 결측값 제거
print("결측값 제거 후 데이터:")
print(data_cleaned)

# 이상치 탐지 및 처리 (IQR 기반)
Q1 <- quantile(data_cleaned$Income, 0.25)
Q3 <- quantile(data_cleaned$Income, 0.75)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR
data_no_outliers <- data_cleaned %>%
  filter(Income >= lower_bound & Income <= upper_bound)
print("이상치 제거 후 데이터:")
print(data_no_outliers)

# 데이터 정규화 (Normalization)
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
data_normalized <- data_no_outliers %>%
  mutate(Age_Normalized = normalize(Age),
         Income_Normalized = normalize(Income))
print("정규화된 데이터:")
print(data_normalized)

# 범주형 변수 인코딩 (One-Hot Encoding)
data_encoded <- dummyVars("~ .", data = data_normalized) %>%
  predict(data_normalized)
print("One-Hot Encoding된 데이터:")
print(data_encoded)

```

1. **데이터 전처리**
    - **결측값 처리**: `na.omit` 함수를 사용하여 결측값을 제거하거나 `caret` 패키지의 `preProcess` 함수를 활용해 결측값을 대체(impute)할 수 있습니다.
    - **이상치 탐지 및 처리**: 사분위수(IQR)를 기반으로 이상치를 탐지하고 필터링합니다.
    - **데이터 정규화/스케일링**: 최소-최대 정규화(Normalization) 또는 표준화(Standardization)를 통해 데이터를 스케일링합니다.
    - **범주형 변수 인코딩**: `dummyVars` 함수를 사용하여 One-Hot Encoding을 수행합니다.
      - `dummyVars` 에서 `predict`는 실제 데이터에 적용하는 과정
      




## 2. Data Visualization

```R
# 2. 데이터 시각화 (Data Visualization)
# 기본 시각화: 산점도, 박스플롯, 히스토그램
ggplot(data_normalized, aes(x = Age, y = Income)) +
  geom_point() +
  labs(title = "Age vs Income (Scatter Plot)")

ggplot(data_normalized, aes(x = Gender, y = Income)) +
  geom_boxplot() +
  labs(title = "Gender vs Income (Box Plot)")

ggplot(data_normalized, aes(x = Income)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Income Distribution (Histogram)")

# 상관관계 행렬 시각화 (Heatmap)
cor_matrix <- cor(select(data_normalized, Age, Income, Score))
# 상관계수 행렬 확인
print(cor_matrix)

library(reshape2)
melted_cor <- melt(cor_matrix)
ggplot(melted_cor, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0.95) +
  labs(title = "Correlation Heatmap")
```
1. **데이터 시각화**
    - **기본 시각화**: `ggplot2`를 사용하여 산점도, 박스플롯, 히스토그램 등을 그립니다.
    - **상관관계 행렬 시각화**: `cor` 함수로 상관관계 행렬을 계산하고, `ggplot2`를 사용하여 Heatmap으로 시각화합니다.


## 3. **Correlation**

```R
# 필요한 패키지 설치 및 로드
if (!require("corrplot")) install.packages("corrplot", dependencies = TRUE)
if (!require("lattice")) install.packages("lattice", dependencies = TRUE)
if (!require("ggplot2")) install.packages("ggplot2", dependencies = TRUE)

library(corrplot)  # 상관행렬 시각화
library(lattice)   # 산점도 행렬 및 기타 그래프
library(ggplot2)   # 고급 시각화 도구

# mtcars 데이터셋 로드 및 확인
data(mtcars)
head(mtcars)  # 데이터 구조 확인
summary(mtcars)  # 요약 통계량 출력

# 특정 변수 간 상관계수 계산
cor_gear_carb <- cor(mtcars$gear, mtcars$carb)
cat("Correlation between gear and carb:", round(cor_gear_carb, 2), "\n")

# 전체 변수 간 상관행렬 계산 및 반올림
cor_matrix <- cor(mtcars)
rounded_cor_matrix <- round(cor_matrix, 2)
print(rounded_cor_matrix)

# gear와 carb 간 산점도 그리기
# xyplot()은 lattice 패키지에 속한 함수로, 조건부 산점도(conditional scatter plot)를 그릴 때 사용
xyplot(gear ~ carb, data = mtcars, main = "Scatterplot of Gear vs Carb",
       xlab = "Carburetors", ylab = "Gears")

# 산점도와 선형 회귀선 추가
#기본 그래픽 시스템을 사용
plot(mtcars$carb, mtcars$gear, main = "Gear vs Carb with Regression Line",
     xlab = "Carburetors", ylab = "Gears", pch = 19, col = "blue")
# abline(): 회귀선 추가
# lm(): carb를 설명 변수(X), gear를 반응 변수(Y)로 하는 선형 회귀 모델(Linear Model)을 적합(fit)
abline(lm(gear ~ carb, data = mtcars), col = "red", lwd = 2)


# 상관행렬 시각화
# corrplot() 함수는 corrplot 패키지에 속하며, 상관 계수 행렬을 시각화
# tl.cex는 tick label(축라벨)의 character extension(기본 1.0)
corrplot(cor_matrix, method = "circle", type = "upper", 
         tl.cex = 0.8, title = "Correlation Matrix of mtcars")

# gear와 carb 간 산점도
# theme_minimal() 은 ggplot2 패키지에서 제공하는 그래프 테마(theme) 중 하나로,그래프를 간결하고 깔끔하게 
ggplot(mtcars, aes(x = gear, y = carb)) +
  geom_point(color = "darkgreen", size = 3) +
  labs(title = "Scatterplot of Gear vs Carb", x = "Gears", y = "Carburetors") +
  theme_minimal()


# wt와 mpg 간 산점도 (carb로 색상 구분)
# scale_color_brewer(): 범주형 데이터의 색상을 ColorBrewer의 "Set1" 팔레트로 지정하는 함수
ggplot(mtcars, aes(x = wt, y = mpg, color = factor(carb))) +
  geom_point(size = 3) +
  scale_color_brewer(palette = "Set1") +
  labs(title = "Weight vs MPG (Colored by Carburetors)", x = "Weight", y = "MPG") +
  theme_minimal()

# wt와 mpg 간 상관계수 계산
# 화면에 보여주려면 → cat()
# 문자열을 만들어서 쓰려면 → paste(), 반환값 있음
cor_wt_mpg <- cor(mtcars$wt, mtcars$mpg)
cat("Correlation between Weight and MPG:", round(cor_wt_mpg, 2), "\n")

```

1. **패키지 관리**:  
   - `if (!require(...))` 구문을 사용하여 패키지가 설치되어 있지 않을 경우 자동으로 설치하도록 처리했습니다.
   
2. **데이터 탐색**:  
   - `head()`와 `summary()` 함수를 통해 데이터셋의 구조와 요약 통계를 확인할 수 있습니다.

3. **상관관계 계산**:  
   - `cor()` 함수를 사용하여 특정 변수 간 상관계수와 전체 상관행렬을 계산하고, `round()` 함수로 소수점 2자리까지 반올림하여 가독성을 높였습니다.

4. **시각화**:  
   - `lattice`, `plot()`, `corrplot`, `ggplot2` 등 다양한 시각화 도구를 활용하여 데이터를 다각도로 분석했습니다.
   - 특히 `ggplot2`에서는 색상(`color`)을 활용하여 데이터를 더욱 직관적으로 표현했습니다.


## 4. **Regression**

```R
# 한글 폰트 사용시
par(family = "NanumGothic")  # macOS/Linux (Nanum 폰트 설치 시)

# 1. 데이터 구성: 근무 연수와 연봉 데이터를 생성하고 데이터프레임으로 결합
work_experience <- c(26, 16, 20, 7, 22, 15, 29, 28, 17, 3, 1, 16, 19, 13, 27, 4, 30, 8, 3, 12)
income <- c(1267, 887, 1022, 511, 1193, 795, 1713, 1477, 991, 455, 324, 944, 1232, 808, 1296, 486, 1516, 565, 299, 830)
salary_data <- data.frame(work_experience, income)

# 2. 데이터 요약 통계: 데이터의 기초 통계량(최소값, 최대값, 평균 등) 확인
print("기초 통계량:")
summary(salary_data)

# 3. 산점도 시각화: 근무 연수와 연봉 간의 관계를 그래프로 확인
plot(work_experience, income, main = "근무 연수와 연봉의 관계", xlab = "근무 연수", ylab = "연봉")

# 4. 상관관계 분석: 근무 연수와 연봉 간의 선형 관계 강도를 상관계수로 확인
correlation <- cor(work_experience, income)
cat("근무 연수와 연봉 간 상관계수:", correlation, "\n")

# 5. 선형 회귀 모델 생성: 근무 연수가 연봉에 미치는 영향을 분석
linear_model <- lm(income ~ work_experience, data = salary_data)

# 6. 회귀 분석 결과 요약: 모델의 기울기, 절편, 유의미성 등을 확인
cat("선형 회귀 분석 결과:\n")
summary(linear_model)
```
```

Call:
lm(formula = income ~ work_experience, data = salary_data)

Residuals:
     Min       1Q   Median       3Q      Max 
-115.282  -59.636   -3.018   37.011  215.873 

Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
(Intercept)      252.375     39.766   6.346 5.59e-06 ***
work_experience   42.922      2.179  19.700 1.25e-13 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 89.02 on 18 degrees of freedom
Multiple R-squared:  0.9557,	Adjusted R-squared:  0.9532 
F-statistic: 388.1 on 1 and 18 DF,  p-value: 1.25e-13
```

```
- Coefficients
   - Estimate: 회귀 계수 추정값
      - (Intercept) = 252.375: 근무 경력이 0년일 때 예상 연봉 (기본 급여)
      - work_experience = 42.922: 근무 경력이 1년 증가할 때마다 연봉이 약 42.92 증가
   - Pr(>|t|)(p-value): 두 변수 모두 (***) 표시 → p-value < 0.001
      - 절편과 근무 경력 모두 통계적으로 매우 유의미
- 모델 설명력 (R-squared): R² = 0.9557, 근무 경력이 연봉 변동의 95.6%를 설명
- 전체 모델의 유의성(F-test): p-value ≈ 0 → 전체 회귀 모델이 통계적으로 매우 유의미
- 잔차(Residuals) — 모델 적합도 확인:
   -  Median이 0에 가까움 → 좋은 신호
   - Max와 Min의 크기 차이가 크면 이상치(outlier) 의심
```

**1. 데이터 구성**
- `work_experience`와 `income` 변수는 각각 근무 연수와 연봉 데이터를 저장합니다.
- `data.frame()` 함수를 사용하여 두 변수를 결합한 데이터프레임 `salary_data`를 생성합니다.
- 이 데이터프레임은 이후 분석의 기초 자료로 사용됩니다.

**2. 데이터 요약 통계**
- `summary()` 함수를 통해 데이터의 기초 통계량(최소값, 1사분위수, 중앙값, 평균, 3사분위수, 최대값)을 확인.
- 이를 통해 데이터의 전체적인 분포와 특성을 파악할 수 있습니다.

**3. 산점도 시각화**
- `plot()` 함수를 사용하여 근무 연수(`work_experience`)와 연봉(`income`) 간의 관계를 시각적으로 확인.
- 산점도는 두 변수 간의 패턴이나 경향성을 탐색하는 데 유용합니다.

**4. 상관관계 분석**
- `cor()` 함수를 사용하여 근무 연수와 연봉 간의 상관계수를 계산합니다.
- 상관계수는 두 변수 간의 선형 관계 강도를 나타내며, 값이 1에 가까울수록 강한 양의 상관관계를 의미합니다.

**5. 선형 회귀 모델 생성**
- `lm()` 함수를 사용하여 선형 회귀 모델을 생성합니다.
- 모델 식 `income ~ work_experience`는 근무 연수를 독립변수로, 연봉을 종속변수로 설정하여 근무 연수가 연봉에 미치는 영향을 분석합니다.

**6. 회귀 분석 결과 요약**
- `summary()` 함수를 통해 선형 회귀 모델의 결과를 요약합니다.
- 출력 결과에는 모델의 기울기(회귀 계수), 절편, 결정계수(R-squared), p-value 등이 포함되어 모델의 적합성과 통계적 유의성을 평가할 수 있습니다.
   

## 5. **ANOVA**

```R
# 데이터셋 로드 및 패키지 설치
data("PlantGrowth")  # 기본 제공 데이터셋 PlantGrowth 사용
head(PlantGrowth)

# car 패키지는 회귀 분석 진단 및 확장 기능을 제공,
# 주로 잔차 분석, 다중공선성 검토(VIF), 이상치 탐지, 가정 검정(예: 등분산성) 등
install.packages("car")  # car 패키지 설치
library(car)

# 분산 동질성 검정 (Levene Test)
# p-value = 0.3412 (> 0.05) → 분산의 차이가 통계적으로 유의미하지 않음
leveneTest(weight ~ group, data = PlantGrowth)

# 방법 1: aov를 사용한 ANOVA 분석
anova_result1 <- aov(weight ~ group, data = PlantGrowth)
summary(anova_result1)

# 방법 2: lm과 anova를 조합한 분석
# anova(lm(...))은 이미 요약된 분석 결과를 반환하므로 summary() 불요
anova_result2 <- anova(lm(weight ~ group, data = PlantGrowth))
anova_result2

# 방법 3: oneway.test를 사용한 분석
# 등분산성 가정 없이 수행한 일원분산분석(Welch’s ANOVA)으로 등분산성이 성립되지 않을 때 대안 방식
oneway_result <- oneway.test(weight ~ group, data = PlantGrowth)
oneway_result
```
```
# 각 그룹의 분산이 동일한지(등분산성)를 검정
> leveneTest(weight ~ group, data = PlantGrowth)
Levene's Test for Homogeneity of Variance (center = median)
      Df  F value  Pr(>F)
group  2  1.1192   0.3412
      27  
# p-value > 0.05 이므로 그룹간 유의미한 차이 없음

> summary(anova_result1)
            Df Sum Sq Mean Sq F value Pr(>F)  
group        2  3.766  1.8832   4.846 0.0159 *
Residuals   27 10.492  0.3886                 
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# p-value < 0.05 이므로 그룹간 유의미한 차이 있음

> anova_result2
Analysis of Variance Table

Response: weight
          Df  Sum Sq Mean Sq F value  Pr(>F)  
group      2  3.7663  1.8832  4.8461 0.01591 *
Residuals 27 10.4921  0.3886                  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# p-value < 0.05 이므로 그룹간 유의미한 차이 있음

> oneway_result
	One-way analysis of means (not assuming equal variances)

data:  weight and group
F = 5.181, num df = 2.000, denom df = 17.128, p-value = 0.01739
# p-value < 0.05 이므로 그룹간 유의미한 차이 있음
```

1. **데이터셋 로드**: `PlantGrowth` 데이터셋을 사용하며, 식물의 그룹(group)과 무게(weight) 변수를 분석 대상으로 합니다.
2. **분산 동질성 검정**: `leveneTest` 함수로 각 그룹 간 분산의 동일성을 검정합니다. 이는 ANOVA의 전제 조건인 등분산성을 확인하는 과정입니다.
3. **ANOVA 분석**:
   - `aov`: 기본적인 일원분산분석을 수행하여 그룹 간 평균 차이를 검정합니다.
   - `lm` + `anova`: 선형 모델을 활용한 분산분석으로 동일한 가설을 검증합니다.
   - `oneway.test`: 등분산성을 가정하지 않는 경우에도 적용 가능한 방법으로, 그룹 간 평균 차이를 평가합니다.
4. **결과 출력**: 각 분석 결과를 통해 그룹 간 평균 차이의 통계적 유의성을 확인할 수 있습니다.


## 6. **PCA**

```R
# 데이터 생성: 16명의 학생이 응시한 네 과목(국어, 영어, 수학, 과학) 점수를 벡터로 정의
kor = c(26, 46, 57, 36, 57, 26, 58, 37, 36, 56, 78, 95, 88, 90, 52, 56)
eng = c(35, 74, 73, 73, 62, 22, 67, 34, 22, 42, 65, 88, 90, 85, 46, 66)
math = c(35, 76, 38, 69, 25, 25, 87, 79, 36, 26, 22, 36, 58, 36, 25, 44)
sci = c(45, 89, 54, 55, 33, 45, 67, 89, 47, 36, 40, 56, 68, 45, 37, 56)

# 데이터 결합: 네 벡터를 열로 결합하여 데이터 프레임 생성
student_scores = data.frame(kor, eng, math, sci)

# 열 및 행 이름 지정: 열 이름을 과목명으로, 행 이름을 학생 번호로 설정
# 데이터 프레임(또는 행렬)의 열 이름(column names)을 확인하거나 변경
colnames(student_scores) = c("국어", "영어", "수학", "과학")
# paste(): 기본 구분자가 공백
# paste0(): 기본 구분자 없음
rownames(student_scores) = paste0("학생", 1:16) 

# 데이터 확인: 데이터의 처음 몇 행을 출력
head(student_scores)

# 주성분 분석 수행: prcomp 함수를 사용하여 PCA 실행
# student_scores 데이터를 표준화(scale. = TRUE)한 후 주성분 분석(PCA)을 수행
pca_result = prcomp(student_scores, scale. = TRUE)

# 결과 요약: 주성분 분석 결과를 요약하여 출력
summary(pca_result)

# 어떤 변수가 PC1과 PC2를 구성하는 데 가장 큰 영향을 미치는지 구체적으로 파악
# PCA에서 rotation(회전)은 주성분이 원래 변수들과 어떻게 선형 결합되는지를 나타내는 계수 행렬
pca_result$rotation
```
```
> head(student_scores)
      국어 영어 수학 과학
학생1   26   35   35   45
학생2   46   74   76   89
학생3   57   73   38   54
학생4   36   73   69   55
학생5   57   62   25   33
학생6   26   22   25   45

> summary(pca_result)
Importance of components:
                          PC1    PC2     PC3    PC4
Standard deviation     1.4154 1.3087 0.43779 0.3040
Proportion of Variance 0.5008 0.4281 0.04791 0.0231
Cumulative Proportion  0.5008 0.9290 0.97690 1.0000
# PC1과 PC2가 전체 분산의 **92.9%**를 설명함 (PC1: 50.1%, PC2: 42.8%)

> pca_result$rotation
           PC1        PC2        PC3        PC4
국어 0.2388128 -0.6895993 -0.5325178 -0.4287728
영어 0.4604720 -0.5393126  0.5603653  0.4278997
수학 0.6038420  0.3514805  0.3277028 -0.6359616
과학 0.6052345  0.3317472 -0.5431634  0.4781303
# PC1에서 수학과 과학의 계수가 크고 비슷함 → PC1은 전반적 학업 성향을 반영
# PC2는 국어(음수) → 문과 vs 이과 성향 차이를 나타낼 수 있음
```

1. **데이터 생성**: `kor`, `eng`, `math`, `sci`는 각각 국어, 영어, 수학, 과학 점수를 나타내는 벡터로, 16명의 학생 데이터를 포함합니다.
2. **데이터 결합**: `data.frame`을 사용해 네 벡터를 결합하고, `student_scores`라는 데이터 프레임으로 저장.
3. **열 및 행 이름 지정**: 열 이름은 과목명(국어, 영어, 수학, 과학)으로, 행 이름은 "학생1", "학생2"와 같이 학생 번호로 지정합니다.
4. **데이터 확인**: `head(student_scores)`로 데이터의 처음 몇 행을 확인합니다.
5. **주성분 분석 수행**: `prcomp` 함수를 사용하여 PCA를 실행해, `scale. = TRUE` 옵션을 추가하여 데이터를 표준화
6. **결과 요약**: `summary(pca_result)`로 주성분 분석 결과를 요약해 출력(예: 각 주성분의 설명 분산 비율 등).


## 7. **Logistic Regression**

```R
# 데이터 생성 및 준비
drug_dose <- c(1, 1, 2, 2, 3, 3)  # 약물 용량
patient_response <- c(0, 1, 0, 1, 0, 1)  # 환자 반응 (0: 없음, 1: 있음)
frequency <- c(7, 3, 5, 5, 2, 8)  # 각 조건의 발생 빈도
experiment_data <- data.frame(drug_dose, patient_response, frequency)

# 로지스틱 회귀 모델 생성
# Generalized Linear Model (일반화 선형 모델)
logistic_model <- glm(patient_response ~ drug_dose, # 종속변수(환자 반응)를 독립변수(약물 용량)로 예측
                      weights = frequency, # 각 관측치에 빈도 가중치 적용
                      family = binomial(link = "logit"), # 이진 결과(성공/실패)를 로지스틱 함수로 모델링
                      data = experiment_data) # 일반화 선형 모델 함수

# 모델 요약 정보 출력
print(summary(logistic_model))

# 예측 확률 시각화
plot(patient_response ~ drug_dose, data = experiment_data, 
     type = 'n', xlab = "Drug Dose", ylab = "Response Probability", 
     main = "Predicted Response Probability by Drug Dose")
curve(predict(logistic_model, newdata = data.frame(drug_dose = x), type = "response"), 
      add = TRUE, col = "blue", lwd = 2)
```

```
> print(summary(logistic_model))

Call:
glm(formula = patient_response ~ drug_dose, family = binomial(link = "logit"), 
    data = experiment_data, weights = frequency)

Coefficients:
            Estimate Std. Error z value Pr(>|z|)  
(Intercept)  -2.0496     1.0888  -1.882   0.0598 .
drug_dose     1.1051     0.5186   2.131   0.0331 *
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 41.455  on 5  degrees of freedom
Residual deviance: 36.196  on 4  degrees of freedom
AIC: 40.196

Number of Fisher Scoring iterations: 4

# 약물 용량은 환자 반응에 통계적으로 유의한 양의 효과가 있다.
# Deviance 감소 = 모델의 설명력 향상
# 알고리즘이 4번의 반복 후 수렴했다는 뜻
```

1. **데이터 준비**:  
   - `drug_dose`, `patient_response`, `frequency`라는 변수를 정의하고, 이를 `experiment_data`라는 데이터프레임으로 구성합니다.     

2. **모델 생성**:  
   - `glm()` 함수를 사용해 로지스틱 회귀 모델을 생성하며, `link = "logit"`을 명시적으로 추가하여 이항 분포와 로짓 링크 함수를 사용함을 강조했습니다.  
   - glm 함수 시그니처
      - glm(formula, family, data, weights, ...)
      - formula: 모델 공식 (y ~ x)
      - family: 분포와 링크 함수
      - data: 데이터프레임
      - weights: 각 관측치의 빈도/가중치 

3. **결과 확인**:  
   - `summary(logistic_model)`로 모델의 계수, 유의성 등을 출력합니다.  

4. **시각화**:  
   - `plot()`에서 축 레이블(`xlab`, `ylab`)을 추가하여 그래프의 설명력을 높였습니다.  
      - 'n' = "no plotting" (점이나 선을 그리지 않음)
   - `curve()`에서 `col = "blue"`와 `lwd = 2`를 사용해 예측 확률 곡선을 시각적으로 강조했습니다.
      - type = "response", 로짓값이 아닌 실제 확률값(0~1)으로 변환
      - add = TRUE, 새 그래프를 그리지 않고 기존 그래프에 추가
      - col = "blue", lwd = 2, 파란색, 선 두께 2


## 8. **Prediction Analytics**

```R
# 의사결정나무 분석을 위한 패키지 설치 및 로드
if (!require("rpart")) install.packages("rpart", dependencies = TRUE)
library(rpart)

# iris 데이터셋 확인 및 모델링 준비
data(iris) # 내장 데이터셋 로드
str(iris)  # 데이터 구조 확인, str은 structure의 약어
summary(iris) # 데이터 요약 정보

# Species를 종속 변수로 설정하고 나머지 변수를 독립 변수로 사용
# 붓꽃의 종(Species)을 꽃받침과 꽃잎의 측정값들을 기반으로 예측하겠다
# as.formula() 함수는 문자형(string)을 R의 공식(formula) 객체로 변환하는 함수
tree_formula <- as.formula("Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
decision_tree <- rpart(tree_formula, data = iris, method = "class") 

# 의사결정나무 시각화
# png()와 dev.off()는 하나의 세트를 구성(시작과 끝)
png("decision_tree.png", width = 800, height = 600) # 그래프를 PNG 파일로 저장
plot(decision_tree, uniform = TRUE, main = "Decision Tree for Iris Dataset")
text(decision_tree, use.n = TRUE, cex = 0.8, col = "blue") # 노드 레이블 추가
dev.off() # 그래픽 장치 종료

# 모델 결과 출력
print(decision_tree)
```
```
> str(iris)  # 데이터 구조 확인, str은 structure의 약어
'data.frame':	150 obs. of  5 variables:
 $ Sepal(꽃받침).Length: num  5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
 $ Sepal.Width : num  3.5 3 3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 ...
 $ Petal(꽃잎).Length: num  1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 ...
 $ Petal.Width : num  0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 ...
 $ Species     : Factor w/ 3 levels "setosa","versicolor",..: 1 1 1 1 1 1 1 1 1 1 ...

> summary(iris) # 데이터 요약 정보
  Sepal.Length    Sepal.Width     Petal.Length    Petal.Width          Species  
 Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100   setosa    :50  
 1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300   versicolor:50  
 Median :5.800   Median :3.000   Median :4.350   Median :1.300   virginica :50  
 Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199                  
 3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800                  
 Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500   

```

1. **패키지 관리**: `require()`를 사용하여 `rpart` 패키지가 이미 설치되어 있는지 확인하고, 없으면 설치합니다.
   - 재귀적 분할 및 회귀 트리(Recursive Partitioning and Regression Trees) 알고리즘을 구현한 패키지로, 의사결정나무(Decision Tree) 모델을 구축
2. **데이터 확인**: `str()`과 `summary()`를 통해 데이터 구조와 요약 정보를 확인하며, 데이터셋의 이해도를 높입니다.
3. **모델 정의**: `as.formula()`를 사용하여 명시적으로 공식을 정의하고, `rpart()` 함수로 의사결정나무 모델을 생성합니다.
   - method='class': 분류 문제로 설정 (범주형 예측)
4. **시각화**: `png()`를 사용하여 그래프를 파일로 저장하며, `uniform = TRUE` 옵션으로 나무의 깊이에 따른 노드 크기 차이를 줄입니다.
   - uniform = TRUE: 노드 간격 균일하게 조정
   - main = ...: 그래프 제목 설정
   - use.n = TRUE: 각 노드의 관측치 개수 표시
   - cex = 0.8: 텍스트 크기 80%로 설정
   - dev.off(): 그래픽 출력 종료 및 파일 저장 완료, PNG 파일 생성 마무리
5. **결과 출력**: `print()`로 모델의 텍스트 결과를 콘솔에 출력합니다.

```
> print(decision_tree)
n= 150 

node), split, n, loss, yval, (yprob)
      * denotes terminal node

1) root 150 100 setosa (0.33333333 0.33333333 0.33333333)  
2) Petal.Length< 2.45 50   0 setosa (1.00000000 0.00000000 0.00000000) *
3) Petal.Length>=2.45 100  50 versicolor (0.00000000 0.50000000 0.50000000)  
6) Petal.Width< 1.75 54   5 versicolor (0.00000000 0.90740741 0.09259259) *
7) Petal.Width>=1.75 46   1 virginica (0.00000000 0.02173913 0.97826087) *
```

- **의사결정나무 구조 요약 설명**
   - 이 결과는 **iris 데이터를 분류한 의사결정나무 모델의 규칙**을 보여줍니다.
- **주요 분기 규칙:**
   - **첫 번째 분기**: `Petal.Length < 2.45`
      - 50개 샘플 모두 **setosa**로 완벽 분류
   - **두 번째 분기**: `Petal.Length >= 2.45` 중에서
      - `Petal.Width < 1.75`: 54개 중 49개 **versicolor** (91% 정확도)
      - `Petal.Width >= 1.75`: 46개 중 45개 **virginica** (98% 정확도)
- **핵심 통찰:**
   - **가장 중요한 변수**: 꽃잎 길이(Petal.Length)
   - **두 번째 중요한 변수**: 꽃잎 너비(Petal.Width)
   - **단 2개의 규칙**으로 150개 샘플을 95% 이상 정확하게 분류

**즉, 꽃잎 측정값만으로 붓꽃 종류를 효과적으로 구분할 수 있음을 보여줍니다.**


