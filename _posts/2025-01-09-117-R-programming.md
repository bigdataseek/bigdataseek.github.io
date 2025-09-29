---
title: 1차시 17(빅데이터 분석):R Programming Basic 
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
   - **Plots 탭**: 그래프나 시각화 결과가 표시됩니다.
   - **Files/Help 탭**: 도움말이나 파일 탐색기 기능을 제공합니다.

4. **우측 하단: 파일, 플롯, 패키지, 도움말**  
   - **Files 탭**: 작업 디렉토리 내 파일 목록을 확인할 수 있습니다.
   - **Packages 탭**: 설치된 패키지 목록과 로드 상태를 확인할 수 있습니다.
   - **Help 탭**: 함수 또는 패키지에 대한 도움말을 검색할 수 있습니다.

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
data <- data.frame(a = c(1, 2, 3), b = c(4, 5, 6))
filtered_data <- filter(data, a > 1)
print(filtered_data)
```


### **2.8 연습문제**
1. **1부터 10까지의 숫자를 저장하는 벡터를 만들고, 짝수만 필터링하세요.**
    - **풀이 과정**
        1. `c()` 함수를 사용해 1부터 10까지의 숫자를 포함하는 벡터를 생성합니다.
        2. 벡터에서 짝수만 필터링하기 위해 조건문(`%%` 연산자)을 사용합니다.
        - `%`는 나머지 연산자를 의미하며, `x %% 2 == 0`은 "x가 2로 나누어 떨어지는가?"를 확인
    - **소스 코드**
        ```R
        # 1부터 10까지의 숫자 벡터 생성
        numbers <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

        # 짝수만 필터링
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
        data(mtcars)

        # mpg 열의 평균 계산
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
    print(df$Name)  # 특정 열 선택
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
xyplot(gear ~ carb, data = mtcars, main = "Scatterplot of Gear vs Carb",
       xlab = "Carburetors", ylab = "Gears")

# 산점도와 선형 회귀선 추가
plot(mtcars$carb, mtcars$gear, main = "Gear vs Carb with Regression Line",
     xlab = "Carburetors", ylab = "Gears", pch = 19, col = "blue")
abline(lm(gear ~ carb, data = mtcars), col = "red", lwd = 2)


# 상관행렬 시각화
corrplot(cor_matrix, method = "circle", type = "upper", 
         tl.cex = 0.8, title = "Correlation Matrix of mtcars")

# gear와 carb 간 산점도
ggplot(mtcars, aes(x = gear, y = carb)) +
  geom_point(color = "darkgreen", size = 3) +
  labs(title = "Scatterplot of Gear vs Carb", x = "Gears", y = "Carburetors") +
  theme_minimal()

# wt와 mpg 간 산점도 (carb로 색상 구분)
ggplot(mtcars, aes(x = wt, y = mpg, color = factor(carb))) +
  geom_point(size = 3) +
  scale_color_brewer(palette = "Set1") +
  labs(title = "Weight vs MPG (Colored by Carburetors)", x = "Weight", y = "MPG") +
  theme_minimal()

# wt와 mpg 간 상관계수 계산
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

**1. 데이터 구성**
- `work_experience`와 `income` 변수는 각각 근무 연수와 연봉 데이터를 저장합니다.
- `data.frame()` 함수를 사용하여 두 변수를 결합한 데이터프레임 `salary_data`를 생성합니다.
- 이 데이터프레임은 이후 분석의 기초 자료로 사용됩니다.

**2. 데이터 요약 통계**
- `summary()` 함수를 통해 데이터의 기초 통계량(최소값, 1사분위수, 중앙값, 평균, 3사분위수, 최대값)을 확인합니다.
- 이를 통해 데이터의 전체적인 분포와 특성을 파악할 수 있습니다.

**3. 산점도 시각화**
- `plot()` 함수를 사용하여 근무 연수(`work_experience`)와 연봉(`income`) 간의 관계를 시각적으로 확인합니다.
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

install.packages("car")  # car 패키지 설치
library(car)

# 분산 동질성 검정 (Levene Test)
leveneTest(weight ~ group, data = PlantGrowth)

# 방법 1: aov를 사용한 ANOVA 분석
anova_result1 <- aov(weight ~ group, data = PlantGrowth)
summary(anova_result1)

# 방법 2: lm과 anova를 조합한 분석
anova_result2 <- anova(lm(weight ~ group, data = PlantGrowth))
anova_result2

# 방법 3: oneway.test를 사용한 분석
oneway_result <- oneway.test(weight ~ group, data = PlantGrowth)
oneway_result
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
colnames(student_scores) = c("국어", "영어", "수학", "과학")
rownames(student_scores) = paste0("학생", 1:16)

# 데이터 확인: 데이터의 처음 몇 행을 출력
head(student_scores)

# 주성분 분석 수행: prcomp 함수를 사용하여 PCA 실행
pca_result = prcomp(student_scores, scale. = TRUE)

# 결과 요약: 주성분 분석 결과를 요약하여 출력
summary(pca_result)

# 어떤 변수가 PC1과 PC2를 구성하는 데 가장 큰 영향을 미치는지 구체적으로 파악
pca_result$rotation
```

1. **데이터 생성**: `kor`, `eng`, `math`, `sci`는 각각 국어, 영어, 수학, 과학 점수를 나타내는 벡터로, 16명의 학생 데이터를 포함합니다.
2. **데이터 결합**: `data.frame`을 사용해 네 벡터를 결합하고, `student_scores`라는 데이터 프레임으로 저장합니다.
3. **열 및 행 이름 지정**: 열 이름은 과목명(국어, 영어, 수학, 과학)으로, 행 이름은 "학생1", "학생2"와 같이 학생 번호로 지정합니다.
4. **데이터 확인**: `head(student_scores)`로 데이터의 처음 몇 행을 확인합니다.
5. **주성분 분석 수행**: `prcomp` 함수를 사용하여 PCA를 실행하며, `scale. = TRUE` 옵션을 추가하여 데이터를 표준화합니다.
6. **결과 요약**: `summary(pca_result)`로 주성분 분석 결과를 요약해 출력합니다(예: 각 주성분의 설명 분산 비율 등).


## 7. **Logistic Regression**

```R
# 데이터 생성 및 준비
drug_dose <- c(1, 1, 2, 2, 3, 3)  # 약물 용량
patient_response <- c(0, 1, 0, 1, 0, 1)  # 환자 반응 (0: 없음, 1: 있음)
frequency <- c(7, 3, 5, 5, 2, 8)  # 각 조건의 발생 빈도
experiment_data <- data.frame(drug_dose, patient_response, frequency)

# 로지스틱 회귀 모델 생성
logistic_model <- glm(patient_response ~ drug_dose, 
                      weights = frequency, 
                      family = binomial(link = "logit"), 
                      data = experiment_data)

# 모델 요약 정보 출력
print(summary(logistic_model))

# 예측 확률 시각화
plot(patient_response ~ drug_dose, data = experiment_data, 
     type = 'n', xlab = "Drug Dose", ylab = "Response Probability", 
     main = "Predicted Response Probability by Drug Dose")
curve(predict(logistic_model, newdata = data.frame(drug_dose = x), type = "response"), 
      add = TRUE, col = "blue", lwd = 2)
```

1. **데이터 준비**:  
   - `drug_dose`, `patient_response`, `frequency`라는 변수를 정의하고, 이를 `experiment_data`라는 데이터프레임으로 구성합니다.     

2. **모델 생성**:  
   - `glm()` 함수를 사용해 로지스틱 회귀 모델을 생성하며, `link = "logit"`을 명시적으로 추가하여 이항 분포와 로짓 링크 함수를 사용함을 강조했습니다.  

3. **결과 확인**:  
   - `summary(logistic_model)`로 모델의 계수, 유의성 등을 출력합니다.  

4. **시각화**:  
   - `plot()`에서 축 레이블(`xlab`, `ylab`)을 추가하여 그래프의 설명력을 높였습니다.  
   - `curve()`에서 `col = "blue"`와 `lwd = 2`를 사용해 예측 확률 곡선을 시각적으로 강조했습니다.


## 8. **Prediction Analytics**

```R
# 의사결정나무 분석을 위한 패키지 설치 및 로드
if (!require("rpart")) install.packages("rpart", dependencies = TRUE)
library(rpart)

# iris 데이터셋 확인 및 모델링 준비
data(iris) # 내장 데이터셋 로드
str(iris)  # 데이터 구조 확인
summary(iris) # 데이터 요약 정보

# Species를 종속 변수로 설정하고 나머지 변수를 독립 변수로 사용
tree_formula <- as.formula("Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
decision_tree <- rpart(tree_formula, data = iris, method = "class")

# 의사결정나무 시각화
png("decision_tree.png", width = 800, height = 600) # 그래프를 PNG 파일로 저장
plot(decision_tree, uniform = TRUE, main = "Decision Tree for Iris Dataset")
text(decision_tree, use.n = TRUE, cex = 0.8, col = "blue") # 노드 레이블 추가
dev.off() # 그래픽 장치 종료

# 모델 결과 출력
print(decision_tree)
```

1. **패키지 관리**: `require()`를 사용하여 `rpart` 패키지가 이미 설치되어 있는지 확인하고, 없으면 설치합니다.
2. **데이터 확인**: `str()`과 `summary()`를 통해 데이터 구조와 요약 정보를 확인하며, 데이터셋의 이해도를 높입니다.
3. **모델 정의**: `as.formula()`를 사용하여 명시적으로 공식을 정의하고, `rpart()` 함수로 의사결정나무 모델을 생성합니다.
4. **시각화**: `png()`를 사용하여 그래프를 파일로 저장하며, `uniform = TRUE` 옵션으로 나무의 깊이에 따른 노드 크기 차이를 줄입니다.
5. **결과 출력**: `print()`로 모델의 텍스트 결과를 콘솔에 출력합니다.


## 9. **Time Series**

```R
# 대한민국의 1인당 GDP와 전체 GDP 데이터를 분석하고, 2018년부터 2022년까지의 국민총생산을 예측하는 과정

# 필요한 패키지 설치 및 로드
if (!require("WDI")) install.packages("WDI")
if (!require("forecast")) install.packages("forecast")
library(WDI)
library(forecast)

# 세계은행 데이터에서 대한민국의 1960년부터 2017년까지의 GDP 데이터 수집
gdp_data <- WDI(
  country = "KR",
  indicator = c("NY.GDP.PCAP.CD", "NY.GDP.MKTP.CD"),
  start = 1960,
  end = 2017
)

# 데이터 열 이름 변경 및 확인
colnames(gdp_data) <- c("Country", "ISO2", "ISO3", "Year", "PerCapitaGDP", "TotalGDP")
head(gdp_data)

# 대한민국의 1인당 GDP 데이터 추출 및 시계열 변환
korea_gdp <- gdp_data$PerCapitaGDP
korea_ts <- ts(korea_gdp, start = min(gdp_data$Year), end = max(gdp_data$Year))

# ARIMA 모델 적합 및 예측
arima_model <- auto.arima(korea_ts)
future_forecast <- forecast(arima_model, h = 5) # 향후 5년 예측

# 예측 결과 출력 및 시각화
print(future_forecast)
plot(future_forecast, main = "대한민국 1인당 GDP 예측 (2018~2022)")
```

1. **데이터 수집**: `WDI` 패키지를 사용하여 세계은행 데이터에서 대한민국의 1960년부터 2017년까지의 1인당 GDP와 전체 GDP 데이터를 가져옵니다.
2. **데이터 정리**: 데이터 열 이름을 명확하게 변경하고, 대한민국의 1인당 GDP 데이터만 추출하여 시계열 객체로 변환합니다.
3. **예측 모델 생성**: `auto.arima` 함수를 사용하여 ARIMA 모델을 자동으로 적합시킵니다.
4. **예측 수행**: 적합된 모델을 기반으로 향후 5년(2018~2022년)의 1인당 GDP를 예측합니다.
5. **시각화**: 예측 결과를 그래프로 표시하여 직관적으로 확인할 수 있도록 합니다.


## 10. **Clustering**

```R
# Load the iris dataset and prepare it for clustering
data(iris)
df <- iris
df$Species <- NULL  # Remove the species labels for unsupervised clustering

# Perform K-means clustering with 3 clusters
set.seed(123)  # Ensure reproducibility of results
kmeans_result <- kmeans(df, centers = 3)

# Summarize and interpret the clustering results
print(kmeans_result)  # Display clustering details
summary(kmeans_result)  # Provide a summary of cluster statistics
cluster_comparison <- table(iris$Species, kmeans_result$cluster)  # Compare true species with clusters
print(cluster_comparison)

# Visualize the clusters using Sepal.Length and Sepal.Width
plot(df[c("Sepal.Length", "Sepal.Width")], 
     col = kmeans_result$cluster, 
     pch = 19, 
     main = "K-means Clustering (Sepal Dimensions)",
     xlab = "Sepal Length", 
     ylab = "Sepal Width")
```

1. **데이터 준비**: `iris` 데이터셋을 불러와 `df` 변수에 저장하고, 군집화를 위해 `Species` 열을 제거합니다.
2. **군집화 수행**: `kmeans` 함수로 데이터를 3개의 군집으로 나누며, `set.seed(123)`을 사용해 결과의 재현성을 보장합니다.
3. **결과 확인**: 군집화 결과를 출력하고, 요약 정보를 제공하며, 실제 종과 군집 간의 대응 관계를 표로 비교합니다.
4. **시각화**: `Sepal.Length`와 `Sepal.Width`를 기준으로 산점도를 그려 군집별로 색상을 다르게 표현합니다. 이를 통해 군집의 분포를 직관적으로 확인할 수 있습니다.



## 11. **Derived Variable**

```R
# 패키지 설치 및 로드
install.packages("nycflights13")
install.packages("dplyr")
library(nycflights13)
library(dplyr)

# 데이터 구조 및 열 이름 확인
data(flights) # nycflights13 패키지의 flights 데이터셋 사용
str(flights)
colnames(flights)

# 데이터를 tibble 형식으로 변환
flights_df <- as_tibble(flights)

# 도착 지연과 출발 지연의 차이를 계산하여 새로운 열 추가
flights_with_gain <- flights_df %>%
  mutate(gain = arr_delay - dep_delay)

# 결과 확인
flights_with_gain
```

1. **패키지 설치 및 불러오기**:  
   - `nycflights13` 패키지(뉴욕 항공편 데이터 포함)와 `dplyr` 패키지를 설치하고 로드합니다.
2. **데이터 확인**:  
   - `flights` 데이터셋을 사용하며, `str()`과 `colnames()`로 데이터 구조와 열 이름을 확인합니다.
3. **데이터 변환**:  
   - `flights` 데이터를 `tibble` 형식으로 변환하여 현대적인 데이터프레임으로 처리합니다.
4. **새로운 열 추가**:  
   - `mutate()` 함수를 사용해 `arr_delay`(도착 지연 시간)에서 `dep_delay`(출발 지연 시간)를 뺀 값을 `gain`이라는 새 열로 추가합니다.  
   - `%>%` 파이프 연산자를 활용하여 코드를 더 간결하게 작성합니다.

## 12. **Ensemble Analytics**

```R
# Ensemble Analytics - RandomForest를 활용한 분류 모델 예제

# 1. 필요한 패키지 설치 및 로드
install.packages("randomForest")
library(randomForest)

# 2. 데이터셋 확인 및 준비
data(iris) # iris 데이터셋 로드
set.seed(123) # 재현성을 위한 시드 설정
sample_idx <- sample(c(TRUE, FALSE), size = nrow(iris), replace = TRUE, prob = c(0.7, 0.3))
train_data <- iris[sample_idx, ] # 훈련 데이터 (70%)
test_data <- iris[!sample_idx, ] # 테스트 데이터 (30%)

# 3. RandomForest 모델 학습
rf_model <- randomForest(Species ~ ., data = train_data, ntree = 100, importance = TRUE)

# 4. 모델 평가 및 결과 확인
print(rf_model) # 모델 요약 정보 출력
importance(rf_model) # 변수 중요도 확인
confusion_matrix <- table(train_data$Species, predict(rf_model, train_data)) # 혼동 행렬 생성
print(confusion_matrix) # 혼동 행렬 출력
```

1. **패키지 설치 및 로드**: `randomForest` 패키지를 설치하고 불러옵니다.
2. **데이터 준비**: `iris` 데이터셋을 사용하며, `set.seed`를 통해 무작위 샘플링의 재현성을 보장합니다. 데이터를 70% 훈련, 30% 테스트로 나눕니다.
3. **모델 학습**: `randomForest` 함수를 사용해 분류 모델을 생성합니다. `ntree=100`으로 트리 개수를 설정하고, `importance=TRUE`로 변수 중요도를 계산합니다.
4. **결과 확인**: 모델 요약 정보와 변수 중요도를 출력하며, 훈련 데이터에 대한 예측 결과를 혼동 행렬로 확인합니다.


## 13. **Prediction Error**

```R
# 데이터 확인
data(mtcars)
head(mtcars, 3)
str(mtcars)
?mtcars

# 선형 회귀 모델 생성: 연비(mpg)와 차중(wt) 간의 관계 분석
model <- lm(mpg ~ wt, data = mtcars)
summary(model)

# 회귀 계수 및 기본 결과 확인
coef(model)
# 학습된 모델(model)이 훈련 데이터에 대해 예측한 값(fitted values) 중 처음 5개 관측치의 결과를 추출
fitted_values <- fitted(model)[1:5]  # 처음 5개 관측값의 예측값
residuals_values <- residuals(model)[1:5]  # 처음 5개 관측값의 잔차
confint(model)  # 회귀 계수의 신뢰구간
deviance(model)  # 모델의 편차

# 새로운 데이터에 대한 예측
new_data <- data.frame(wt = 3.5)
predicted_value <- predict(model, newdata = new_data)
manual_prediction <- coef(model)[1] + coef(model)[2] * 3.5
confidence_interval <- predict(model, newdata = new_data, interval = "confidence")
prediction_interval <- predict(model, newdata = new_data, interval = "prediction")

# 시각화 및 진단
par(mfrow = c(2, 2)) #없어도 2 by 2 그림이 그려짐(default 임)
plot(model)

# 잔차 정규성 검정
shapiro_test <- shapiro.test(residuals(model))

# 잔차 독립성 검정 (Durbin-Watson 테스트)
if (!require(lmtest)) install.packages("lmtest", dependencies = TRUE)
library(lmtest)
dw_test <- dwtest(model)

# 잔차 독립성 위반 ->  mpg에 영향을 끼치는 누락된 변수 추가(hp 혹은 cyl)
```

1. **데이터 확인**:  
   - `head(mtcars, 3)`과 `str(mtcars)`로 데이터 구조를 파악하고, `?mtcars`로 도움말을 확인.

2. **선형 회귀 모델 생성**:  
   - `lm(mpg ~ wt, data = mtcars)`로 차량 무게(`wt`)와 연비(`mpg`) 간의 선형 관계를 모델링.

3. **모델 결과 분석**:  
   - `coef(model)`로 회귀 계수 확인.  
   - `fitted(model)`과 `residuals(model)`로 예측값과 잔차 계산.  
   - `confint(model)`으로 회귀 계수의 신뢰구간 추정.  

4. **예측**:  
   - `predict()` 함수를 사용해 새로운 데이터(`wt = 3.5`)에 대한 예측 수행.  
   - 수동으로 예측값 계산 및 신뢰구간/예측구간 포함 예측.

5. **시각화 및 진단**:  
   - `plot(model)`로 잔차 분석을 위한 4개의 진단 플롯 생성.  

6. **잔차 진단**:  
   - `shapiro.test()`로 잔차의 정규성 검정.  
   - `dwtest()`로 잔차의 독립성 검정(`lmtest` 패키지 필요).  

7. **예측오류와 잔차**:
    - **잔차**: 학습 데이터 내에서의 오차로, 모델의 적합성을 평가.
    - **예측오류**: 새로운 데이터에서의 오차로, 모델의 일반화 성능을 평가.
    - **관계**: 잔차를 최소화하는 것이 예측오류를 줄이는 데 도움이 되지만, 과적합을 방지해야 함.




## 14. **K-Flod**

```R
# 패키지 설치 및 로드
if (!require("party")) install.packages("party")
if (!require("caret")) install.packages("caret")
library(party)
library(caret)

# 데이터 확인
data(iris)
head(iris)
str(iris)

# 3-fold 교차 검증 설정
set.seed(123) # 재현성을 위한 시드 설정
folds <- createFolds(iris$Species, k = 3, list = TRUE, returnTrain = TRUE)

# 정확도 저장용 벡터 초기화
accuracy_list <- numeric(length(folds))

# 교차 검증 루프
for (i in seq_along(folds)) {
  train_index <- folds[[i]] # 학습 데이터 인덱스
  test_index <- setdiff(1:nrow(iris), train_index) # 테스트 데이터 인덱스
  
  # 데이터 분할
  train_data <- iris[train_index, ]
  test_data <- iris[test_index, ]
  
  # 모델 생성 및 예측
  model <- ctree(Species ~ ., data = train_data)
  predictions <- predict(model, newdata = test_data)
  
  # 혼동 행렬 및 정확도 계산
  confusion_matrix <- table(Predicted = predictions, Actual = test_data$Species)
  print(confusion_matrix)
  accuracy_list[i] <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
}

# 결과 출력
cat("테스트 데이터 크기:", nrow(test_data), "\n")
cat("학습 데이터 크기:", nrow(train_data), "\n")
cat("각 폴드별 정확도:", accuracy_list, "\n")
cat("평균 정확도:", mean(accuracy_list), "\n")
```

1. **패키지 사용**: `caret` 패키지의 `createFolds` 함수를 사용하여 데이터를 교차 검증용으로 나눕니다.
2. **데이터 분할**: `createFolds`는 학습 데이터 인덱스를 반환하므로, `setdiff`를 사용해 테스트 데이터 인덱스를 추출합니다.
3. **모델링 및 평가**: 각 폴드에서 `ctree` 모델을 학습하고, 혼동 행렬을 통해 정확도를 계산합니다. `diag` 함수를 사용해 혼동 행렬의 대각선 요소(정답 수)를 추출합니다.
4. **결과 출력**: 각 폴드별 정확도와 전체 평균 정확도를 출력합니다. `cat` 함수를 사용해 결과를 명확히 표시합니다.



## 15. **Confusion Matrix**

```R
# Load necessary libraries and dataset
library(rpart)
library(caret)
data(iris)

# Inspect the dataset
print(head(iris, 10))
print(summary(iris))

# Split the data into training (70%) and testing (30%) sets
set.seed(123) # For reproducibility
trainIndex <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]

# Train a decision tree model using rpart
decisionTreeModel <- rpart(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                           data = trainData, method = "class")

# Evaluate the model on the training data
trainPred <- predict(decisionTreeModel, trainData, type = "class")
confusionMatrix(trainPred, trainData$Species)

# Visualize the decision tree
install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(decisionTreeModel)

# Evaluate the model on the test data
testPred <- predict(decisionTreeModel, testData, type = "class")
confusionMatrix(testPred, testData$Species)
```

1. **데이터 확인**: `head`와 `summary`를 사용하여 데이터셋의 구조와 요약 통계를 출력합니다.
2. **패키지 및 데이터 준비**: `rpart`와 `caret` 패키지를 사용하며, `createDataPartition` 함수로 데이터를 훈련 및 테스트 세트로 분할합니다.
3. **모델 학습**: `rpart`를 사용해 의사결정나무 모델을 학습하며, 입력 변수로 꽃받침과 꽃잎의 길이/너비를 사용합니다.
4. **모델 평가**: 훈련 및 테스트 데이터에 대해 혼동 행렬(`confusionMatrix`)을 생성하여 모델 성능을 평가합니다.
5. **시각화**: `rpart.plot`을 사용해 의사결정나무를 시각화합니다. 



## 16. **ROC**

```R
# 1. 필요한 패키지 설치 및 로드
install.packages("caret")
install.packages("pROC")
install.packages("mlbench")

library(caret)
library(pROC)
library(mlbench) # Sonar 데이터

# 2. 데이터셋 로드 및 탐색
data(Sonar)
summary(Sonar)
prop.table(table(Sonar$Class)) # 클래스 비율 확인

# 3. 데이터 분할
set.seed(123)
trainIndex <- createDataPartition(Sonar$Class, p = .7, list = FALSE, times = 1)
sonarTrain <- Sonar[trainIndex, ]
sonarTest <- Sonar[-trainIndex, ]

# 4. 의사결정나무 모델 학습
library(rpart)
decision_tree <- rpart(Class ~ ., data = sonarTrain, method = "class")

# 5. 테스트 데이터에 대한 예측 (확률)
predictions <- predict(decision_tree, newdata = sonarTest, type = "prob")[, 2]

# 6. ROC 곡선 생성 및 시각화
roc_curve <- roc(sonarTest$Class, predictions)

plot(roc_curve, main = "ROC Curve for Sonar Data",
     col = "#1c61b6",  # 곡선 색상
     lwd = 2,           # 곡선 두께
     print.auc = TRUE)  # AUC 값 표시
```

1. **패키지 관리**: `if` 조건문을 사용하여 `rpart` 패키지가 설치되어 있지 않을 경우 자동으로 설치하고 로드합니다.
2. **데이터 탐색**: `summary()` 함수를 통해 데이터의 요약 정보를 출력하며, `prop.table()`로 클래스 비율을 확인합니다.
3. **모델 학습**: `rpart` 함수에서 `method = "class"`를 명시적으로 지정하여 분류 문제임을 강조합니다.
4. **예측**: `predict()` 함수의 `type = "prob"` 옵션을 사용하여 확률값을 추출합니다.
5. **성능 평가**: `ROCR` 패키지를 활용하여 정확도와 ROC 곡선을 시각화합니다.


## 17. **Model Evaluation**

```R
# 필요한 패키지 설치 및 로드
install.packages("ROCR")
library(ROCR)

# 타이타닉 데이터셋 불러오기 및 초기 확인
data(Titanic)
titanic_data <- as.data.frame(Titanic)

# 데이터 전처리: 각 행을 개별 승객으로 확장
expanded_data <- titanic_data[rep(1:nrow(titanic_data), titanic_data$Freq), ]
expanded_data <- expanded_data[, -which(names(expanded_data) == "Freq")] # Freq 열 제거
expanded_data$Survived <- ifelse(expanded_data$Survived == "Yes", 1, 0) # 생존 여부를 이진 값으로 변환

# 데이터 요약 및 확인
head(expanded_data)
summary(expanded_data)

# 학습 및 테스트 데이터 분할 (70% 학습, 30% 테스트)
set.seed(123) # 재현성을 위한 시드 설정
train_indices <- sample(1:nrow(expanded_data), size = floor(0.7 * nrow(expanded_data)))
train_data <- expanded_data[train_indices, ]
test_data <- expanded_data[-train_indices, ]

# 로지스틱 회귀 모델 학습
logistic_model <- glm(Survived ~ Class + Sex + Age, data = train_data, family = binomial)
summary(logistic_model)

# 테스트 데이터로 예측 수행
predicted_prob <- predict(logistic_model, newdata = test_data, type = "response")

# ROC 곡선 및 AUC 계산
roc_pred <- prediction(predicted_prob, test_data$Survived)
roc_perf <- performance(roc_pred, measure = "tpr", x.measure = "fpr")
auc_value <- performance(roc_pred, measure = "auc")@y.values[[1]]

# ROC 곡선 시각화
plot(roc_perf, col = "blue", main = "ROC Curve")
abline(a = 0, b = 1, lty = "dotted")
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "blue", lty = 1)

# 결과 저장
results <- cbind(test_data, Predicted_Probability = predicted_prob)
write.csv(results, file = "titanic_logistic_results.csv", row.names = FALSE)
```

1. **데이터 전처리**:  
   - `Titanic` 데이터셋을 개별 승객 단위로 확장하여 `Freq` 열을 제거하고, `Survived`를 이진 값(0 또는 1)으로 변환합니다.

2. **데이터 분할**:  
   - 전체 데이터를 학습용(`train_data`)과 테스트용(`test_data`)으로 나누며, 층화 추출 없이 랜덤 샘플링을 사용합니다.

3. **모델 학습**:  
   - `glm` 함수를 사용해 로지스틱 회귀 모델을 학습하며, 독립 변수로 `Class`, `Sex`, `Age`를 사용합니다.

4. **예측 및 평가**:  
   - 테스트 데이터에 대한 생존 확률을 예측하고, ROC 곡선을 그립니다.  
   - AUC 값을 계산하고, 이를 그래프에 표시합니다.  
   - 최종 결과는 CSV 파일로 저장됩니다.  
