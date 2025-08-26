---
title: 7차시 6:R Programming Basic 1
layout: single
classes: wide
categories:
  - R programming
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

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
