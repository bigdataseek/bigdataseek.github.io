---
title: 파이썬 기초 개념
layout: single
classes: wide
categories:
  - 파이썬 강좌
tags:
  - variable
  - string
  - integer
  - bool
  - list
  - tuple
  - set
  - dict
  - operator
  - function
  - conditional statement
  - loop statement
  - class
  - exception
  - module
  - package
  - file i/o
---

## 1. 파이썬 소개 

1.1. 컴퓨터와 프로그래밍:데이터를 처리
- 컴퓨터의 어원
- 컴퓨터는 입력된 데이터를 처리하여 결과를 출력한다.

1.2. 프로그래밍 언어 종류
- [TIOBE](https://www.tiobe.com/tiobe-index/) 사이트
- 웹 개발/ 모바일 개발/ 데이터과학/ 게임개발/ 
- 시스템 프로그래밍/ 임베디드 시스템

1.3. 파이썬의 두각
- 간결하고 유연, 많은 커뮤니티 존재
- 인터넷 시대 > 스마트폰 시대 > AI 시대
- 데이터 과학, 인공지능, 웹개발 분야에서 활용

1.4.1. 파이썬 설치 및 IDE 설치
- 버전 확인: 
```python 
python --version
```

- VS Code Extension 설치
    - Python
    - Python Debugger
    - Korean Language Pack for VSCode
    - Prettier
    - Rainbow CSV
    - Git Graph
    - gitignore
    - Markdown PDF
    - Material Icon Theme

1.4.2. 가상환경
- 콘솔에서 가상환경 구성
- VSCode에서 가상환경 구성

1.4.3. Git (버전관리, 백업, 공유)
- GitHub에서 개발 Email로 가입(개발용 Gmail 구비)
- git 설치하고 확인(콘솔에서 git 명령어 입력)
- 해당 프로젝트 터미널에서 GitHub의 계정정보 입력
    - git config --local user.name 'alice'
    - git config --local user.email 'alice@abc.com'
- Working Direcotry > Stage > Repository

1.4.4. MarkDown
- GitHub에 README.md 추가 후 Pull 실행

1.5.1. 첫번째 프로그램 작성 및 실행
- 임의의 파일 생성 후 print('Hello World') 작성
```python 
print("Hellow World")
```
- 콘솔에서 실행하거나 IDE에서 실행

1.5.2. 주석
- 일종의 메모

1.5.3. 구문오류
```python 
print('helo world'
         ^
SyntaxError: '(' was never closed
```


## 2. 변수

2.1. 프로그래밍 4가지 핵심요소
> 데이터를 논리적으로 처리

- 변수: 데이터
- 함수: 처리/계산
- 제어문: 논리적 절차
- 클래스: 변수(데이터) + 함수(처리)

2.2. 부가적으로 알아야 할 영역
- 연산자
- 파일입출력
- 예외처리
- 모듈과 패키지

2.3. 정보 사례
> 성별, 이름, 나이, 키

2.4.1. 변수 할당
- 데이터를 저장하기위해 이름을 붙이는 과정

2.4.2. 변수명 짓기
- 영문 대소문자, 숫자, 밑줄(_)만 사용
- 숫자로 시작 불가
- 예약어 사용 불가

2.4.3. 네이밍 유형
- snake_case, camelCase, PascalCase
- 변수는 snake_case, 클래스는 PascalCase 사용

2.4.4. 예약어
```python
help("keywords")
```

2.4.5. 수강생 정보 변수화

2.5. 변수의 자료형(문수불리 플스딕)
- 문자열(str)
- 숫자(int, float)
- 불리언(bool)
- 리스트(list)
- 튜플(tuple)
- 셋(set)
- 딕셔너리(dict)

2.6. Iterable, Sequence, Immutable
- Iterable: 반복적 순회 가능 객체
- Sequence: 순서대로 나열된 객체(문자열, 리스트, 튜플)
- Immutable: 변경불가 객체(문자열, 숫자, 튜플)


## 3. 연산자

3.1. 문자열, 숫자, 불리언 변수 할당 및 출력
- 문자열 변수 선언 및 출력
- 숫자 변수 선언 및 출력
- 불리언 변수 선언 및 출력

3.2.1. 산술연산자
- 사칙연산, 몫, 나머지, 제곱 연산자

3.2.2. 비교연산자
- ==, !=, >,<, >=, <=

3.3.1. 논리연산자
- and, or, not

3.3.2. 대입연산자, 복합 대입연산
- =, +=, *=

3.4.1. 멤버쉽 연산자
- in, not in

3.4.2. 아이덴티티 연산자
- is, is not

3.5. 데이터 타입 변환(형변환)
- 숫자와 문자를 + 연산시 에러

3.6. 문자열 생성 및 연산
- 문자열 생성_ "", '', """ """, ''' '''
- 문자열 연결(+), 문자열 반복(*)

3.7. print() 함수와 문자열 포맷팅
```python
    name = "Alice"
    age = 30
    print("이름:", name, "나이:", age)

    print(f"이름: {name}, 나이: {age}")
    print("이름: {}, 나이: {}".format(name, age))
    print("이름: %s, 나이: %d" % (name, age))
```

3.8. input()
- input()에 의해 입력받은 값은 문자열


## 4. 조건문과 반복문 
4.1.1. if 조건문
```python
x = 10
y = 5

if x > y:
    print("x는 y보다 큽니다.")
```

4.1.2. if ~ else 조건문

4.1.3. if ~ elif ~ else 조건문

4.1.4. 조건판단의 논리 연산자

4.1.5. 삼항 연산자

4.1.6. 중첩 조건문

4.2.1. for ~ in 반복문
```python
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)
```

4.2.2. range() 함수

4.3.1. while 반복문
```python
numbers = []
count = 0
while count < 5:
    numbers.append(count*2)
    count += 1
print(numbers)
```
4.3.2. break
- 현재 실행 중인 반복문을 종료

4.3.3. continue
- 현재 반복의 나머지 부분을 건너뛰고 다음 반복으로 넘어감.

4.4. 중첩 반복문
- 구구단 출력


## 5. 리스트와 튜플
5.1.1. 리스트 정의
- 여러 데이터를 순서대로 저장
```python
fruits = ["apple", "banana", "cherry"]
print(fruits[1])  # 리스트의 두 번째 요소 출력 (banana)
```

5.1.2. 리스트 접근(인덱싱)
- []안에 인덱스를 지정하여 요소에 접근(0,-1)

5.2.1. 리스트 요소 추가_ append(), insert(), extend()
- append(): 마지막에 요소 추가
- insert(): 특정 인덱스에 요소 추가
- extend(): 마지막에 여러 요소 추가

5.2.2. 리스트 요소 삭제_ remove(), pop(), clear()
- remove(): 특정 요소 삭제
- pop(): 특정 인덱스 요소 삭제, 디폴트는 last
- clear(): 모든 요소 삭제

5.2.3. 리스트 요소 수정
- 특정 인덱스에 새로운 값 지정

5.3. 리스트 메서드_ sort(), reverse()_ enumerate(), sorted()
- sort() : 리스트 정렬
- reverse(): 리스트 순서 뒤집기
- enumerate() 내장함수: iterable 요소 순회,인덱스와 값을 튜플로 반환
- sorted() 내장함수: iterable 요소 정렬하고 리스트 반환
- len() 내장함수: 요소 개수 반환

5.4. 리스트 슬라이싱
- 대괄호([])안에 시작인덱스와 끝인덱스를 콜론으로 구분.
- 세번째 인자로 step지정 가능
    print(my_list[::-1])

5.5. 리스트 컴프리헨션
- for문을 이용한 리스트 생성을 간략화시킴
```python
numbers = [1, 2, 3, 4, 5]
doubled_numbers = [x * 2 for x in numbers]
print(doubled_numbers)  # 출력: [2, 4, 6, 8, 10]
```
5.6. 튜플의 정의
- 리스트와 동일.()사용, 쉽표(,) 사용, 요소 수정 불가
```python
coordinates = (3, 7)
print(coordinates[0])  # 튜플의 첫 번째 요소 출력 (3)
```

5.7. 튜플의 언패킹
- 언패킹시 튜플개수와 변수 개수 일치해야
- 펼침연산자(*) 사용 가능


## 6. 딕셔너리와 셋
6.1.1. 딕셔너리 정의
- 키-값 구조
- 키로 값을 조회
```python 
students = {'st01':'Alice', 'st02':'Bob'}
print(students['st01'])
```

6.1.2. 존재하지 않는 키 접근 위험_ get(), setdefault()

6.2.1. 딕셔너리 요소 추가
- 새로운 키값을 이용해 추가

6.2.2. 딕셔너리 요소 삭제_ pop(), popitem(), clear()
- pop() : 키를 명시해야
- popitem() : 키를 명시하지 않음

6.3. 딕셔너리 메서드_ keys(), values(), items()
- keys() : 모든 키 반환, 사용하려면 list로 형변환
- values() : 모든 값 반환, 사용하렴ㄴ list로 형변환
- items() : 딕셔너리의 모든 키-값 쌍을 튜플 형태로

6.4. 딕셔너리 컴프리헨션
- 간결하게 딕셔너리 생성
```python 
squared_dict = {num: num**2 for num in numbers}
```

6.5. 셋의 정의
- 중복 제거시 사용

## 7. 함수

7.1. 함수 정의
- 파이썬에서 특정작업을 수행하는 코드 블록
- def 키워드. 함수 이름, 매개변수, 콜론(:)으로 구성
- 들여쓴 코드 블록이 함수의 내용을 정의
```python
def say_hello(name):
    print("Hello", name)
```

7.2. 함수 호출
- 함수 이름과 괄호()를 사용, 필요시 인자 전달
```python
say_hello("Alice")
```

7.3. 반환값
- return 문은 함수의 실행을 종료, 지정된 값을 호출자에게 반환
```python
def add(a, b):
    return a + b

result = add(3, 5)
```

7.4.1. 일반매개변수
```python
def greet(name):
    print("안녕하세요,", name + "님!")

greet("Bob") #출력: 안녕하세요, Bob님!
```

7.4.2. 기본값 매개변수
```python
def greet(name, greeting="안녕하세요"):
    print(greeting + ",", name + "님!")

 greet("Bob")  # 출력: 안녕하세요, Bob님!
 greet("Charlie", "Hello")  # 출력: Hello, Charlie님!
```

7.5.1. 가변 매개변수
```python
def sum_numbers(*args):
    total = 0
    for number in args:
        total += number
    return total

result = sum_numbers(1, 2, 3, 4, 5)
print(result)  # 출력: 15

temp_list = [10, 20, 30]
result = sum_numbers(*temp_list)
print(result)  # 출력: 60
```

7.5.2. 키워드 가변 매개변수
```python
def print_info(**kwargs):
    for k, v in kwargs.items():
        print(k + ":", v)

print_info(name="Alice", age=30, city="Seoul")

temp_dict = {"name": "Bob", "age": 25, "city": "Busan"}
print_info(**temp_dict)
```

7.6.1. 지역변수와 전역변수
- 함수 내 변수는 지역변수, 함수 외부에 있는 변수는 전역변수
```python
global_var = 10  # 전역변수(함수 외부에 존재)

def my_function():
    local_var = 5  # 지역변수(함수 내 존재)
    print(global_var)  # 전역변수 접근 가능
    print(local_var)  # 지역변수 접근 가능

my_function()
print(global_var)  # 전역변수 접근 가능
print(local_var) # 지역변수 접근시 오류
```

7.6.2. global, nonlocal 키워드
- global : 함수 내부에서 전역변수에 접근하고 수정
- global 사용하지 않고 전역변수와 동일한 지역 변수명 사용시 같은 이름의 지역변수가 함수내에서 생성

- nonlocal : global과 동일하지만 중첩 함수에서 사용.
- nonlocal은 바깥쪽 함수의 지역 변수에 접근하고 수정하기 위해

7.7.1. 내장함수와 사용자 정의 함수
- print(), input(), len(), int(), str(), list(), dict(), range()
- 사용자 정의 함수는 개발자가 정의

7.7.2. 람다함수
- 함수에 이름을 부여하지 않고 함수의 기능만을 정의하는 방식
- lambda 매개변수1, 매개변수2, ...: 표현식
```python
# 숫자 두 개를 더하는 람다 함수
add = lambda x, y: x + y
result = add(3, 5)
print(result)  # 출력: 8
```

7.7.3. map(), zip(), filter() 내장함수
- map(function, iterable):
    - 주어진 함수(function)를 반복 가능한 객체(iterable)의 각 요소에 적용하고, 그 결과를 새로운 반복 가능한 객체로 반환
    ```python
    numbers = [1, 2, 3, 4, 5]

    # map 함수 예시
    squared = list(map(lambda x: x**2, numbers))  # 각 요소를 제곱
    print(squared)  # [1, 4, 9, 16, 25]
    ```
- zip(iterables):
    - 여러 개의 반복 가능한 객체를 동시에 순회하면서, 각 객체의 같은 인덱스에 있는 요소들을 튜플로 묶어 새로운 반복 가능한 객체를 반환
    ```python
    # zip 함수 예시
    names = ['Alice', 'Bob', 'Charlie']
    ages = [30, 25, 35]
    for name, age in zip(names, ages):
        print(name, age)
    ```
- filter(function, iterable):
    - 주어진 함수(function)가 True를 반환하는 iterable의 요소들만 추출하여 새로운 반복 가능한 객체로 반환
    ```python
    # filter 함수 예시
    even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
    print(even_numbers)  # [2, 4]
    ```

## 8. 파일 입출력 

8.1. 파일 열기, 읽기/쓰기, 파일 닫기
- open()함수 사용, 파일 경로, 모드, close()
```python
def file_read():
    file = open("example.txt", "r")
    content = file.read()
    print(content)
    file.close()
```

8.2. with문 사용
- with 키워드는 컨텍스트 관리를 위한 기능을 제공.
```python
def using_with_keyword():
    with open("example.txt", "r") as file:
        content = file.read()
    print(content)
```

8.3. 파일 열기 모드
- "r"_ 읽기 모드 ( 파일이 없으면 오류 발생)
- "w"_ 쓰기 모드 ( 파일 없으면 생성, 파일이 이미 존재하면 덮어씀.)
- "a"_ 추가 모드 ( 파일 없으면 생성, 파일이 이미 존재하면 파일 끝에 내용 추가)
- "x"_ 새로운 파일 쓰기 모드 (파일이 없으면 생성, 파일이 이미 존재하면 오류 발생)
- 이전 모드들과 결함하는 "+"_ 읽기와 쓰기를 모두 허용하는 모드('r+','w+', 'a+' 가능)

8.4. 텍스트 모드 / 바이너리 모드
- 파일을 열때 기본값은 텍스트모드(t, 예를 들어 ‘rt’)
- 바이너리모드(b)에서 파일은 이진 데이터로 해석. 파일의 내용을 바이트(byte) 객체로 처리
- 텍스트가 아닌 데이터를 다룰 때 사용(이미지, 음악 파일, 'rb')
- pip install Pillow
```python
import io
from PIL import Image

def using_binary_mode():
    - 바이너리 모드로 이미지 파일 열기
    with open("컴퓨터구조.png", "rb") as f:
        image_data = f.read()

    # 이미지 데이터를 PIL Image 객체로 변환
    image = Image.open(io.BytesIO(image_data))

    # 이미지 처리 (예: 크기 조정)
    resized_image = image.resize((512, 256))

    # 처리된 이미지 저장
    resized_image.save("resized_image.png")
```

8.5. 파일 읽기 메서드_ read(), readline(), readlines()
- read() : 전체 내용을 문자열로 반환
- readline() : 한 줄씩 문자열로 반환
- readlines() : 모든 각 줄을 리스트로 반환
```python
def only_read_file():
    with open("example.txt", "r") as file:
        content = file.read()
    print(content)

def only_readline_file():
    with open("example.txt", "r") as file:
        line1 = file.readline()
        line2 = file.readline()
    print(line1, line2)

def only_readlines_file():
    with open("example.txt", "r") as file:
        lines = file.readlines()
    print(lines)
```

8.6. 파일 쓰기 메서드_ write(), writelines()
- write() : 문자열을 파일에 작성
- 한글 포함시 일반적으로 encoding='utf-8' 사용
```python
def only_write_file():
    with open("output.txt", "w", encoding="utf8") as file:
        file.write("안녕하세요!\n")
        file.write("This is a sample file.\n")
```

- writelines()_ 리스트를 파일에 작성
```python
def only_writelines_file():
    with open("output.txt", "w") as file:
        lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
        file.writelines(lines)
```

8.7. json파일 읽고 쓰기

- json파일을 load()하여 dict/list 객체 반환
```python
import json
def read_json_file():
    with open("sample.json", "r") as file:
        data = json.load(file)
    print(data)
```

- json파일을 read()시 문자열 반환
- json문자열을 loads()하여 dict/list 객체 반환
```python
def read_json_file_as_dict():
    with open("sample.json", "r") as file:
        data_str = file.read()
        data_dict = json.loads(data_str)
        print(data_dict)
```

- dumps( )사용시, dict객체를 json 포맷의 문자열로
```python
def write_json_file_as_string():
    data = {"name": "John", "age": 30, "city": "New York"}
    data_str = json.dumps(data)
    print(data_str)
```

- dump() 사용시, dict객체를 json 파일로
```python
def write_json_file_as_dict():
    data = {"name": "John", "age": 30, "city": "New York"}
    with open("data.json", "w") as file:
        json.dump(data, file)
```

## 9. 예외처리

9.1. 예외처리 정의
- 예외는 프로그램 실행 중에 발생하는 오류(ZeroDivisionError, TypeError 등)

9.2.1. try ~ except 문
```python
def sample_try_except():
    try:
        result = 10 / 0
    except ZeroDivisionError:
        print("Error: Division by zero is not allowed.")
```

9.2.2. try ~ except ~ else ~ finally 구조
```python
def sample_try_except_else_finally():
    try:  - 예외 발생 예상 표현식
        result = 10 / 2
    except ZeroDivisionError:  - 예외 발생시
        print("Error: Division by zero is not allowed.")
    else:  - 예외가 발생하지 않으면 결과 출력
        print("The result is:", result)
    finally:  - 예외 발생 여부와 관계없이 실행
        print("This block will always execute.")
```

9.2.3. raise 키워드
- 의도적으로 예외 발생시킴
```python
def divide(x, y):
    if y == 0:
        raise ZeroDivisionError(
            "0으로 나눌 수 없습니다."
        )  - ZeroDivisionError 예외 의도적 유발
    return x / y

def sample_raise():
    try:
        result = divide(10, 0)
    except ZeroDivisionError as e:
        print("Error:", e)
```

9.3. 모듈의 정의
- 모듈은 함수, 변수, 클래스 등을 정의한 .py 파일
- math 모듈 전체를 가져오기: import math
- math 모듈에서 sqrt 함수만 가져오기: from math import sqrt
- math 모듈에서 모든 함수와 변수 가져오기(비추천): from math import *

9.4. 표준 모듈
- 예:math,random
```python
import random

def sample_random_module():
    print(random.random())  # 0 이상 1미만
    print(random.randint(1, 10))  # 10포함
    print(random.choice(["apple", "banana", "cherry"]))
    print(random.sample(range(1, 46), 2))  # 무작위로 2개 샘플링
```

9.5. 사용자 정의 모듈과 패키지
- 패키지는 여러 모듈을 모아 놓은 폴더
- 패키지 폴더에는 __init__.py 파일이 있어야
- my_package 라는 폴더를 만들고 그 안에 module1.py,module2.py,__init__.py파일을 설정한다.


## 10. 클래스

10.1. 클래스와 객체의 정의
- 클래스는 객체를 생성하기 위한 틀 또는 청사진
- 객체는 클래스에 정의된 속성과 메서드를 가지는 실체입니다.
- 객체는 클래스의 인스턴스라고도 합니다.
- 객체지향프로그래밍: 현실 세계의 사물이나 개념을 객체라는 단위로 단순화.
- 클래스는 class 키워드를 사용
```python
class Dog:  # class 키워드 사용
    def __init__(self, name, age):  # self는 첫번째 매개변수로 인스턴스(객체) 지칭
        self.name = name
        self.age = age

    def bark(self):  # self는 매개변수로 인스턴스(객체) 지칭
        print("멍멍!")

my_dog = Dog("Buddy", 3)
```

10.2. 생성자(__init__메서드), 속성, 메서드
```python
class Car:  # 자동차 클래스 정의
    version = 1.0 # 클래스 변수(모든 인스턴스가 공유)

    # 초기화메서드: 객체 생성 시 속성 초기화
    def __init__(self, color, model, year):  
        # 속성(혹은 인스턴스 변수)
        self.color = color 
        self.model = model
        self.year = year

    # 인스턴스 메서드: 자동차 시동 기능
    def start(self):  
        print("부릉부릉! 시동을 걸었습니다.")

    # 인스턴스 메서드: 자동차 정지 기능
    def stop(self):  
        print("끼익! 차를 멈췄습니다.")

#클래스명()으로 호출하여 인스턴스 생성
my_car = Car("빨강", "쏘나타", 2022) 
friend_car = Car("파랑", "모닝", 2020)

# 객체 속성 출력
print(my_car.color) # 객체의 속성 접근시 dot(.)사용
print(friend_car.model)
print(my_car.version) # 클래스 변수는 모든 인스턴스가 공유

# 객체 메서드 호출
my_car.start()   # 객체의 메서드 호출시 dot(.)사용
friend_car.stop()  # 출력: 끼익! 차를 멈췄습니다.
```

10.3. 생성자 없는 클래스
```python
class MyClass:
    def my_method(self):
        print("Hello!")

obj = MyClass()
obj.my_method()  # Hello! 출력
```

- 명시적으로 __init__ 메서드를 정의하지 않으면 파이썬은 기본 생성자를 제공
```python
def __init__(self):
    pass
```

10.4. 클래스 메서드
- @classmethod 데코레이터를 사용
- 클래스 메서드는 클래스 레벨에서 작동
- 첫 번째 매개변수로 cls 를 사용하여 클래스 자신을 참조
- 클래스를 통해 호출
```python
class SampleClassMethod:
    count = 0  # 클래스 변수

    def __init__(self):
        SampleClassMethod.count += 1

    @classmethod
    def get_count(cls):  #생성된 객체의 수 확인
        return cls.count
# 객체 생성
instance1 = SampleClassMethod()
instance2 = SampleClassMethod()

# 클래스 메서드 호출
print(SampleClassMethod.get_count())  # 2 출력
```

10.5. 스태틱 메서드
- 클래스에 속하지만 클래스나 인스턴스와는 무관하게 동작(util 함수용)
- @staticmethod 데코레이터를 사용
```python
class SampleStaticMethod:
    @staticmethod
    def add(x, y):
        return x + y

#static method 호출
print(SampleStaticMethod.add(5, 3))  # 8 출력
```

10.6.1. 상속
- 기존 클래스(부모 클래스)의 속성과 메서드를 물려받아 새로운 클래스(자식 클래스)를 만드는 것
- Person을 상속한 Student 클래스
- 자식 클래스에 __init__ 메서드가 명시적으로 없는 경우 부모 클래스의 __init__이 자동 호출됨

```python
class Person:  # 부모 클래스 (사람)
    def __init__(self, name):
        self.name = name

    def speak(self):
        print(f"{self.name}는 말합니다.")


class Student(Person):  # 자식 클래스 (학생)

    def study(self):
        print(f"{self.name}는 열심히 공부합니다.")


student_alice = Student('Alice')
student_alice.speak()  # Alice는 말합니다
student_alice.study()  # Alice는 열심히 공부합니다
```

10.6.2. 메서드 오버라이딩
- 부모 클래스에서 상속받은 메서드를 자식 클래스에서 재정의
```python
class Animal:  # 부모 클래스 (동물)
    def __init__(self, name):
        self.name = name

    def speak(self):
        print("동물이 소리를 냅니다.")

# 자식 클래스 (강아지) - Animal 클래스 상속
class Dog(Animal):  
    def speak(self):
        print("멍멍!")

# 자식 클래스 (고양이) - Animal 클래스 상속
class Cat(Animal):  
    def speak(self):
        print("야옹!")

# 자식 클래스 (Bird) - Animal 클래스 상속
class Bird(Animal):  
    def __init__(self, name, wingspan):        
        super().__init__(name)
        self.wingspan = wingspan

    def wing_length(self):
        return self.wingspan

# 객체 생성
animal = Animal("동물")
dog = Dog("바둑이")
cat = Cat("나비")
bird = Bird("독수리", 150)

#speak() 메서드 호출
animal.speak()  # 출력: 동물이 소리를 냅니다.
dog.speak()    # 출력: 멍멍!
cat.speak()    # 출력: 야옹!
bird.speak()  # 동물이 소리를 냅니다
print(bird.wing_length())  #출력: 150
```

10.7. 데이터클래스
- 데이터를 저장하는 클래스를 정의시 코드를 간결하게 유지 및 관리
- __init__, __repr__,__eq__ 등의 특수 메서드가 자동 생성
```python
from dataclasses import dataclass

@dataclass
class Customer:
    name: str
    age: int
    city: str

person1 = Customer("Alice", 30, "New York")
person2 = Customer("Bob", 30, "New York")
print(person1)  
print(person1 == person2)  # 비교가능
```
