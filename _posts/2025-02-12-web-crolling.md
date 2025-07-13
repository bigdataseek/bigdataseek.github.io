---
title: 3차시 1:웹 스크래핑
layout: single
classes: wide
categories:
  - web crawling
tags:
  - html
  - beautifulsoup
  - selenium
---


## **1. HTML 기본 개념 소개**
HTML은 웹 크롤링의 기초입니다. 간단한 예제를 통해 확인하자.

### **(1) HTML 기본 구조**
```html
<!DOCTYPE html>
<html>
<head>
    <title>예제 페이지</title>
</head>
<body>
    <h1>환영합니다!</h1>
    <p id="intro">안녕하세요. 이것은 예제입니다.</p>
    <ul>
        <li class="item">첫 번째 항목</li>
        <li class="item">두 번째 항목</li>
    </ul>
</body>
</html>
```

- `<html>`: 문서의 시작과 끝.
- `<head>`: 메타 정보(예: 제목, 스타일 등).
- `<body>`: 실제 콘텐츠가 포함된 부분.
- `<h1>`, `<p>`, `<ul>`, `<li>`: 각각 제목, 단락, 목록, 항목을 나타내는 태그.
- `id`와 `class`: 특정 요소를 식별하거나 그룹화하는 속성.

### **(2) CSS 선택자**

*   클래스 선택자 (Class Selector)
    *  **표기법:** HTML에서는 `class="클래스명"`으로 지정하고, CSS나 선택자에서는 `.클래스명` 형식으로
    *   **그룹화:** 특정 공통 속성을 가진 요소들을 논리적으로 그룹화할 때 유용합니다. 예를 들어, 여러 종류의 "뉴스 기사"가 있다면 `class="news-item"`을 부여하여 묶을 수 있다.
    *  **특징:**
        * **중복 가능:** 하나의 HTML 문서 내에서 동일한 클래스 이름을 가진 요소가 여러 개 있을 수 있다.
        * **다중 클래스:** 하나의 HTML 요소는 여러 개의 클래스를 가질 수 있습니다. (예: `<p class="text important">`) 이 경우 CSS에서 `.text`와 `.important` 스타일이 모두 적용됩니다.
  
* ID 선택자 (ID Selector)
    * **표기법:** HTML에서는 `id="아이디명"`으로 지정하고, CSS나 선택자에서는 `#아이디명` 형식으로 사용
    * **고유한 요소 식별:** HTML 문서 내에서 **오직 하나만 존재하는 고유한 요소**를 식별할 때 사용합니다. 예를 들어, 웹 페이지의 메인 메뉴 영역, 푸터 영역 등은 보통 ID를 부여하여 고유하게 식별합니다.
    * **특징:**
        * **고유성:** 하나의 HTML 문서 내에서 **동일한 ID를 가진 요소는 오직 하나만 존재해야 합니다.** (기술적으로는 여러 개를 넣어도 웹 브라우저가 오류를 내지는 않지만, 이는 표준을 위반하며 예측 불가능한 동작을 초래할 수 있습니다.)
        * **높은 명시도:** CSS에서 ID 선택자는 클래스 선택자보다 더 높은 우선순위(명시도)를 가집니다.

## **2. BeautifulSoup 기본 사용법**
BeautifulSoup은 HTML을 파싱하고 원하는 데이터를 추출하기 위한 도구입니다. 

### **(1) 설치 및 임포트**
```bash
pip install bs4
```
```python
from bs4 import BeautifulSoup
```

### **(2) find()와 select() 비교**

웹 크롤링을 할 때 `find()`, `find_all()`, `select()`, `select_one()`은 Beautiful Soup 라이브러리에서 자주 사용되는 메서드들로, HTML 또는 XML 문서에서 원하는 데이터를 추출하는 데 사용됩니다. 각 메서드의 차이점을 간단히 설명하면 다음과 같습니다:

#### 1. **`find()`**
- **기능**: 주어진 조건에 맞는 첫 번째 요소만 반환합니다.
    - CSS선택자 사용불가
- **반환값**: 단일 객체 (조건에 맞는 요소가 없으면 `None` 반환).
- **사용 예**:
  ```python
  element = soup.find('div', class_='example') #class_ : 파이썬의 예약어를 피하려고
  ```
  - 위 코드는 클래스 이름이 `example`인 첫 번째 `<div>` 태그를 찾습니다.

#### 2. **`find_all()`**
- **기능**: 주어진 조건에 맞는 모든 요소를 리스트 형태로 반환합니다.
- **반환값**: 리스트 (조건에 맞는 요소가 없으면 빈 리스트 `[]` 반환).
- **사용 예**:
  ```python
  elements = soup.find_all('a', href=True)
  ```
  - 위 코드는 `href` 속성을 가진 모든 `<a>` 태그를 리스트로 반환합니다.



#### 3. **`select()`**
- **기능**: CSS 선택자를 사용하여 조건에 맞는 모든 요소를 리스트 형태로 반환합니다.
- **반환값**: 리스트 (조건에 맞는 요소가 없으면 빈 리스트 `[]` 반환).
- **사용 예**:
  ```python
  elements = soup.select('div.example a')
  ```
  - 위 코드는 클래스 이름이 `example`인 `<div>` 내부의 모든 `<a>` 태그를 리스트로 반환합니다.

  ```
  elements = soup.select('div.example > a') #직계자식 선택자(>)
  ```
  - div 태그이면서 example 클래스를 가진 요소의 직계 자식인 `<a>` 태그만을 찾습니다. 즉, div.example 바로 아래에 있는 `<a>` 태그만 선택하고, 그 `<a>` 태그 안에 또 다른 `<a>` 태그가 있다면 그것은 선택하지 않습니다.


#### 4. **`select_one()`**
- **기능**: CSS 선택자를 사용하여 조건에 맞는 첫 번째 요소만 반환합니다.
- **반환값**: 단일 객체 (조건에 맞는 요소가 없으면 `None` 반환).
- **사용 예**:
  ```python
  element = soup.select_one('div.example a')
  ```
  - 위 코드는 클래스 이름이 `example`인 `<div>` 내부의 첫 번째 `<a>` 태그를 반환합니다.


### **(3) 간단한 예제**
위에서 작성한 HTML 파일을 파싱해 보는 예제를 제공하세요.

```python
from bs4 import BeautifulSoup

# HTML 문서 (앞서 작성한 예제)
html = """
<!DOCTYPE html>
<html>
<head>
    <title>예제 페이지</title>
</head>
<body>
    <h1>환영합니다!</h1>
    <p id="intro">안녕하세요. 이것은 예제입니다.</p>
    <ul>
        <li class="item">첫 번째 항목</li>
        <li class="item">두 번째 항목</li>
    </ul>
</body>
</html>
"""

# BeautifulSoup 객체 생성
soup = BeautifulSoup(html, "html.parser")

# 데이터 추출(find()는 태그 이름, 클래스, ID, 속성 등 키워드 인자 기반)
title = soup.title.text  # <title> 태그의 텍스트
heading = soup.h1.text   # <h1> 태그의 텍스트
intro = soup.find(id="intro").text  # id="intro"인 요소의 텍스트
items = [item.text for item in soup.find_all(class_="item")]  # class="item"인 모든 요소의 텍스트

# 결과 출력
print("제목:", title)
print("헤딩:", heading)
print("소개:", intro)
print("항목들:", items)
```

<br>

## **3. 네이버 날씨 크롤링**
네이버 날씨 페이지에서 현재 온도를 가져오기.

```python
import requests
from bs4 import BeautifulSoup

# 네이버 날씨 페이지 URL
url = "https://weather.naver.com/"

# HTTP GET 요청
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}
response = requests.get(url, headers=headers)

# HTML 파싱
soup = BeautifulSoup(response.text, "html.parser")

# 현재 온도 추출(select_one()은 CSS 선택자 기반)
current_temp = soup.select_one(".current").text.strip()

# 결과 출력
print("현재 온도:", current_temp)
```

<br>

## **4. HTML태그와 CSS선택자 비교**

### **(1) HTML 태그**
- **HTML 태그**는 웹 페이지의 구조를 정의하는 기본 요소입니다.
- 태그는 `<`와 `>`로 감싸져 있으며, 특정 콘텐츠를 나타내거나 의미를 부여합니다.

#### **예시**
```html
<h1>제목</h1>
<p>단락 텍스트입니다.</p>
<a href="https://www.example.com">링크</a>
```

- `<h1>`: 제목을 나타냅니다.
- `<p>`: 단락(문단)을 나타냅니다.
- `<a>`: 하이퍼링크를 나타냅니다.

#### **역할**
- 웹 페이지의 **구조**와 **콘텐츠**를 정의합니다.
- 각 태그는 고유한 의미를 가지며, 브라우저가 이를 해석하여 화면에 표시합니다.


### **(2) CSS 선택자**
- **CSS 선택자**는 HTML 문서 내에서 특정 요소를 선택하기 위한 규칙입니다.
- CSS 선택자는 스타일을 적용하거나 JavaScript/jQuery, BeautifulSoup 등을 통해 특정 요소를 찾을 때 사용됩니다.

#### **예시**
```css
/* 클래스 선택자 */
.highlight {
    color: red;
}

/* ID 선택자 */
#main-title {
    font-size: 24px;
}

/* 태그 선택자 */
h1 {
    text-align: center;
}
```

#### **종류**
1. **태그 선택자**: HTML 태그 이름을 직접 사용합니다.
   - 예: `h1`, `p`, `div`
2. **클래스 선택자**: 클래스 속성(`class`)을 사용합니다.
   - 예: `.highlight`, `.container`
3. **ID 선택자**: ID 속성(`id`)을 사용합니다.
   - 예: `#main-title`, `#footer`
4. **속성 선택자**: 특정 속성을 가진 요소를 선택합니다.
   - 예: `a[href]`, `input[type="text"]`
5. **계층 선택자**: 요소 간의 관계를 이용합니다.
   - 예: `div span`, `.container > p`


### **(3) 차이점 비교**

| **항목**         | **HTML 태그**                                   | **CSS 선택자**                              |
|------------------|-----------------------------------------------|--------------------------------------------|
| **정의**         | 웹 페이지의 구조와 콘텐츠를 정의               | HTML 문서 내 특정 요소를 선택하기 위한 규칙 |
| **사용 목적**     | 브라우저가 콘텐츠를 표시하기 위해 사용          | 스타일 적용 또는 특정 요소를 찾기 위해 사용 |
| **형식**         | `<태그명>`으로 작성                            | `.클래스`, `#아이디`, `태그명` 등으로 작성  |
| **예시**         | `<h1>`, `<p>`, `<a href="...">`               | `h1`, `.highlight`, `#main-title`          |
| **주요 도구**    | HTML 문서 작성                                | CSS, JavaScript, jQuery, BeautifulSoup     |

### **(4) 예제를 통한 비교**

#### **HTML 코드**
```html
<div id="header">
    <h1 class="title">환영합니다!</h1>
    <p class="description">이곳은 예제 페이지입니다.</p>
</div>
```

#### **HTML 태그**
- `<div>`: 컨테이너 역할을 하는 태그.
- `<h1>`: 제목을 나타내는 태그.
- `<p>`: 단락 텍스트를 나타내는 태그.

#### **CSS 선택자**
- `#header`: ID가 `header`인 `<div>`를 선택.
- `.title`: 클래스가 `title`인 `<h1>`을 선택.
- `.description`: 클래스가 `description`인 `<p>`를 선택.


### **(5) BeautifulSoup에서의 활용**
BeautifulSoup에서는 CSS 선택자를 사용하여 HTML 문서에서 원하는 요소를 쉽게 찾을 수 있습니다.

#### **예제**
```python
from bs4 import BeautifulSoup

html = """
<div id="header">
    <h1 class="title">환영합니다!</h1>
    <p class="description">이곳은 예제 페이지입니다.</p>
</div>
"""
soup = BeautifulSoup(html, "html.parser")

# HTML 태그로 접근
print(soup.h1.text)  # 출력: 환영합니다!

# CSS 선택자로 접근
print(soup.select_one("#header .title").text)  # 출력: 환영합니다!
print(soup.select_one(".description").text)    # 출력: 이곳은 예제 페이지입니다.
```
<br>

## **5. 주의사항: 크롤링 정책(`robots.txt`)을 준수**
- **`robots.txt`**는 웹사이트의 관리자가 크롤러(예: 검색 엔진 봇)에게 어떤 페이지를 크롤링할 수 있고, 어떤 페이지를 크롤링하지 말아야 하는지를 지정하는 **텍스트 파일**입니다. 

- 이 파일은 웹사이트의 루트 디렉토리에 위치하며, 크롤러가 웹사이트를 방문할 때 가장 먼저 확인하는 규칙 파일입니다. 
- 법적 강제력 없음
- 네이버: `https://www.naver.com/robots.txt`

```
User-agent: *
Disallow: /
Allow: /$
Allow: /.well-known/privacy-sandbox-attestations.json
```

#### 1. **`Disallow: /`**
- 이 규칙은 "모든 경로(`/`)에 대해 크롤링을 금지한다"는 의미입니다.
- 즉, 기본적으로 웹사이트의 모든 페이지와 리소스는 크롤러가 접근할 수 없습니다.

#### 2. **`Allow: /$`**
- 이 규칙은 "루트 URL(예: `https://example.com/`)만 크롤링을 허용한다"는 의미입니다.
- 여기서 `$`는 정규식 기호로, "문자열의 끝"을 나타냅니다.
- 따라서, `/$`는 "루트 경로로 끝나는 URL"만 허용한다는 뜻입니다.
  - 예: `https://example.com/` → 허용
  - 예: `https://example.com/page1`, `https://example.com/subdir/` → 차단

#### 3. **`Allow: /.well-known/privacy-sandbox-attestations.json`**
- 이 규칙은 특정 파일인 `.well-known/privacy-sandbox-attestations.json`에 대해서만 크롤링을 허용합니다.
- 이 파일은 Privacy Sandbox 관련 정보를 담고 있으며, 검색 엔진이나 외부 서비스에서 접근해야 할 필요가 있습니다.

#### 4. **결론**
`Disallow: /`는 기본적으로 모든 경로를 차단하는 규칙이고,  
`Allow: /$`와 `Allow: /.well-known/privacy-sandbox-attestations.json`은 특정 경로(루트 URL과 특정 파일)만 예외적으로 허용하는 규칙입니다.  

즉, **"전체를 막고 필요한 부분만 개방"** 하는 방식으로 작동하며, 이를 통해 웹사이트 운영자는 크롤러의 접근을 세밀하게 제어할 수 있습니다.