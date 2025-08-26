---
title: 1차시 16(빅데이터 분석):Plotly로 인터랙티브 시각화 + Dash/Streamlit로 웹 앱 만들기
layout: single
classes: wide
categories:
  - Plotly, Dash, Streamlit
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---



## 1. Plotly

Plotly는 Python에서 인터랙티브한 그래프를 쉽게 그릴 수 있는 강력한 라이브러리입니다. 

### 1.1 **산점도 (Scatter Plot)**
- **이론**: 산점도는 두 변수 간의 관계를 시각화하는 데 사용됩니다. 점들은 데이터 포인트를 나타내며, x축과 y축에 변수를 매핑합니다.
- **실습 예제**:
  ```python
  import plotly.express as px
  df = px.data.iris()  # Iris 데이터셋 로드
  fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
  fig.show()
  ```
![plotly_scatter](/assets/images/plotly/plotly_scatter.png)


### 1.2 **선 그래프 (Line Plot)**
- **이론**: 선 그래프는 시간에 따른 데이터의 변화를 보여주는 데 적합합니다. x축은 시간 또는 순서를, y축은 값을 나타냅니다.
- **실습 예제**:
  ```python
  import plotly.express as px
  df = px.data.stocks()  # 주식 데이터셋 로드
  fig = px.line(df, x="date", y="GOOG", title='Google 주가 변화')
  fig.show()
  ```
![plotly_scatter](/assets/images/plotly/plotly_line.png)

### 1.3 **막대 그래프 (Bar Chart)**
- **이론**: 막대 그래프는 범주형 데이터를 비교하는 데 사용됩니다. 각 막대는 범주별 값을 나타냅니다.
- **실습 예제**:
  ```python
  import plotly.express as px
  df = px.data.tips()  # 팁 데이터셋 로드
  fig = px.bar(df, x="day", y="total_bill", color="sex", title="요일별 총 지출액")
  fig.show()
  ```
![plotly_scatter](/assets/images/plotly/plotly_bar.png)



### 1.4 **히스토그램 (Histogram)**
- **이론**: 히스토그램은 데이터의 분포를 시각화합니다. x축은 데이터의 구간(bin), y축은 빈도를 나타냅니다.
- **실습 예제**:
  ```python
  import plotly.express as px
  df = px.data.tips()  # 팁 데이터셋 로드
  fig = px.histogram(df, x="total_bill", nbins=20, title="총 지출액 분포")
  fig.show()
  ```
![plotly_scatter](/assets/images/plotly/plotly_histogram.png)


### 1.5 **파이 차트 (Pie Chart)**
- **이론**: 파이 차트는 전체에 대한 각 부분의 비율을 시각화하는 데 사용됩니다.
- **실습 예제**:
  ```python
  import plotly.express as px
  df = px.data.tips()  # 팁 데이터셋 로드
  fig = px.pie(df, names="day", values="total_bill", title="요일별 지출 비율")
  fig.show()
  ```
![plotly_scatter](/assets/images/plotly/plotly_pie.png)


### 1.6 **박스 플롯 (Box Plot)**
- **이론**: 박스 플롯은 데이터의 분포와 이상치를 시각화합니다. 중앙값, 사분위수, 최소값, 최대값 등을 보여줍니다.
- **실습 예제**:
  ```python
  import plotly.express as px
  df = px.data.tips()  # 팁 데이터셋 로드
  fig = px.box(df, x="day", y="total_bill", color="sex", title="요일별 지출액 분포")
  fig.show()
  ```
![plotly_scatter](/assets/images/plotly/plotly_boxplot.png)


### 1.7 **히트맵 (Heatmap)**
- **이론**: 히트맵은 행렬 형태의 데이터를 색상으로 표현하여 패턴을 시각화합니다.
- **실습 예제**:
  ```python
  import plotly.express as px
  import numpy as np

  # 랜덤 데이터 생성
  data = np.random.rand(10, 10)
  fig = px.imshow(data, labels=dict(x="X Axis", y="Y Axis", color="Value"),
                  title="히트맵 예제")
  fig.show()
  ```
![plotly_scatter](/assets/images/plotly/plotly_heatmap.png)


### 1.8 **버블 차트 (Bubble Chart)**
- **이론**: 버블 차트는 산점도의 변형으로, 점의 크기를 통해 세 번째 변수를 표현합니다.
- **실습 예제**:
  ```python
  import plotly.express as px
  df = px.data.gapminder()  # Gapminder 데이터셋 로드
  fig = px.scatter(df.query("year==2007"), x="gdpPercap", y="lifeExp",
                   size="pop", color="continent", hover_name="country",
                   size_max=60, title="2007년 국가별 GDP와 기대수명")
  fig.show()
  ```
![plotly_scatter](/assets/images/plotly/plotly_bubble.png)


### 1.9 **3D 산점도 (3D Scatter Plot)**
- **이론**: 3D 산점도는 3차원 데이터를 시각화하는 데 사용됩니다.
- **실습 예제**:
  ```python
  import plotly.express as px
  df = px.data.iris()  # Iris 데이터셋 로드
  fig = px.scatter_3d(df, x="sepal_width", y="sepal_length", z="petal_width",
                      color="species", title="3D 산점도 예제")
  fig.show()
  ```
![plotly_scatter](/assets/images/plotly/plotly_3d_scatter.png)


### 1.10 **지도 시각화 (Choropleth Map)**
- **이론**: 지도 시각화는 지리적 데이터를 색상으로 표현하여 지역별 차이를 보여줍니다.
- **실습 예제**:
  ```python
  import plotly.express as px
  df = px.data.gapminder().query("year==2007")  # Gapminder 데이터셋 로드
  fig = px.choropleth(df, locations="iso_alpha", color="lifeExp",
                      hover_name="country", title="2007년 국가별 기대수명")
  fig.show()
  ```
![plotly_scatter](/assets/images/plotly/plotly_cholopleth_map.png)



## 2. Dash

### **2.1 Dash의 기본 개념**
- **Dash**는 Flask, Plotly.js, React.js를 기반으로 한 Python 프레임워크입니다.
- **핵심 구성 요소**:
  - **Layout**: 애플리케이션의 UI를 정의합니다. HTML 요소와 Plotly 그래프를 포함합니다.
  - **Callbacks**: 사용자 입력에 따라 동적으로 애플리케이션을 업데이트합니다.
  - **Graphs**: Plotly를 활용한 데이터 시각화를 제공합니다.

### **2.2 Dash의 주요 모듈**
- `dash.Dash`: Dash 애플리케이션을 생성합니다.
- `dash.html`: HTML 요소를 정의합니다.
- `dash.dcc`: 대화형 컴포넌트(슬라이더, 드롭다운 등)를 제공합니다.
- `@app.callback`: 입력(Input)과 출력(Output)을 연결하여 동적 인터페이스를 구현합니다.

### **2.3 Dash의 장점**
- **코드 기반 개발**: HTML/CSS/JavaScript를 직접 작성하지 않고도 Python 코드로 웹 애플리케이션을 구축할 수 있습니다.
- **데이터 시각화**: Plotly.js를 통한 고급 데이터 시각화 지원.
- **확장성**: 복잡한 애플리케이션에도 적용 가능하며, Flask 확장성을 그대로 활용할 수 있습니다.

### **2.4 기본 Dash 애플리케이션 만들기**

```python
from dash import Dash, html

# Dash 애플리케이션 생성
app = Dash(__name__)

# Layout 정의: HTML 요소 추가
app.layout = html.Div([
    html.H1("첫 번째 Dash 애플리케이션"),
    html.P("이것은 간단한 Dash 애플리케이션입니다.")
])

# 서버 실행
if __name__ == "__main__":
    app.run(debug=True)
```

**실습 목표**:
- Dash 애플리케이션의 기본 구조를 이해합니다.
- `html.Div`, `html.H1`, `html.P`와 같은 HTML 요소를 사용하여 레이아웃을 구성합니다.



### **2.5 대화형 그래프 추가하기**


```python
from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd

# 데이터 로드
df = px.data.gapminder().query("country=='Canada'")

# Dash 애플리케이션 생성
app = Dash(__name__)

# Layout 정의: 그래프 추가
app.layout = html.Div([
    html.H1("캐나다의 GDP 변화"),
    dcc.Graph(
        figure=px.line(df, x="year", y="gdpPercap", title="GDP per Capita Over Time")
    )
])

# 서버 실행
if __name__ == "__main__":
    app.run(debug=True)
```

**실습 목표**:
- `dcc.Graph`를 사용하여 Plotly 그래프를 애플리케이션에 추가합니다.
- 데이터 시각화의 기본 개념을 학습합니다.



### **2.6 Callback을 활용한 상호작용**


```python
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# 데이터 로드
df = px.data.gapminder()

# Dash 애플리케이션 생성
app = Dash(__name__)

# Layout 정의: 드롭다운 메뉴와 그래프 추가
app.layout = html.Div([
    html.H1("국가별 GDP 변화"),
    dcc.Dropdown(
        id="country-dropdown",
        options=[{"label": country, "value": country} for country in df["country"].unique()],
        value="Canada"
    ),
    dcc.Graph(id="gdp-graph")
])

# Callback 정의: 드롭다운 선택에 따라 그래프 업데이트
@app.callback(
    Output("gdp-graph", "figure"),
    Input("country-dropdown", "value")
)
def update_graph(selected_country):
    filtered_df = df[df["country"] == selected_country]
    fig = px.line(filtered_df, x="year", y="gdpPercap", title=f"{selected_country}의 GDP 변화")
    return fig

# 서버 실행
if __name__ == "__main__":
    app.run(debug=True)
```

1.**작동 원리 요약**
- **사용자 입력**:
   - 사용자가 드롭다운 메뉴에서 국가를 선택하면, 해당 값이 `Input("country-dropdown", "value")`로 전달됩니다.

- **함수 호출**:
   - Dash는 `@app.callback`에 의해 등록된 `update_graph()` 함수를 호출합니다.
   - `selected_country` 매개변수에 사용자가 선택한 값이 전달됩니다.

- **데이터 처리**:
   - `update_graph()` 함수는 선택된 국가의 데이터를 필터링하고, Plotly를 사용하여 그래프 객체(`fig`)를 생성합니다.

- **그래프 업데이트**:
   - `update_graph()` 함수가 반환한 `fig` 객체는 `Output("gdp-graph", "figure")`에 의해 `dcc.Graph`의 `figure` 속성에 할당됩니다.
   - 결과적으로, 화면에 새로운 그래프가 표시됩니다.

2.**실습 목표**:
- `dcc.Dropdown`을 사용하여 사용자 입력을 받습니다.
- `@app.callback`을 통해 입력과 출력을 연결하고, 동적 인터페이스를 구현합니다.



### **2.7 스타일링과 레이아웃 최적화**


```python
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# 데이터 로드
df = px.data.gapminder()

# Dash 애플리케이션 생성
app = Dash(__name__)

# Layout 정의: 스타일링 추가
app.layout = html.Div(style={"padding": "20px", "font-family": "Arial"}, children=[
    html.H1("국가별 GDP 변화", style={"color": "blue"}),
    dcc.Dropdown( #드롭다운 메뉴를 생성
        id="country-dropdown", # 이 ID를 통해 값을 참조
        options=[{"label": country, "value": country} for country in df["country"].unique()],
        value="Canada",
        style={"width": "50%", "margin-bottom": "20px"}
    ),
    dcc.Graph(id="gdp-graph") #그래프를 표시하는 영역을 생성,이 ID를 통해 그래프를 업데이트
])

# Callback 정의
@app.callback( #사용자 입력에 따라 애플리케이션이 동적으로 반응
    Output("gdp-graph", "figure"), #dcc.Graph의 figure 속성을 업데이트
    Input("country-dropdown", "value") #dcc.Dropdown에서 선택된 값을 입력으로
)
def update_graph(selected_country):
    filtered_df = df[df["country"] == selected_country]
    fig = px.line(filtered_df, x="year", y="gdpPercap", title=f"{selected_country}의 GDP 변화")
    return fig

# 서버 실행
if __name__ == "__main__":
    app.run(debug=True)
```

**실습 목표**:
- CSS 스타일을 적용하여 애플리케이션의 디자인을 개선합니다.
- `style` 속성을 사용하여 요소의 크기, 색상 등을 조정합니다.

### **2.8 Dash를 간편하게 배포하는 방법**

1.**Render.com 사용**
[Render](https://render.com/)은 Heroku와 유사한 클라우드 플랫폼으로, 무료로 Dash 애플리케이션을 배포할 수 있습니다.

- 절차:
    1. Render 계정 생성 및 GitHub 저장소 연동.
    2. 저장소에서 Dash 애플리케이션 코드와 `requirements.txt`를 업로드(gunicorn 추가).
    4. render.com 프로젝트 설정에서 start code에 다음과 같이 입력
        ```
        #[앞의 app은 소스 파일명(app.py)]:[두번째 app은 코드내 Dash()의 인스턴스명]
        gunicorn app:app
        ```
    3. Render가 자동으로 애플리케이션을 빌드하고 실행.

- 장점:
    - 무료로 사용 가능하며, 자동 배포를 지원합니다.

2.**PythonAnywhere 사용**
-   [PythonAnywhere](https://www.pythonanywhere.com/)는 Python 전용 호스팅 서비스로, Dash 애플리케이션을 쉽게 배포할 수 있습니다.
-   Dash는 기본적으로 Flask 기반으로 만들어져 있으므로, Flask Web App 방식으로 PythonAnywhere에 올리면 됩니다.

- 절차:
    1. PythonAnywhere 계정 생성하고, 새 Flask Web App 생성
    2. 대시보드에서 Bash 콘솔을 열고, GitHub 저장소를 클론하거나 파일을 업로드.
    3. Dash 앱을 Flask로 감싸기(기존 app.py 일부 수정)
        - Dash는 자체 서버를 가지고 있지만, PythonAnywhere는 WSGI(Web Server Gateway Interface) 방식으로 Flask 앱을 실행하므로 app.py를 다음처럼 수정 필요:
        ```python
        # WSGI용 Flask 서버 연결
        # Dash는 내부적으로 Flask 웹 프레임워크를 기반으로 동작
        # app.server는 Dash 앱 내부에 있는 Flask 애플리케이션 객체를 참조
        app = dash.Dash(__name__)
        server = app.server  # WSGI용 Flask 앱(소스 추가)
        ```
    4. 가상 환경을 만들고 requirements.txt의 내용을 인스톨
    5. yourusername_pythonanywhere_com_wsgi.py파일의 내용을 다음과 같이 설정
        ```python
        import sys
s
        # add your project directory to the sys.path
        project_home = '/home/yourusername/yourprojectfolder'
        if project_home not in sys.path:
            sys.path = [project_home] + sys.path

        # import flask app but need to call it "application" for WSGI to work
        # app은 파일명(app.py)이고 server는 app.py에 있는 server 변수를 말한다
        from app import server as application  # noqa
        ```
    4. Web 탭에서 reload 버튼 클릭 후 사용자명.pythonanywhere.com 찾아가기

- 장점:
    - 무료 계정으로도 충분히 사용 가능.
    - Python 개발자에게 친숙한 인터페이스.


## 3. Streamlit

### **3.1 Streamlit이란?**
- **Streamlit**은 Python 코드를 간단히 웹 애플리케이션으로 변환해주는 오픈소스 프레임워크입니다.
- **특징**:
  - **간결한 문법**: HTML/CSS/JavaScript를 몰라도 Python만으로 웹 앱을 만들 수 있습니다.
  - **대화형 인터페이스**: 슬라이더, 버튼, 입력 상자 등을 쉽게 추가할 수 있습니다.
  - **데이터 시각화**: Plotly, Matplotlib, Seaborn 등의 그래프를 손쉽게 통합할 수 있습니다.
  - **실시간 업데이트**: 코드가 변경되면 자동으로 애플리케이션이 업데이트됩니다.

### **3.2 주요 용도**
- **데이터 시각화**: 데이터를 대화형으로 탐색하고 시각화합니다.
- **머신러닝 모델 배포**: 학습된 모델을 웹 애플리케이션으로 배포하여 사용자와 상호작용합니다.
- **파일 처리**: CSV, Excel, 이미지 등의 파일을 업로드하고 처리합니다.
- **API 테스트**: REST API를 호출하고 결과를 시각화합니다.


### **3.3 Hello World 애플리케이션**

```python
import streamlit as st

# 제목 표시
st.title("첫 번째 Streamlit 애플리케이션")

# 텍스트 출력
st.write("안녕하세요! 이것은 Streamlit의 첫 번째 애플리케이션입니다.")
```

1. 위 코드를 `app.py` 파일로 저장합니다.
2. 터미널에서 다음 명령어를 실행합니다:
   ```bash
   streamlit run app.py
   ```
3. 브라우저에서 자동으로 열리는 URL(예: `http://localhost:8501`)로 접속합니다.

**실습 목표**
- Streamlit 애플리케이션의 기본 구조를 이해합니다.
- `st.title`, `st.write` 같은 기본적인 함수를 사용하여 UI를 구성합니다.



### **3.4 대화형 요소 추가하기**

```python
import streamlit as st

# 제목 표시
st.title("사용자 입력 처리하기")

# 텍스트 입력
user_input = st.text_input("이름을 입력하세요:")
if user_input:
    st.write(f"안녕하세요, {user_input}님!")

# 숫자 입력
number = st.number_input("숫자를 입력하세요:", min_value=0, max_value=100, value=50)
st.write(f"입력된 숫자는 {number}입니다.")

ㄴ
slider_value = st.slider("슬라이더를 조정하세요:", 0, 100, 50)
st.write(f"슬라이더 값: {slider_value}")
```

**실습 목표**
- `st.text_input`, `st.number_input`, `st.slider`와 같은 대화형 요소를 사용하여 사용자 입력을 받습니다.
- 입력값에 따라 동적으로 애플리케이션을 업데이트합니다.


### **3.5 데이터 시각화**

```python
import streamlit as st
import pandas as pd
import plotly.express as px

# 데이터 로드
df = px.data.gapminder()

# 제목 표시
st.title("국가별 GDP 변화")

# 드롭다운 메뉴
country_list = df["country"].unique()
selected_country = st.selectbox("국가를 선택하세요:", country_list)

# 데이터 필터링
filtered_df = df[df["country"] == selected_country]

# 그래프 생성
fig = px.line(filtered_df, x="year", y="gdpPercap", title=f"{selected_country}의 GDP 변화")
st.plotly_chart(fig)
```

**실습 목표**
- `st.selectbox`를 사용하여 사용자가 데이터를 선택할 수 있게 합니다.
- Plotly와 Streamlit을 결합하여 데이터를 시각화합니다.



### **3.6 파일 업로드 및 처리**

```python
import streamlit as st
import pandas as pd

# 제목 표시
st.title("CSV 파일 업로드 및 분석")

# 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요:", type=["csv"])

if uploaded_file is not None:
    # 파일 읽기
    df = pd.read_csv(uploaded_file)
    
    # 데이터 표시
    st.write("업로드된 데이터:")
    st.dataframe(df)
    
    # 기본 통계 정보 표시
    st.write("기본 통계 정보:")
    st.write(df.describe())
```

**실습 목표**
- `st.file_uploader`를 사용하여 파일을 업로드합니다.
- Pandas를 사용하여 데이터를 처리하고 시각화합니다.



### **3.7 머신러닝 모델 배포**
- **모델 학습시키기**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib

# 데이터 준비
iris = load_iris()
X, y = iris.data, iris.target

# 모델 학습
model = RandomForestClassifier()
model.fit(X, y)

# 모델 저장
joblib.dump(model, "rfc_model.pkl", compress=3)
```

- **학습된 모델 로드하여 에측하기**

```python
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import joblib
import numpy as np


# 모델 로드 (학습된 모델 파일 필요)
model = joblib.load("rfc_model.pkl")  # model.pkl은 학습된 모델 파일

# 제목 표시
st.title("머신러닝 모델 예측")

# 사용자 입력
feature1 = st.number_input("Sepal Length:", value=4.9)
feature2 = st.number_input("Sepal Width:", value=3.0)
feature3 = st.number_input("Petal Length:", value=1.4)
feature4 = st.number_input("Petal Width:", value=0.2)

# 예측 버튼
if st.button("예측하기"):
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    prediction = model.predict(input_data)
    res = ""
    if prediction[0] == 0:
        res = "Setosa"
    elif prediction[0] == 1:
        res = "Versicolor"
    else:
        res = "Virginica"
    st.write(f"예측 결과: {res}")
```

### **3.8 날씨 API 연동**

```python
import streamlit as st
import requests

# 제목 표시
st.title("날씨 정보 조회")

# OpenWeatherMap API 키 (https://openweathermap.org/ 에서 발급)
API_KEY = "자신의 API 키로 대체하세요"  

# 사용자 입력
city = st.text_input("도시 이름을 입력하세요:", "Seoul")

# API 호출
if city:
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&lang=kr&units=metric"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()

        # 날씨 정보 추출
        weather = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]

        # 결과 표시
        st.write(f"### {city}의 현재 날씨")
        st.write(f"- 상태: {weather}")
        st.write(f"- 온도: {temperature}°C")
        st.write(f"- 습도: {humidity}%")
    else:
        st.error("도시 이름을 확인하거나 API 키를 다시 확인해주세요.")
``

**실행 방법**
1. OpenWeatherMap(https://openweathermap.org/)에서 무료 API 키를 발급받습니다.
2. 위 코드에서 `your_api_key_here`를 발급받은 API 키로 대체합니다.
3. `streamlit run app.py` 명령어로 실행합니다.

**실습 목표**
- `requests` 라이브러리를 사용하여 API 데이터를 가져옵니다.
- JSON 형식의 데이터를 파싱하고 필요한 정보를 추출합니다.
- Streamlit으로 API 결과를 시각화합니다.

### 3.9 Stremalit 앱 배포하기
- [streamlit 클라우드에](https://streamlit.io/cloud) 무료 회원가입
- 배포할 소스코드를 Github와 연동하여 배포하기
