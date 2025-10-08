import requests
import json

# 1. API 정보 설정
# 발급받은 API 키를 여기에 입력하세요.
# https://www.data.go.kr/tcs/dss/selectApiDataDetailView.do?publicDataPk=15073861

api_key = "Q4hfV10oV1Z8Q8J8Bf8MMS1YfQQRgK3b5IpyjhzKfTxelN%2Fh9ZG8UcEhTLeJ1CHy0WODvaEb61tjjbMyTQaJgQ%3D%3D"

# API 요청 URL
url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMinuDustFrcstDspth"

# 2. 요청 변수(Parameters) 설정
# 필요한 요청 변수들을 딕셔너리 형태로 정의합니다.
params = {
    "serviceKey": api_key,  # API 키
    "returnType": "json",  # 응답 데이터 형식 (JSON)
    "numOfRows": "100",  # 한 페이지 결과 수
    "searchDate": "2025-08-19",
    "InformCode": "PM10",
}

try:
    # 3. HTTP GET 요청 보내기
    response = requests.get(url, params=params)

    # 4. 응답 확인
    if response.status_code == 200:
        # 응답 본문을 JSON 형태로 변환
        data = response.json()

        # 5. 데이터 파싱 및 출력
        # 필요한 정보(예: 측정소 이름, 미세먼지 농도)를 추출하여 출력합니다.
        items = data["response"]["body"]["items"]

        if items:
            print("=== 서울시 미세먼지 측정 정보 ===")
            for item in items:
                print(f"측정소: {item['stationName']}, 시간: {item['dataTime']}")
                print(f"미세먼지(PM10): {item['pm10Value']} µg/m³")
                print("-" * 30)
        else:
            print("데이터가 없습니다.")
    else:
        print(f"API 요청 실패: 상태 코드 {response.status_code}")
        print(f"에러 메시지: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"네트워크 오류 발생: {e}")




좋아요 👍 자바스크립트 입문자용으로 이해하기 쉬운 **외부 JavaScript 파일 사용 예시**를 아래처럼 작성할 수 있습니다.
HTML과 JS 파일을 분리한 기본 예제입니다.

---

### 📄 index.html

```html
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>외부 JavaScript 예제</title>
</head>
<body>
  <h1>외부 JavaScript 파일 예제</h1>
  <button onclick="sayHello()">인사하기</button>

  <!-- 외부 JS 파일 불러오기 -->
  <script src="app.js"></script>
</body>
</html>
```

### 📄 app.js

```javascript
// 외부 JavaScript 파일 (app.js)
function sayHello() {
  alert("안녕하세요! 외부 JavaScript 파일에서 실행되었습니다 😊");
}
```

---

### 💡설명

* `app.js` 파일은 HTML과 분리되어 있으므로, 코드를 **더 깔끔하게 관리**할 수 있습니다.
* `<script src="app.js"></script>` 태그를 이용해 **HTML에 연결**합니다.
* 여러 페이지에서 같은 `app.js`를 불러오면 **공통 기능을 재사용**할 수 있습니다.

---

원하신다면 이 예제를 “HTML 안에 직접 JS를 넣은 버전 → 외부 JS로 분리한 버전” 비교 형태로도 만들어드릴까요? (입문자에게 차이를 시각적으로 보여주기에 좋습니다.)
