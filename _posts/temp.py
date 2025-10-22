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


## 1. 실습용 게시판
App.js - 전체 상태 관리 및 조율
LoginPage.js - 로그인 화면
Header.js - 헤더 (사용자명, 로그아웃)
PostForm.js - 게시물 작성/수정 폼
BoardList.js - 게시물 목록
PostItem.js - 개별 게시물
CommentSection.js - 댓글 섹션
CommentItem.js - 개별 댓글
CommentForm.js - 댓글 입력 폼

React 게시판 프로젝트를 처음 시작하는 학생들에게는 **점진적 학습과 성취감 경험**이 가장 중요합니다. 다음 접근법을 제안드립니다:

## 🎯 학습 전략 방향

### 1. **컴포넌트 계층 구조 이해부터 시작**
- **트리 구조로 시각화**: App.js를 루트로 한 컴포넌트 관계도 그리기
- **데이터 흐름 이해**: props가 어떻게 상위→하위로 전달되는지 설명
- **상태 위치 결정 원칙**: 각 상태가 어떤 컴포넌트에 위치해야 하는지 논의

### 2. **점진적 구현 단계 설정**

```
1단계: 정적 UI 구현
2단계: 상태 관리 추가
3단계: 상호작용 구현
4단계: 데이터 흐름 최적화
```

