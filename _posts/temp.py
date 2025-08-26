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
