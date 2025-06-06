---
title: 15차시 1:n8n (Beginner Course)
layout: single
classes: wide
categories:
  - n8n
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

![beginner_n8n](/assets/images/beginner_n8n.png)

## **1. Automation 소개**
### **1.1 자동화는 데이터기반 의사결정의 핵심 도구이다.**
- 자동화는 반복적이고 시간이 많이 걸리는 작업을 효율적으로 처리하여, 사람들의 창의적이고 중요한 업무에 집중할 수 있도록 돕습니다.
- 특히 데이터를 수집하고 분석하여 의사결정을 지원하는 데 중요한 역할을 합니다. 예를 들어, 특정 조건이 충족되면 바로 알림을 보내거나 데이터를 정리해주는 방식으로 활용됩니다.

### **1.2 자동화의 주요 개념**
- **Trigger (트리거)**  
  - 자동화가 시작되는 지점입니다. 특정 이벤트나 조건이 발생하면 자동화 프로세스가 작동합니다.  
    - 예: 이메일 수신, 파일 업로드, 데이터베이스 변경 등.

- **Filtering (필터링)**  
  - 트리거된 데이터 중 필요한 정보만 선택하거나 조건에 맞는 데이터를 걸러냅니다.  
    - 예: "특정 키워드가 포함된 이메일만 처리한다"와 같이 조건을 설정.

- **Actions (액션 / 앱)**  
  - 필터링된 데이터를 기반으로 수행되는 작업입니다. 다양한 앱이나 서비스와 연동하여 실행됩니다.  
    - 예: 데이터 저장, 메시지 전송, 파일 변환 등.

- **사례**  
  - **예시 1:** 새로운 고객이 웹사이트에서 회원가입을 하면(Trigger), 이메일로 환영 메시지를 보내고(Action) CRM 시스템에 정보를 저장합니다(Filtering).  
  - **예시 2:** 특정 해시태그가 포함된 트윗이 게시되면(Trigger), 이를 데이터베이스에 저장하고(Action) 관련 팀에게 알림을 보냅니다(Filtering).

### **1.3 자동화를 위한 최선의 관행 (Automation Best Practices)**
- **Mapping (매핑)**  
  - 서로 다른 시스템 간의 데이터를 연결하고 일치시키는 과정입니다.  
    - 예: 한 시스템에서 사용자 이름이 "Name"으로 표시되고, 다른 시스템에서는 "Full Name"으로 표시될 때, 이를 매핑하여 데이터가 올바르게 전달되도록 합니다.  
  - 매핑은 자동화에서 오류를 줄이고 데이터의 일관성을 유지하는 데 중요합니다.

- **권장 사항 (Best Practices)**  
  - **단순화(Simplicity):** 복잡한 자동화보다는 작은 단위로 나눠서 구현하세요. 큰 프로세스도 여러 작은 워크플로우로 분리하면 관리가 쉽습니다.  
  - **테스트(Testing):** 자동화된 프로세스를 실제 적용하기 전에 테스트 환경에서 충분히 검증하세요. 특히 에러 핸들링(오류 처리) 부분을 꼭 확인해야 합니다.  
  - **유연성(Flexibility):** 요구사항이 변경되거나 확장될 때 쉽게 수정할 수 있는 구조로 설계하세요. 예를 들어, 조건이나 규칙을 파라미터화하여 재사용성을 높이는 것이 좋습니다.  
  - **문서화(Documentation):** 자동화된 워크플로우의 목적과 동작 방식을 문서화하세요. 나중에 다른 사람이 이해하거나 유지보수하기 쉬워집니다.

## **2. API 및 Webhook**
### **2.1. API란 무엇인가?**
- **API(Application Programming Interface)**는 소프트웨어나 서비스가 외부에서 사용할 수 있도록 제공하는 인터페이스입니다.
  - 예: 날씨 정보를 제공하는 서비스, 결제 시스템 등.
- 개발자는 API를 통해 특정 서비스의 기능을 "소비"하거나 데이터를 주고받을 수 있습니다.
- **동작 방법**은 문서(API 문서)로 정의되어 있으며, 이를 통해 요청(request)과 응답(response) 방식을 이해.
- **클라이언트**는 서버에 요청을 보내고, **서버**는 그 요청에 대해 응답을 반환합니다.

### **2.2 Request(요청)의 구성 요소**
API 요청은 다음 네 가지 주요 요소로 구성됩니다:
- **URL**: 요청을 보낼 주소 (예: `https://api.example.com/weather`).
- **Method**: 요청 유형 (예: GET, POST, PUT, DELETE).  
  - GET: 데이터를 읽기 위해 사용.  
  - POST: 데이터를 생성하거나 업데이트하기 위해 사용.
- **Header**: 요청에 대한 부가 정보를 포함 (예: 인증 토큰, 데이터 형식).
- **Body**: 요청과 함께 보낼 데이터 (예: 사용자 정보, 파일 등).
- 추가적으로 **Credential**(인증 정보)가 필요할 수 있습니다.  
  - 예: API 키, OAuth 토큰 등.

### **2.3 Response(응답)의 구성 요소**
서버로부터 받는 응답은 다음 세 가지 주요 요소로 구성됩니다:
- **Status Code**: 요청이 성공했는지 실패했는지를 나타내는 코드.  
  - 예: 200(성공), 404(찾을 수 없음), 500(서버 오류).
- **Header**: 응답에 대한 부가 정보 (예: 데이터 형식, 캐싱 정보).
- **Body**: 실제 데이터가 포함된 부분 (예: JSON 형식의 데이터).

### **2.4 Webhook**
- **Webhook**은 특정 이벤트가 발생했을 때 자동으로 알림을 보내는 메커니즘입니다.
  - 예: 새로운 메시지가 도착했을 때, 결제가 완료되었을 때 등.
- **사례**: 친구가 집에 도착했을 때 현관 벨이 울리는 것처럼, 특정 이벤트가 발생시 서버가 당신에게 즉시 알림.

- **Polling과의 차이점**  
  - **Polling**: 클라이언트가 서버에 "계속 물어보는" 방식.  
    - 예: "새로운 메시지가 있니?"라고 반복적으로 확인.
    - 비효율적일 수 있음(불필요한 요청 발생 가능).
  - **Webhook**: 서버가 이벤트가 발생했을 때 "즉시 알려주는" 방식.  
    - 예: "새로운 메시지가 도착했으니 확인해!"라고 알림.
    - 더 효율적이고 실시간 처리가 가능.

## **3. n8n Nodes**
### **3.1. What is a Node?**
- **노드(Node)**는 n8n에서 자동화 워크플로우를 구성하는 기본 블록입니다.  
  - 각 노드는 특정 작업을 수행하며, 여러 노드를 연결하여 복잡한 자동화를 구현합니다.

- **노드의 3가지 카테고리**
  - **Entry Point (진입점)**: 자동화를 시작하는 지점입니다.  
    - 예: 트리거(Trigger) 노드.
  - **Function (기능)**: 데이터를 처리하거나 액션을 수행하는 노드입니다.  
    - 예: 데이터 변환, 앱과의 상호작용.
  - **Exit Point (종료점)**: 자동화 결과를 내보내거나 저장하는 노드입니다.  
    - 예: 이메일 발송, 파일 저장.

- **노드의 유형**
  - **Trigger**: 자동화를 시작하는 이벤트를 감지합니다.  
  - **Actions in Apps**: 특정 앱(예: Google Sheets, Slack 등)과 상호작용합니다.  
  - **Data Transformation**: 데이터를 필터링, 매핑, 형식 변환합니다.  
  - **Flow**: 워크플로우의 흐름을 제어합니다.  
  - **Files**: 파일을 읽거나 씁니다.  
  - **Advanced**: 고급 설정 및 커스텀 로직을 추가합니다.

### **3.2. Canvas**
- **Canvas**는 n8n에서 워크플로우를 설계하고 노드를 배치하는 공간입니다.  

- **Adding Nodes to the Canvas**
  - **첫 번째 노드 추가**:  
    - Canvas에서 "Add Node" 버튼을 클릭하거나, 검색창에 원하는 노드 이름을 입력합니다.  
    - 예: "Manual Trigger" 노드를 추가하여 워크플로우를 시작합니다.
  - **노드 추가**:  
    - 기존 노드에 마우스를 올리고 "+" 버튼을 클릭하여 다음 노드를 추가합니다.  
    - 예: "Google Sheets" 노드를 추가하여 시트 데이터를 처리합니다.

### **3.3 Node Actions**
- **주요 액션**
  - **Excuse Node (Play)**: 노드를 실행합니다.  
  - **Open Node Settings**: 노드의 설정을 열어 수정합니다.  
  - **Remove/Deactivate/Delete**: 노드를 삭제하거나 비활성화합니다.  
  - **Pin/Copy/Duplicate**: 노드를 고정, 복사, 또는 복제합니다.

- **Parameters (매개변수)**
  - **Credentials**: API 키나 인증 정보를 입력합니다.  
  - **Resource**: 사용할 리소스(예: 시트, 메시지).  
  - **Operation**: 수행할 작업(예: 데이터 읽기, 쓰기).  
  - **Action Settings**: 작업에 필요한 세부 설정을 조정합니다.

- **Settings (설정)**
  - **Notes**: 노드에 대한 메모를 추가합니다.  
  - **Visual Settings**: 노드의 색상이나 아이콘을 변경합니다.  
  - **Execution Settings**: 실행 조건이나 제한을 설정합니다.

### **3.4 간단한 데모**

- **Step 1: Manual Trigger 노드 추가**  
  - Canvas에 "Manual Trigger" 노드를 추가하여 워크플로우를 시작합니다.  
  - 이 노드는 수동으로 실행될 때 작동합니다.

- **Step 2: Google Sheets 노드 추가 및 설정**  
  - "Google Sheets" 노드를 추가하고, Google 계정을 연결합니다.  
  - 매개변수를 다음과 같이 설정합니다:
    - **Operation**: `Get Rows` 선택 (시트의 모든 행을 가져옵니다).  
    - **Sheet ID**: 데이터를 가져올 Google Sheet의 ID를 입력하거나 시트 이름을 선택.  
      - 예: "학생명단"이라는 시트를 선택합니다.  
    - **Range**: 기본값(전체 시트)으로 두거나 별도 설정 없이 진행합니다.  
  - 이렇게 하면 특정 시트의 모든 데이터를 가져올 준비가 완료됩니다.

- **Step 3: 워크플로우 실행**  
  - "Play" 버튼을 클릭하여 워크플로우를 실행합니다.  
  - Google Sheets의 데이터가 성공적으로 가져와지는지 확인합니다.  
  - 결과는 n8n의 **Debug Panel**에서 확인(예: 행과 열 데이터가 테이블 형식으로 표시됨).

## **4. n8n Data**
### **4.1 주요 Data 개념**
- **Data Structure**
  - **JSON**:  
    - JSON(JavaScript Object Notation)은 n8n에서 데이터를 표현하는 기본 형식입니다.  
    - 예: `{ "name": "Alice", "age": 25 }`  
    - 각 데이터는 키(key)와 값(value)으로 구성됩니다.
    
  - **Array of JSON Objects**:  
    - 여러 개의 JSON 객체를 모아놓은 배열(Array) 형태입니다.  
    - 예: `[ { "name": "Alice", "age": 25 }, { "name": "Bob", "age": 30 } ]`

- **JSON 객체 배열과 테이블 Mapping**  
  - JSON 객체 배열은 테이블 데이터와 1:1로 매핑될 수 있습니다.  
    - 배열의 각 JSON 객체는 테이블의 한 행(row)에 해당합니다.  
    - JSON 객체의 키는 테이블의 열(column)에 해당합니다.  
    - 예:  
      ```json
      [
        { "name": "Alice", "age": 25 },
        { "name": "Bob", "age": 30 }
      ]
      ```
      → 이는 다음과 같은 테이블로 표현될 수 있습니다:
      ```
      | name   | age |
      |--------|-----|
      | Alice  | 25  |
      | Bob    | 30  |
      ```

- **용어: Item, Items**
  - **Item**: JSON 객체 하나를 의미합니다. (예: `{ "name": "Alice", "age": 25 }`)  
  - **Items**: JSON 객체들의 배열을 의미합니다. (예: `[ { "name": "Alice", "age": 25 }, { "name": "Bob", "age": 30 } ]`)  

### **4.2 노드가 items를 사용하는 법**
- **Item Execution**
  - n8n의 노드는 입력 데이터의 **각 아이템(item)**마다 실행됩니다.  
    - 예: 입력 데이터가 3개의 JSON 객체(items)로 구성된 경우, 노드는 각 객체(item)에 대해 한 번씩 실행됩니다.  

- **Execution Schema**
  - n8n의 워크플로우는 데이터를 처리하기 위해 **데이터 흐름 기반**으로 동작합니다.  
    - 각 노드는 이전 노드에서 출력된 데이터(items)를 입력으로 받아 처리합니다.  
    - 결과적으로, 데이터는 체인(chain)처럼 연결된 노드들 사이를 흐릅니다.

- **Reading Data from Items**
  - 한 노드에서 생성된 데이터를 다음 노드에서 사용할 수 있습니다.  
    - 예: Google Sheets 노드에서 데이터를 읽어온 후, Slack 노드로 전송할 때 데이터를 참조합니다.  
    - 데이터 참조 방법:  
      - `{%raw%}{{$json.key}}{%endraw%}` 형식으로 이전 노드의 데이터를 접근할 수 있습니다.  
      - 예: `{%raw%}{{$json.name}}{%endraw%}` → 이전 노드에서 "name" 필드의 값을 가져옵니다.

- **Edit Fields Node**
  - **Edit Fields** 노드는 데이터를 수정하거나 새로운 필드를 추가하는 데 사용됩니다.  
    - 주요 작업:  
      - 기존 필드 값을 변경하거나 삭제합니다.  
      - 새로운 필드를 추가합니다.  
    - 예:  
      - 기존 데이터: `{ "name": "Alice", "age": 25 }`  
      - 새로운 필드 추가: `{ "name": "Alice", "age": 25, "status": "active" }`

## **5.Workflow 개요**
### **5.1 주요 Workflow 개념들**

- **Workflow**
  - **워크플로우(Workflow)**는 n8n에서 자동화를 구성하는 기본 단위입니다.  
    - 여러 개의 **노드(Node)**를 연결하여 데이터를 처리하거나 작업을 수행합니다.  
    - 예: 이메일 수신 → 데이터 저장 → 알림 발송.

- **Workflow Settings**
  - **워크플로우 설정**은 전체 워크플로우의 동작 방식을 정의합니다.  
    - 주요 설정 항목:
      - **이름(Name)**: 워크플로우의 이름을 지정합니다.  
      - **실행 모드(Execution Mode)**: 워크플로우가 언제 실행될지 설정합니다 (예: 수동 실행, 스케줄링).  
      - **에러 처리(Error Handling)**: 오류 발생 시 어떻게 대응할지 설정합니다.  


### **5.2 노드 간 연결**
- **Triggers**
  - **트리거(Trigger)**는 워크플로우를 시작하는 지점입니다.  
    - 특정 이벤트가 발생하면 워크플로우가 실행됩니다.  
    - 예:  
      - "Manual Trigger": 사용자가 수동으로 실행.  
      - "Webhook Trigger": 외부 서비스로부터 요청이 들어올 때 실행.  

- **Branches**
  - **분기(Branch)**는 워크플로우에서 여러 경로로 나뉘어 처리되는 경우를 말합니다.  
    - 분기를 사용하면 조건에 따라 서로 다른 작업을 수행할 수 있습니다.  

  - **분기 생성 방법**
    - **조건에 따라 여러 출력 옵션**:  
      - 하나의 노드에서 여러 가지 출력 옵션을 제공하는 경우, 각 옵션마다 다른 경로를 생성할 수 있습니다.  
      - 예: 조건에 따라 "성공" 또는 "실패"로 나누어 처리.

    - **2개의 출력 라인 생성**:  
      - 특정 노드에서 두 개 이상의 출력 라인을 생성하여 데이터를 다른 노드로 전달합니다.  
      - 예:  
        - 첫 번째 출력: 데이터 저장.  
        - 두 번째 출력: 알림 발송.  

## **6.Workflow Nodes**
### **6.1 유용한 Nodes**
- **Edit Fields (Set)**
  - **Edit Fields** 노드는 데이터를 수정하거나 새로운 필드를 추가하는 데 사용됩니다.  
    - 주요 기능:  
      - 기존 필드 값을 변경하거나 삭제합니다.  
      - 새로운 필드를 추가하여 데이터를 확장합니다.  
    - 예:  
      - 원본 데이터: `{ "name": "Alice", "age": 25 }`  
      - 새로운 필드 추가 후: `{ "name": "Alice", "age": 25, "status": "active" }`

- **Aggregate**
  - **Aggregate** 노드는 여러 개의 데이터 항목(items)을 하나로 결합    
    - 예: aggregate할 필드가 "name"인 경우
      - 입력 데이터: `[ { "name": "Alice", "score": 80 }, { "name": "Bob", "score": 90 } ]`  
      - 출력 데이터: `{ "name":["Alice", "Bob"]}`

- **Merge**
  - **Merge** 노드는 여러 입력 분기(branch)의 데이터를 하나로 합치는 데 사용됩니다. 
    - Append: 모든 입력 데이터를 순서대로 추가합니다.
    - Combine:
      - Matching Fields: 특정 필드를 기준으로 데이터를 결합합니다. SQL의 JOIN과 유사한 방식으로 작동합니다.
      - Position: 입력 데이터의 순서를 기준으로 데이터를 결합합니다.
      - All Possible Combinations: 가능한 모든 조합을 생성합니다.
    - Choose Branch: 특정 분기의 데이터만 선택합니다.
    
    - 예: Append할 경우
      - 입력 데이터: `[ { "name": "Alice", "score": 80 } ]` 와  ` [{ "name": "Bob", "score": 90 }]` 인 경우
      - 출력 데이터: `[ { "name": "Alice", "score": 80 },{ "name": "Bob", "score": 90 }]`


- **Webhook**
  - **Webhook** 노드는 외부 서비스로부터 이벤트를 받아 워크플로우를 실행하는 데 사용됩니다.  
    - 주요 기능:  
      - 특정 URL에 요청이 도착하면 워크플로우가 시작됩니다.  
      - 실시간으로 데이터를 받을 수 있습니다.  
    - 예: 외부 서비스에서 POST 요청을 보내면, 해당 데이터를 처리하고 저장합니다.

## **7. Error Handling**
### **7.1 Execution Logs**
- **Workflow Execution History**
  - 워크플로우가 실행될 때마다 **실행 기록**이 저장됩니다.  
    - 실행 시간, 상태(성공/실패), 입력 및 출력 데이터를 확인할 수 있습니다.  
    - **용도**: 문제 해결 또는 이전 실행 결과를 검토할 때 유용합니다.  

- **Node Execution History**
  - 각 노드의 실행 기록도 개별적으로 확인할 수 있습니다.  
    - 특정 노드에서 발생한 오류나 출력 데이터를 상세히 분석할 수 있습니다.  
    - **용도**: 노드별로 디버깅하거나 데이터 흐름을 추적하는 데 도움이 됩니다.  

### **7.2 Error Handling**
- **Error Node**
  - **Error 노드**는 워크플로우에서 발생한 오류를 캡처하고 처리하는 데 사용됩니다.  
    - 주요 기능:  
      - 오류 메시지를 로그로 남기거나 알림을 보냅니다.  
      - 실패한 작업을 재시도하거나 대체 작업으로 연결합니다.  
    - 예:  Google Sheets에 데이터를 저장 실패시, Slack으로 오류 알림을 보냄.  

- **Error Triggers & Workflow**
  - **에러 트리거**는 워크플로우 내에서 오류가 발생했을 때 자동으로 실행되는 경로입니다.  
    - 주요 설정:  
      - 특정 노드에서 오류가 발생하면, 이를 별도의 "에러 처리" 워크플로우로 연결합니다.  
      - 예: 데이터 처리 중 오류가 발생하면, 이를 기록하고 관리자에게 알립니다.  
    - **용도**: 안정적인 자동화를 위해 오류를 체계적으로 관리합니다.  

## **8. Debugging**
### **8.1 Debugging?**
- **디버깅(Debugging)**은 워크플로우가 제대로 작동하지 않을 때 문제를 찾아내고 해결하는 과정입니다.  
  - 주요 목표: 데이터 흐름이나 노드 동작에서 발생한 오류를 확인하고 수정합니다.  

### **8.2 Debug in Editor**
- n8n의 **디버그 패널(Debug Panel)**을 사용하여 각 노드의 입력 및 출력 데이터를 실시간으로 확인할 수 있습니다.  
  - **용도**:  
    - 특정 노드에서 데이터가 어떻게 변형되는지 추적합니다.  
    - 잘못된 데이터나 예상치 못한 결과를 식별합니다.  
  - **방법**:  
    - 노드를 실행하고 Debug Panel에서 "Input"과 "Output"을 검토합니다.  

### **8.3 Retrying**
- **Retry 기능**: 디버깅이 끝난 후 워크플로우 실행을 재시도하는 기능입니다.  
  - 오류가 발생한 노드부터 다시 시작하여 문제를 해결한 후의 동작을 확인합니다.

- **Retry 방법**
  - **현재 저장된 워크플로우로 재시도(Retry with currently saved workflow)**:  
    - 디버깅 과정에서 변경 사항이 반영된 최신 버전의 워크플로우로 재실행합니다.  
    - 예: 오류가 발생한 노드를 수정한 후, 그 노드부터 재시작합니다.

  - **원래 워크플로우로 재시도(Retry with original workflow)**:  
    - 오류가 발생하기 전의 원래 상태로 돌아가서 재실행합니다.  
    - 예: 오류가 발생한 노드 이전의 데이터 흐름을 다시 검증합니다.
    - "내가 수정한 내용 때문이 아니라, 원래 워크플로우 자체에 문제가 있었던 건가?"를 검증.

- **Copy to Editor 기능**
  - 오류가 발생한 노드 이전의 데이터를 복사하여 에디터에 붙여넣을 수 있습니다.  
    - 이를 통해 디버깅 중에 데이터를 잃지 않고 원하는 지점에서 재시작할 수 있습니다.  

### **8.4 Edit Output**
- **Edit Output**은 노드의 출력 데이터를 직접 수정하거나 가공하는 기능입니다.  
  - 주요 용도:  
    - 다음 노드로 전달되기 전에 데이터를 조정하거나 필터링합니다.  
    - 예: 불필요한 필드를 제거하거나 새로운 값을 추가합니다.  
  - **방법**:  
    - "Edit Fields" 노드를 사용하거나 직접 JSON 데이터를 수정합니다.  

### **8.5 Workflow Version History**
- **워크플로우 버전 이력(Version History)**은 이전에 저장된 워크플로우 상태를 복구하거나 비교할 수 있는 기능입니다.  
  - 주요 용도:  
    - 변경 사항이 문제가 될 경우 이전 버전으로 돌아갈 수 있습니다.  
    - 여러 버전 간 차이점을 확인하여 문제를 분석합니다.  
  - **방법**:  
    - 워크플로우 설정에서 "Version History"를 열어 이전 버전을 선택하거나 복원합니다.  

학생들에게 **n8n**의 협업(Collaboration) 기능을 설명할 때, 아래와 같이 간단하고 핵심적인 내용으로 전달할 수 있습니다.

## **9. Collaboration**

### **9.1 Community**
- **Community 포럼 (community.n8n.io)**:  
  - n8n 사용자들이 서로 질문하고 답변하며 경험을 공유하는 공간입니다.  
  - 예: 문제 해결, 워크플로우 아이디어, 팁과 트릭 등.  

### **9.2 Templates**
- **템플릿(Templates)**은 미리 만들어진 워크플로우를 제공하여 빠르게 시작할 수 있도록 돕습니다.  
  - 주요 용도:  
    - 자주 사용되는 작업(예: 이메일 발송, 데이터 수집)을 쉽게 설정.  
    - 커뮤니티에서 다른 사용자가 공유한 템플릿 활용.  

### **9.3 User Management**
- **사용자 관리**는 팀 내에서 역할에 따라 접근 권한을 부여하는 기능입니다.  
  - 주요 역할:  
    - **Owner**: 전체 시스템을 관리하고 모든 권한을 가집니다.  
    - **Admin**: 워크플로우와 사용자를 관리하지만 소유권은 제한적입니다.  
    - **Member**: 특정 워크플로우에만 접근하거나 수정할 수 있습니다.  

### **9.4 Workflow Sharing**
- **워크플로우 공유**는 팀원들과 워크플로우를 함께 사용하거나 협업할 수 있는 기능입니다.  
  - **Workflow Access**:  
    - 특정 사용자 또는 그룹에게 읽기/쓰기 권한을 부여합니다.  
    - 예: 팀원이 워크플로우를 수정하거나 실행하도록 허용.  

### **9.5 Credential Sharing**
- **자격 증명 공유**는 API 키나 로그인 정보 같은 민감한 데이터를 팀원들과 안전하게 공유하는 기능입니다.  
  - 주요 특징:  
    - 암호화된 상태로 저장 및 공유됩니다.  
    - 팀원들이 각자의 워크플로우에서 동일한 자격 증명을 사용할 수 있습니다.  

### **9.6 n8n API**
- **n8n API**는 외부 서비스나 스크립트를 통해 n8n을 프로그래밍 방식으로 제어할 수 있는 인터페이스입니다.  
  - 주요 용도:  
    - 워크플로우를 원격으로 실행하거나 관리.  
    - 데이터를 자동으로 가져오거나 내보내기.  
