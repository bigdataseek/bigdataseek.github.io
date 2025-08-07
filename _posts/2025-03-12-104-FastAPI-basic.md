---
title: 4차시 4:FastAPI 
layout: single
classes: wide
categories:
  - FastAPI
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---
## 1. Fast API Crash Course

- 출처: [FastAPI Full Crash Course - Python’s Fastest Web Framework](https://www.youtube.com/watch?v=rvFsGRvj9jo&t=3272s)

### 1.1 소개
* **목표:** 간결하고 빠르게 Fast API 학습
* **핵심 내용:**
  * Fast API란 무엇이며, 언제 Django나 Flask 대신 사용하는가?
    * 각 프레임워크의 장단점 비교를 통한 적절한 사용 시나리오 이해
    * 프로젝트 규모와 요구사항에 따른 프레임워크 선택 가이드
  * API 구축 기본 개념 (async/await, HTTP 메서드, 상태 코드, 쿼리/경로 매개변수, POST 요청 등)
    * 현대적인 비동기 프로그래밍 패러다임 이해
    * RESTful API 설계 원칙과 HTTP 프로토콜의 핵심 개념 습득
  * Pydantic을 활용한 데이터 검증
    * 타입 힌팅과 데이터 유효성 검사를 통한 안정적인 API 개발
    * 자동 문서화 기능과 연계된 데이터 모델 설계 방법

### 1.2 Fast API vs Flask vs Django
* **Django:** 포괄적인 기능 제공, ORM (Object-Relational Mapping) 및 다양한 내장 기능 포함
  * 대규모 웹 애플리케이션 개발에 적합한 "배터리 포함" 프레임워크
  * 관리자 패널, 인증 시스템, 폼 처리 등 풍부한 기능 제공으로 빠른 개발 가능
  * 학습 곡선이 높고 가벼운 프로젝트에는 다소 무거울 수 있음
* **Flask:** 최대한의 단순성과 유연성 제공, 모든 것을 직접 구축
  * 마이크로 프레임워크로서 핵심 기능만 제공하여 가볍고 시작하기 쉬움
  * 확장성이 뛰어나 필요한 기능만 선택적으로 추가 가능
  * 대규모 프로젝트에서는 구조화에 추가 노력이 필요할 수 있음
* **Fast API:** 고성능 API 구축에 최적화, 비동기 처리 지원
  * 현대적인 Python 기능을 최대한 활용한 우수한 개발자 경험 제공
  * Flask보다 빠르고 Django보다 가벼우면서도 타입 안전성 제공
  * 자동 문서화 기능으로 API 개발 및 테스트 과정 간소화

### 1.3 API 기본 개념
* **API (Application Programming Interface):** 서버의 애플리케이션과 통신할 수 있는 엔드포인트들의 집합
  * 소프트웨어 간 상호작용을 가능하게 하는 중간 계층으로서의 역할
  * 프론트엔드와 백엔드의 명확한 분리를 통한 확장성 및 유지보수성 향상
* 클라이언트가 특정 엔드포인트에 요청을 보내고 응답을 받음
  * HTTP/HTTPS 프로토콜을 통한 표준화된 통신 방식
  * JSON, XML 등 다양한 데이터 형식 지원을 통한 유연한 정보 교환
* API는 엔드포인트 정의, 유효성 검사 등을 포함. (백엔드 전체는 API + 데이터베이스 레이어, 유틸리티, 비즈니스 로직 등을 포함)
  * API 설계는 전체 시스템 아키텍처의 일부로서 중요한 의미를 가짐
  * 명확한 관심사 분리와 모듈화를 통한 유지보수성 향상
* **Fast API:** 고성능 API 구축을 위한 Python 프레임워크
  * 2018년에 처음 출시된 비교적 새로운 프레임워크지만 빠르게 인기 상승 중
  * Python 3.6+ 의 타입 힌팅 기능을 최대한 활용한 현대적 설계
* ASGI (Asynchronous Server Gateway Interface) 표준 기반
  * 기존 WSGI의 한계를 넘어선 비동기 처리 지원으로 높은 동시성 처리 가능
  * WebSocket 등 다양한 프로토콜 지원을 통한 실시간 애플리케이션 개발 용이
* Starlette (ASGI 프레임워크) 및 Pydantic (데이터 유효성 검사) 활용
  * 검증된 라이브러리들의 장점을 결합하여 안정성과 성능 모두 확보
  * "배터리 포함" 철학으로 추가 모듈 설치 없이 핵심 기능 바로 사용 가능

### 1.4 API를 사용하는 이유
* API는 웹 애플리케이션의 백엔드 역할을 수행
  * 비즈니스 로직과 데이터 처리를 담당하는 서버 측 구성 요소
  * 다양한 클라이언트(웹, 모바일, IoT 등)에서 일관된 방식으로 접근 가능
* 프론트엔드 (React, Vue, Angular 등)는 API에 요청을 보내 데이터를 가져와 표시 (API와 프론트엔드를 분리하는 것이 일반적인 웹 애플리케이션의 구조)
  * 프론트엔드와 백엔드의 독립적인 개발 및 배포가 가능해짐
  * SPA(Single Page Application)와 같은 현대적인 웹 개발 방식 지원
  * 서버 부하 분산과 클라이언트 측 캐싱을 통한 성능 최적화 가능
* Flask나 Django는 백엔드와 프론트엔드를 하나의 프로젝트에서 처리하는 경우가 많음
  * 전통적인 모놀리식 구조에서는 템플릿 엔진을 통해 HTML을 서버에서 생성
  * API 기반 아키텍처는 백엔드와 프론트엔드의 명확한 분리로 확장성 향상
  * 마이크로서비스 아키텍처로의 전환을 용이하게 함

### 1.5 대화형 API 문서
* **대화형 API 문서:** 
    - FastAPI 애플리케이션을 실행한 후 `/docs` 경로로 접속하면 Swagger UI를 통해 API 엔드포인트와 요청/응답 스키마를 시각적으로 확인하고 테스트 가능.
    - `/redoc` 경로에서는 ReDoc 스타일의 API 문서를 확인할 수 있다.
    - 예시: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
    - 예시: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

* **OpenAPI 스키마:** 
    - FastAPI는 정의된 API를 기반으로 OpenAPI 스키마를 자동으로 생성합니다. 
    - 이 스키마는 `/openapi.json` 경로에서 JSON 형식으로 확인할 수 있으며, API 문서 생성 및 클라이언트 코드 자동 생성 등 다양한 도구에서 활용됩니다.
    - 예시: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)


### 1.6 간단한 To-Do List API 구축 (실습)
* **목표:** To-Do List API를 구축하면서 Fast API의 기본 개념 학습
  * 실제 프로젝트와 유사한 환경에서 핵심 기능 직접 구현
  * 기본 CRUD(Create, Read, Update, Delete) 작업의 구현 방법 습득
* **의사 데이터베이스:** 딕셔너리 형태의 리스트를 사용하여 데이터 저장 (실제 데이터베이스 연결은 다루지 않음)
  * 개념 학습에 집중하기 위한 단순화된 데이터 저장소
  * 메모리 내 데이터 구조를 통한 빠른 프로토타이핑
* **기본 설정**
  * `pip install fastapi` - Fast API 프레임워크 설치
  * `pip install uvicorn` - ASGI 서버 설치 (실행에 필요)
  * `from fastapi import FastAPI` - Fast API 임포트
  * `api = FastAPI()` - Fast API 애플리케이션 인스턴스 생성
  * `uvicorn main:app --reload` - 개발 서버 실행 명령어 (코드 변경 시 자동 재로드)
* **HTTP 메서드:**
  * `GET`: 정보 조회 - 서버에서 데이터를 읽어오는 용도 (멱등성 보장)
  * `POST`: 새로운 정보 생성 - 서버에 새 리소스 생성 요청
  * `PUT`: 기존 정보 수정 - 리소스 전체 대체 (멱등성 보장)
  * `PATCH`: 기존 정보 부분 수정 - 리소스의 일부만 업데이트
  * `DELETE`: 정보 삭제 - 서버에서 리소스 제거 요청

* **엔드포인트 정의**

```python
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
async def read_root():
    return {"message": "Hello World"}
```
  * 데코레이터 패턴을 사용하여 직관적인 라우팅 정의
  * 비동기 함수를 통한 높은 동시성 처리 가능
  * JSON 응답을 자동으로 직렬화하여 반환
* **경로 매개변수 (Path Parameters):** URL 경로의 일부로 정보를 전달 `/todos/{todo_id}`
  * 리소스를 고유하게 식별하는 데 사용되는 매개변수
  * URL 구조에 통합되어 RESTful API 설계 원칙 준수
* **쿼리 매개변수 (Query Parameters):** URL의 쿼리 문자열로 정보를 전달 `/?first_n=3`
  * 필터링, 정렬, 페이지네이션 등에 활용되는 선택적 매개변수
  * 기본값 설정을 통한 사용자 친화적 API 설계 가능
* 매개변수 타입 지정 중요 (예: `todo_id: int`)
  * 타입 힌팅을 통한 자동 유효성 검사 및 문서화
  * 잘못된 데이터 형식 입력 시 자동으로 적절한 오류 응답 생성
* **자동 문서화:** `/docs` 경로에서 Swagger UI를 통해 API 문서 확인 및 테스트 가능
  * 코드 작성과 동시에 자동으로 최신 문서 생성
  * 대화형 UI를 통한 API 테스트로 개발 효율성 향상
  * `/redoc` 경로에서 ReDoc을 통한 대안적 문서화 방식도 제공

### 1.7 Pydantic을 활용한 데이터 모델 정의 및 검증
* **Pydantic:** 데이터 유효성 검사 및 모델 정의를 위한 라이브러리
  * Python의 타입 힌팅을 활용한 런타임 데이터 검증
  * 복잡한 데이터 구조를 선언적으로 정의하고 검증하는 강력한 도구
  * JSON 스키마 생성 및 직렬화/역직렬화 기능 내장
* **데이터 모델 정의:**
  * `BaseModel` 클래스를 상속받아 데이터 모델 정의
    * 객체 지향적 접근 방식으로 데이터 구조화
    * 자동 직렬화/역직렬화로 JSON과 Python 객체 간 변환 용이
  * `Field`를 사용하여 필드의 제약 조건, 설명 등을 설정
    * 최소/최대 길이, 정규식 패턴 등 다양한 검증 규칙 적용 가능
    * 필드 메타데이터를 통한 자동 문서화 강화
  * `Enum`을 사용하여 열거형 데이터 타입 정의
    * 제한된 선택지를 가진 필드를 명확하게 정의
    * 문서화와 타입 안전성 동시 확보

* **데이터 모델 활용**

```python
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field

class Priority(int, Enum):
    LOW = 3
    MEDIUM = 2
    HIGH = 1

class TodoBase(BaseModel):
    todo_name: str = Field(..., min_length=3, max_length=512, description="Name of the to-do")
    todo_description: str = Field(..., description="Description of the Todo")
    priority: Priority = Field(Priority.LOW, description="Priority of the to-do")

class TodoCreate(TodoBase):
    pass

class TodoUpdate(BaseModel):
    todo_name: Optional[str] = None
    todo_description: Optional[str] = None
    priority: Optional[Priority] = None

class Todo(TodoBase):
    todo_id: int = Field(..., description="Unique identifier of the to-do")

# Example use of models in API endpoints
@app.post("/todos/", response_model=Todo)
async def create_todo(todo: TodoCreate):
    # Logic to create the to-do in the database
    # Return the created to-do
    return Todo(**todo.dict(), todo_id=123)
```
  * 상속을 통한 모델 재사용으로 코드 중복 최소화
  * 생성, 조회, 업데이트 작업에 특화된 모델 분리를 통한 명확한 API 설계
  * `response_model` 매개변수를 통한 응답 데이터 구조 명시적 정의
* **HTTPException:** 예외 처리 및 HTTP 상태 코드 반환
  * 적절한 상태 코드와 오류 메시지를 포함한 예외 처리
  * 클라이언트에게 의미 있는 오류 정보 제공으로 디버깅 용이성 향상
  * 예: `raise HTTPException(status_code=404, detail="Todo not found")`

### 1.8 추가 학습 방향
* 쿠키, 미들웨어, 데이터베이스 연결, 프론트엔드 연동 등 Fast API 고급 기능
  * 보안: JWT 인증, OAuth2 통합, CORS 설정
  * 의존성 주입 시스템을 활용한 코드 모듈화 및 테스트 용이성 향상
  * 백그라운드 태스크와 WebSocket을 활용한 실시간 기능 구현
* 실제 프로젝트를 통한 Fast API 활용
  * 마이크로서비스 아키텍처에서의 Fast API 활용 방안
  * 도커 컨테이너화 및 쿠버네티스 배포 전략
  * CI/CD 파이프라인 구축을 통한 자동화된 테스트 및 배포
* SQLAlchemy를 활용한 ORM
  * 관계형 데이터베이스와의 효율적인 연동
  * 비동기 ORM을 통한 데이터베이스 작업의 성능 최적화
  * 마이그레이션 관리 및 데이터베이스 스키마 버전 제어

## 2.FastAPI CRUD 예제- 할 일 관리 API

### **2.1 기본적인 FastAPI 애플리케이션 구조**
```python
app = FastAPI(
    title="할 일 관리 API",
    description="FastAPI를 이용한 CRUD 기능이 있는 할 일 관리 API 예제입니다.",
    version="1.0.0"
)
```

- **설명**: `FastAPI` 인스턴스를 생성하여 애플리케이션을 초기화합니다.
  - `title`, `description`, `version`은 API 문서(Swagger UI 또는 ReDoc)에서 표시되는 메타데이터입니다.
  - FastAPI가 자동으로 API 문서를 생성한다. (`/docs` 또는 `/redoc` 경로에서 확인 가능)

### **2.2 Pydantic 모델과 데이터 검증**
```python
class TodoBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=100, description="할 일 제목")
    description: Optional[str] = Field(None, max_length=1000, description="할 일 상세 설명")
    priority: Priority = Field(default=Priority.MEDIUM, description="할 일 우선순위")
    due_date: Optional[str] = Field(None, description="마감일 (YYYY-MM-DD 형식)")
```

- **설명**:
  - `BaseModel`을 상속받아 데이터 모델을 정의합니다.
  - `Field`를 사용하여 필드에 대한 추가적인 제약 조건(예: 길이, 기본값, 설명)을 설정할 수 있습니다.
  - `Priority`는 `Enum` 클래스를 통해 제한된 값을 가지도록 정의됩니다.
  -  Pydantic의 데이터 검증 기능. 잘못된 데이터가 들어오면 자동으로 오류를 반환.


### **2.3 CRUD 작업**

1.**CREATE - 새로운 할 일 추가**

```python
@app.post("/todos", response_model=Todo, status_code=status.HTTP_201_CREATED, tags=["할 일"])
async def create_todo(todo: TodoCreate):
    new_id = max([t["id"] for t in todos_db], default=0) + 1
    current_date = datetime.now()strftime("%Y-%m-%d")
    new_todo = {
        "id": new_id,
        **todo.dict(),
        "completed": False,
        "created_at": current_date
    }
    todos_db.append(new_todo)
    return new_todo
```

- **설명**:
  - `@app.post` 데코레이터로 POST 요청을 처리합니다.
  - `response_model=Todo`는 응답 데이터의 구조를 명시적으로 정의합니다.
  - `status_code=status.HTTP_201_CREATED`는 성공 시 HTTP 상태 코드를 201로 설정합니다.
  - 새로운 할 일의 ID는 현재 데이터베이스의 최대 ID에 1을 더해 생성합니다.
  - 학생들에게 `todo.dict()`가 Pydantic 모델을 딕셔너리로 변환하는 방법이라고 설명하세요.


2.**READ - 모든 할 일 읽기**

```python
@app.get("/todos", response_model=List[Todo], tags=["할 일"])
async def read_todos(
    completed: Optional[bool] = Query(None, description="완료 상태로 필터링"),
    priority: Optional[Priority] = Query(None, description="우선순위로 필터링")
):
    filtered_todos = todos_db
    if completed is not None:
        filtered_todos = [todo for todo in filtered_todos if todo["completed"] == completed]
    if priority:
        filtered_todos = [todo for todo in filtered_todos if todo["priority"] == priority]
    return filtered_todos
```

- **설명**:
  - `@app.get` 데코레이터로 GET 요청을 처리합니다.
  - `Query`를 사용하여 쿼리 파라미터를 정의하고 설명을 추가할 수 있습니다.
  - 필터링 로직은 리스트 컴프리헨션을 사용하여 간단히 구현됩니다.
  - 학생들에게 쿼리 파라미터를 활용한 데이터 필터링 방법을 강조하세요.


3.**READ - 특정 할 일 읽기**

```python
@app.get("/todos/{todo_id}", response_model=Todo, tags=["할 일"])
async def read_todo(todo_id: int):
    todo = find_todo(todo_id)
    if todo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ID {todo_id}인 할 일을 찾을 수 없습니다."
        )
    return todo
```

- **설명**:
  - `{todo_id}`는 경로 파라미터로, 동적으로 URL에 포함됩니다.
  - `find_todo` 함수를 사용하여 해당 ID의 할 일을 찾고, 없을 경우 `HTTPException`을 발생시킵니다.
  - 학생들에게 HTTP 상태 코드(404 Not Found)를 적절히 사용하는 중요성을 설명하세요.

4.**UPDATE - 전체 업데이트 (PUT)**

```python
@app.put("/todos/{todo_id}", response_model=Todo, tags=["할 일"])
async def update_todo(todo_id: int, todo_update: TodoCreate):
    todo = find_todo(todo_id)
    if todo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ID {todo_id}인 할 일을 찾을 수 없습니다."
        )
    todo_index = todos_db.index(todo)
    todos_db[todo_index] = {
        **todo,  # 기존 데이터 유지
        **todo_update.dict()  # 업데이트된 데이터
    }
    return todos_db[todo_index]
```

- **설명**:
  - `@app.put` 데코레이터로 PUT 요청을 처리합니다.
  - `todo_update.dict()`를 사용하여 업데이트할 데이터를 병합합니다.
  - 학생들에게 PUT과 PATCH의 차이를 설명하세요. PUT은 전체 데이터를 교체하고, PATCH는 부분적으로 업데이트합니다.

5.**DELETE - 할 일 삭제**

```python
@app.delete("/todos/{todo_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["할 일"])
async def delete_todo(todo_id: int):
    todo = find_todo(todo_id)
    if todo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ID {todo_id}인 할 일을 찾을 수 없습니다."
        )
    todos_db.remove(todo)
```

- **설명**:
  - `@app.delete` 데코레이터로 DELETE 요청을 처리합니다.
  - 성공 시 `204 No Content` 상태 코드를 반환합니다.
  - 삭제된 데이터는 응답 본문에 포함되지 않습니다.


### **2.4 추가 기능**
1.**완료된 할 일 통계**

```python
@app.get("/todos/stats/completed", tags=["통계"])
async def get_completed_count():
    completed_count = sum(1 for todo in todos_db if todo["completed"])
    total_count = len(todos_db)
    return {
        "completed_count": completed_count,
        "total_count": total_count,
        "completion_rate": f"{(completed_count / total_count * 100) if total_count > 0 else 0:.1f}%"
    }
```

- **설명**:
  - 완료된 할 일 개수와 총 할 일 개수를 계산하여 완료율을 반환합니다.
  - 추가적인 비즈니스 로직을 API에 포함시키는 방법을 설명.


2.**우선순위별 할 일 통계**

```python
@app.get("/todos/stats/priority", tags=["통계"])
async def get_priority_stats():
    priorities = {p.value: 0 for p in Priority}
    for todo in todos_db:
        priority = todo["priority"]
        priorities[priority] += 1
    return priorities
```

- **설명**:
  - 각 우선순위별 할 일 개수를 계산하여 반환합니다.
  - 학생들에게 데이터를 요약하고 집계하는 방법을 설명하세요.

### **2.5 실행 및 테스트**
```python
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
```

- **설명**:
  - `uvicorn`을 사용하여 애플리케이션을 실행합니다.
  - `reload=True`는 코드 변경 시 자동으로 서버를 재시작합니다.
  - 학생들에게 `/docs` 경로에서 API를 직접 테스트할 수 있다고 알려주세요.

### **2.6 주요 포인트**
1. **FastAPI의 장점**:
   - 간결하고 직관적인 문법.
   - 자동으로 생성되는 API 문서.
   - 비동기 지원으로 성능 향상.

2. **Pydantic의 역할**:
   - 데이터 검증과 직렬화를 쉽게 처리.

3. **CRUD의 기본 원칙**:
   - 각 HTTP 메소드(GET, POST, PUT, DELETE)가 어떤 역할을 하는지 이해하기.

4. **실제 프로젝트에서는 데이터베이스 사용**:
   - 현재는 메모리에 데이터를 저장하지만, 실제 프로젝트에서는 데이터베이스(SQLAlchemy, MongoDB 등)를 사용해야 합니다.
