---
title: 30차시 1:LangChain
layout: single
classes: wide
categories:
  - LangChain
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---
## 1. LangChain 핵심 정리

- 출처: [LangChain Explained in 13 Minutes \| QuickStart Tutorial for Beginners
](https://www.youtube.com/watch?v=aywZrzNaKjs&t=1s)

### 1.1 **LangChain 이란?**

* **AI 개발자를 위한 오픈 소스 프레임워크:** 
    - 인공지능(AI) 기반의 애플리케이션을 개발하는 개발자들이 보다 쉽고 효율적으로 작업을 수행할 수 있도록 다양한 도구와 기능을 제공하는 자유롭게 사용할 수 있는 소프트웨어 개발 도구 모음
* **GPT-4와 같은 대규모 언어 모델(LLM)을 외부 데이터 소스 및 연산과 결합 가능:** 
    - 단순히 텍스트 생성 능력만 가진 LLM을 넘어서, 개발자가 보유한 특정 데이터베이스, 문서 파일 등 외부 정보와 복잡한 계산 기능을 통합하여 더욱 스마트하고 상황에 맞는 AI 서비스를 구축
* **현재 Python, JavaScript(Typescript) 패키지로 제공:** 
    - 현재 가장 널리 사용되는 프로그래밍 언어인 파이썬과 웹 개발에 주로 사용되는 자바스크립트(타입스크립트 포함) 형태로 제공되어, 다양한 개발 환경에서 LangChain을 활용

### 1.2 **LangChain을 사용하는 이유?**
1.  **데이터 연결:**
    * **LLM을 자체 데이터 소스(데이터베이스, PDF 파일 등)에 연결하여 특정 정보 활용 가능:** 
        - LLM이 미리 학습된 일반적인 지식 외에도, 기업 내부 데이터, 개인 문서 등 특정 도메인이나 사용자의 필요에 맞는 정보를 활용하여 답변하거나 작업을 수행
    * **단순 텍스트 스니펫 붙여넣기 방식이 아닌, 전체 데이터베이스 참조:** 
        - 사용자가 필요한 정보를 일일이 복사해서 LLM에 제공하는 번거로움 없이, LangChain을 통해 LLM이 전체 데이터베이스를 검색하고 필요한 정보를 스스로 찾아 활용
2.  **액션 수행:**
    * **필요한 정보를 얻은 후, LangChain을 통해 이메일 전송과 같은 특정 액션 수행 가능:** 
        - 단순히 정보를 제공하는 것을 넘어, 얻어진 정보를 바탕으로 실제로 이메일을 보내거나 특정 API를 호출하는 등 현실 세계와 상호작용하는 기능을 구현

### 1.3 **LangChain 작동 방식**

1.  **문서 분할 (Document Splitting):** 
    - LLM이 한 번에 처리할 수 있는 텍스트의 길이는 제한적이므로, LLM이 참조해야 할 긴 문서를 의미 있는 작은 덩어리(chunks)로 나누는 과정
    - 이를 통해 LLM은 문서 전체의 내용을 효율적으로 이해하고 처리할 수 있습니다.
2.  **벡터 데이터베이스 저장 (Vector Database Storage):** 
    - 분할된 텍스트 덩어리들을 텍스트의 의미를 수치화한 벡터 표현인 임베딩(embeddings) 형태로 변환
    - 유사한 의미를 가진 텍스트 덩어리들이 벡터 공간에서 가까이 위치하도록 벡터 데이터베이스에 저장
    - 이는 나중에 질문과 관련된 정보를 빠르게 찾는 데 중요한 역할을 합니다.
3.  **파이프라인 구축 (Pipeline Construction):** 여러 단계를 연결하여 특정 작업을 자동화
    * **사용자 질문 -> LLM 전달:** 
        - 사용자의 질문이 LangChain 파이프라인의 첫 번째 단계로 LLM에 전달됩니다.
    * **질문의 벡터 표현을 사용하여 벡터 데이터베이스에서 유사성 검색 수행 -> 관련 정보 덩어리 추출:** 
        - 사용자의 질문 또한 벡터 임베딩으로 변환된 후, 
        - 벡터 데이터베이스에서 의미적으로 유사한 텍스트 덩어리들을 빠르게 찾아냅니다.
    * **LLM은 질문 + 관련 정보를 기반으로 답변 제공 또는 액션 수행:** 
        - LLM은 원래의 질문과 검색된 관련 정보를 함께 고려하여 최종 답변을 생성, 정의된 액션을 수행

### 1.4 **LangChain의 가치**

* **데이터 인식 (Data-aware):** 
    - 단순히 학습된 일반 지식에 의존하는 것이 아니라, 벡터 스토어에 저장된 자체 데이터를 참조하여 질문에 답변하거나 작업을 수행할 수 있어 더욱 정확하고 맥락에 맞는 결과를 제공
* **액션 수행 (Agentic):** 
    - 질문에 대한 답변뿐만 아니라, 외부 도구나 API를 활용하여 실제로 이메일을 보내거나 코드를 실행하는 등 다양한 액션을 수행할 수 있는 지능적인 에이전트 구축을 가능
* **개인 비서, 학습, 코딩, 데이터 분석 등 다양한 분야에 적용 가능:** 
    - LangChain의 유연성과 확장성은 개인 맞춤형 AI 비서, 교육 콘텐츠 생성, 코드 자동 완성, 복잡한 데이터 분석 등 광범위한 분야에서 혁신적인 AI 애플리케이션 개발을 가능

### 1.5 **LangChain 주요 구성 요소**

1.  **LLM 래퍼 (LLM Wrappers):** 
    - OpenAI의 GPT-4, 허깅페이스(Hugging Face)의 다양한 트랜스포머 모델 등 다양한 대규모 언어 모델과의 쉽고 일관된 인터페이스를 제공하여, 
    - 개발자가 특정 LLM의 API에 종속되지 않고 편리하게 LLM을 통합
2.  **프롬프트 템플릿 (Prompt Templates):** 
    - LLM에 입력할 텍스트(프롬프트)를 미리 정의된 구조에 따라 동적으로 생성할 수 있도록 지원합니다. 
    - 사용자 입력, 외부 데이터 등을 효과적으로 프롬프트에 포함시켜 LLM의 답변 품질과 일관성 제고
3.  **인덱스 (Indexes):** 
    - LLM이 효율적으로 정보를 검색하고 활용할 수 있도록 데이터를 구조화하고 저장하는 방법을 제공
    - 벡터 스토어는 인덱스의 한 종류로, 텍스트 데이터를 의미 기반으로 검색하는 데 핵심적인 역할
4.  **체인 (Chains):** 
    - 여러 개의 LangChain 구성 요소(LLM, 프롬프트 템플릿, 인덱스 등)를 논리적인 순서로 연결하여 특정 작업을 수행하는 자동화된 파이프라인을 구축
    - 예를 들어, 질문 생성 -> 문서 검색 -> 답변 생성의 단계를 하나의 체인으로 묶음.
5.  **에이전트 (Agents):** 
    - LLM이 미리 정의된 도구(외부 API, 함수 등)를 스스로 선택, 실행하여 복잡한 작업을 자율 수행
    - 에이전트는 주어진 목표를 달성하기 위해 필요한 단계를 추론하고, 적절한 도구를 호출하며, 그 결과를 바탕으로 다음 행동을 결정하는 지능적인 역할

### 1.5 **예시 코드**
1.  **환경 설정:** 
    - LangChain을 사용하기 위해 라이브러리(python-dotenv, langchain, pinecone-client 등)를 설치
    - OpenAI API 키, Pinecone API 키 등 외부 서비스 접근에 필요한 인증 정보를 설정
2.  **LLM 래퍼:** 
    - LangChain이 제공하는 OpenAI 래퍼를 사용하여 GPT-3 또는 GPT-4와 같은 LLM 인스턴스를 생성
    - 간단한 질문을 던져 LLM의 기본적인 텍스트 생성 능력을 확인하는 예시
3.  **프롬프트 템플릿:** 
    - 사용자로부터 입력받은 정보를 템플릿 내의 특정 위치에 삽입하여 LLM에게 전달할 최종 프롬프트를 동적으로 생성하는 방법을 보여줍니다. 
    - 예: "이름: $\[사용자 이름\]$, 나이: $\[사용자 나이\]$인 사람에 대해 설명해 주세요."와 같은 템플릿을 사용
4.  **체인:** 
    - LLM 래퍼와 프롬프트 템플릿을 결합하여, 사용자 입력을 받아 프롬프트를 생성
    - 생성된 프롬프트를 LLM에 전달하여 응답을 얻는 전체 과정을 자동화하는 간단한 체인 구축 예시
5.  **임베딩 및 벡터 스토어:** 
    - 텍스트 데이터를 작은 덩어리로 나누고, 각 덩어리의 의미를 벡터로 표현하는 임베딩을 생성
    - 생성된 임베딩을 Pinecone과 같은 벡터 데이터베이스에 저장하고, 사용자 질문과 유사한 텍스트 덩어리를 검색하는 방법을 제시
6.  **에이전트:** 
    - LangChain의 Python 에이전트 기능을 사용하여 LLM이 Python 코드를 직접 실행하고 그 결과를 바탕으로 다음 행동을 결정하는 예시
    - 이를 통해 LLM은 단순한 텍스트 생성뿐만 아니라, 실제 연산을 수행하거나 외부 시스템과 상호작용


7. **예제 코드**

```python
# 1. 환경 설정
# 설치: pip install langchain langchain-openai langchain-pinecone python-dotenv pinecone

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "your-index-name"  # 실제 Pinecone 인덱스 이름으로 변경

# Pinecone 초기화 (최신 pinecone-client 사용)
from pinecone import Pinecone as PineconeClient

pinecone_client = PineconeClient(api_key=PINECONE_API_KEY)

# 2. LLM 래퍼
from langchain_openai import OpenAI

llm = OpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-instruct")  # 명시적으로 모델 지정
question = "오늘 날씨 어때?"
answer = llm.invoke(question)  # invoke()로 변경
print(f"질문: {question}")
print(f"답변: {answer}")

# 3. 프롬프트 템플릿
from langchain.prompts import ChatPromptTemplate  # ChatPromptTemplate 사용

template = "당신은 {subject} 전문가입니다. {query}에 대해 간결하게 답변해주세요."
prompt = ChatPromptTemplate.from_template(template)  # ChatPromptTemplate로 변경

subject = "고양이"
query = "가장 좋아하는 음식은?"
formatted_prompt = prompt.format(subject=subject, query=query)
output = llm.invoke(formatted_prompt)  # invoke()로 변경
print(f"생성된 프롬프트: {formatted_prompt}")
print(f"LLM 답변: {output}")

# 4. 체인 (RunnableSequence로 최신화)
from langchain_core.runnables import RunnableSequence

chain = prompt | llm  # | 연산자로 체인 구성
result = chain.invoke({"subject": "강아지", "query": "가장 좋아하는 장난감은?"})  # invoke()로 변경
print(f"체인 실행 결과: {result}")

# 5. 임베딩 및 벡터 스토어 (Pinecone)
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore  # langchain_pinecone 사용

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
text = "LangChain은 LLM 기반 애플리케이션 개발을 위한 프레임워크입니다. 다양한 기능을 제공합니다."
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_text(text)

# Pinecone 벡터 스토어 초기화
vectorstore = PineconeVectorStore.from_texts(
    texts,
    embeddings,
    index_name=PINECONE_INDEX_NAME,
    pinecone_api_key=PINECONE_API_KEY
)

# 유사성 검색
query = "LangChain의 주요 기능은 무엇인가요?"
docs = vectorstore.similarity_search(query)
print(docs[0].page_content)

# 6. 에이전트 (최신 create_react_agent 사용)
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.agents import load_tools

tools = load_tools(["python_repl"], llm=llm)
agent = create_react_agent(llm, tools, prompt_template=None)  # 기본 ReAct 프롬프트 사용
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 주의: 아래 코드는 실제 Python 코드를 실행하므로 신뢰할 수 없는 입력에 주의
result = agent_executor.invoke({"input": "2 더하기 2는 얼마야? 그리고 그 결과를 제곱해줘."})
print(f"에이전트 실행 결과: {result['output']}")
```

**참고:**

* `.env` 파일에 `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`를 실제 값으로 설정해야 코드가 정상적으로 실행됩니다.
* Pinecone 관련 코드는 Pinecone 벡터 데이터베이스가 미리 설정되어 있어야 정상적으로 작동합니다. `PINECONE_INDEX_NAME`을 실제 인덱스 이름으로 변경해야 합니다.
* 에이전트 예시 코드는 Python 코드를 실행하는 기능을 포함하므로 사용에 주의해야 합니다.

---

### 1.6 학습 퀴즈

1. LangChain의 주요 목적은 무엇이며, 개발자에게 어떤 이점을 제공합니까?
> LangChain의 주요 목적은 AI 개발자가 GPT-4와 같은 대규모 언어 모델(LLM)을 외부 연산 및 데이터 소스와 결합할 수 있도록 하는 것입니다. 이를 통해 개발자는 LLM의 일반적인 지식과 자체 데이터를 활용하여 특정 요구 사항에 맞는 애플리케이션을 구축할 수 있습니다.

2. 벡터 데이터베이스에서 임베딩이 중요한 이유는 무엇이며, LangChain은 이를 어떻게 활용하여 정보를 검색합니까?
> 임베딩은 텍스트의 벡터 표현이며, 벡터 데이터베이스에서 의미론적 유사성 검색을 가능하게 합니다. LangChain은 사용자 질문의 임베딩을 생성하고 이를 벡터 데이터베이스의 문서 청크 임베딩과 비교하여 관련 정보를 효율적으로 검색하고 LLM에 제공합니다.

3. LLM 래퍼(wrapper)는 LangChain에서 어떤 역할을 수행하며, 구체적인 예시를 들어 설명하십시오.
> LLM 래퍼는 LangChain이 다양한 LLM(예: OpenAI, Hugging Face)과 상호 작용할 수 있도록 하는 인터페이스 역할을 합니다. 예를 들어, OpenAI 래퍼를 사용하면 LangChain 내에서 OpenAI의 text-davinci-003 또는 gpt-3.5-turbo와 같은 모델을 쉽게 호출하고 사용할 수 있습니다.

4. 프롬프트 템플릿의 개념과 이것이 LLM 상호 작용을 어떻게 용이하게 하는지 설명하십시오.
> 프롬프트 템플릿은 LLM에 전달할 텍스트 프롬프트를 동적으로 생성하는 데 사용됩니다. 이를 통해 개발자는 하드 코딩된 텍스트 대신 변수를 사용하여 사용자 입력과 같은 동적 정보를 프롬프트에 삽입하고, 일관되고 사용자 정의 가능한 프롬프트를 생성할 수 있습니다.

5. LangChain에서 "체인(chain)"이란 무엇이며, 순차적 체인(sequential chain)은 어떻게 작동합니까?
> LangChain에서 "체인"은 LLM과 프롬프트 템플릿을 결합하여 사용자 입력에서 LLM 출력을 생성하는 인터페이스입니다. 순차적 체인은 여러 체인을 연결하여 하나의 체인의 출력을 다음 체인의 입력으로 사용하여 복잡한 작업을 수행할 수 있도록 합니다.

6. 텍스트 분할(text splitting)은 벡터 데이터베이스에 데이터를 저장하기 전에 왜 필요한 단계입니까?
> 텍스트 분할은 긴 문서를 더 작고 관리하기 쉬운 청크로 나누는 과정입니다. 이는 벡터 데이터베이스가 임베딩을 생성하고 유사성 검색을 수행하는 데 더 효율적이며, LLM이 더 관련성 높은 정보에 집중할 수 있도록 합니다.

7. LangChain 에이전트의 주요 기능은 무엇이며, 주어진 예시에서 파이썬 에이전트는 어떤 작업을 수행했습니까?
> LangChain 에이전트는 LLM이 외부 도구와 상호 작용하여 작업을 수행할 수 있도록 합니다. 주어진 예시에서 파이썬 에이전트는 numpy 라이브러리를 사용하여 이차 함수의 근을 찾는 파이썬 코드를 실행할 수 있었습니다.

8. LangChain이 데이터 인지적(data-aware) 애플리케이션과 행위 수행적(action-taking) 애플리케이션을 구축하는 데 어떻게 도움이 되는지 설명하십시오.
> LangChain은 벡터 스토어를 통해 자체 데이터 소스를 LLM에 연결하여 데이터 인지적 애플리케이션을 구축할 수 있도록 합니다. 또한 에이전트 프레임워크를 통해 LLM이 외부 API와 상호 작용하여 이메일 보내기와 같은 실제 작업을 수행할 수 있는 행위 수행적 애플리케이션을 만들 수 있습니다.

9. LangChain의 주요 가치 제안을 구성하는 다섯 가지 핵심 개념은 무엇입니까?
> LangChain의 주요 가치 제안은 LLM 래퍼, 프롬프트 템플릿, 그리고 정보를 추출하는 인덱스(벡터 스토어), 여러 구성 요소를 결합하는 체인, 외부 API와 상호 작용하는 에이전트라는 다섯 가지 핵심 개념으로 나눌 수 있습니다.

10. LangChain 프레임워크의 주요 구성 요소(예: 모델, 프롬프트, 체인, 인덱스, 에이전트)를 간략하게 설명하십시오.
>- 모델: LLM(예: GPT-4) 또는 채팅 모델(예: GPT-3.5 Turbo)에 대한 래퍼를 제공하여 LangChain 내에서 쉽게 사용할 수 있도록 합니다.
- 프롬프트: LLM에 대한 입력으로 사용되는 텍스트입니다. LangChain은 프롬프트 생성을 위한 템플릿을 제공합니다.
- 체인: LLM과 프롬프트 및 기타 유틸리티를 결합하여 특정 작업을 수행하는 엔드-투-엔드 파이프라인입니다.
- 인덱스: 외부 데이터 소스의 정보를 LLM이 사용할 수 있도록 구조화하는 방법입니다. 여기에는 벡터 스토어와 임베딩이 포함됩니다.
- 에이전트: LLM이 환경과 상호 작용하고 도구를 사용하여 작업을 수행할 수 있도록 하는 시스템입니다.

### 1.7 에세이 형식 질문
1. LangChain의 등장이 인공지능 애플리케이션 개발 방식에 어떤 혁신적인 변화를 가져왔으며, 그 잠재적 영향에 대해 논의하십시오.
> LangChain은 대형 언어 모델(LLM)을 활용한 애플리케이션 개발 방식에 있어 획기적인 변화를 가져왔다. 기존에는 LLM을 단순히 텍스트 생성 도구로 사용하거나, 특정 작업을 위해 수작업으로 프롬프트를 조정하는 방식이 주를 이루었다. 하지만 LangChain은 이러한 한계를 넘어 LLM을 더 복잡하고 유연한 시스템으로 통합할 수 있는 프레임워크를 제공한다.
>
>**혁신적인 변화:**
- **모듈화된 설계와 확장성**: LangChain은 LLM을 중심으로 다양한 구성 요소(프롬프트, 체인, 데이터베이스, 에이전트 등)를 모듈화하여 개발자가 쉽게 결합하고 재사용할 수 있도록 한다. 이는 반복적인 작업을 줄이고 개발 속도를 가속화한다.
- **데이터 중심의 상호작용**: LangChain은 외부 데이터 소스와의 연결성을 강화하여 LLM이 정적 지식에서 벗어나 실시간 데이터를 활용한 동적인 응답을 생성할 수 있게 한다. 이를 통해 애플리케이션은 더욱 맞춤화되고 현실적인 해결책을 제공할 수 있다.
- **자동화된 워크플로우**: LangChain의 체인과 에이전트 기능은 복잡한 작업을 자동화하여 사용자 입력에 따라 다단계 문제 해결 과정을 실행할 수 있다. 이는 인간의 개입을 최소화하면서도 고도의 지능적 행동을 가능하게 한다.
>
>**잠재적 영향:**
- **비즈니스 프로세스 혁신**: 고객 서비스, 마케팅, 운영 관리 등 다양한 분야에서 LLM 기반 자동화가 보편화될 수 있다. 예를 들어, 고객 문의 처리 시스템에서 LangChain을 활용하면 자연스러운 대화를 통해 문제를 해결하고 필요한 경우 외부 데이터베이스를 참조하여 정확한 정보를 제공할 수 있다.
- **교육 및 연구 분야의 발전**: 교육에서는 학습자의 요구에 맞춘 개인화된 콘텐츠를 생성하거나, 연구에서는 방대한 데이터를 분석하고 새로운 패턴을 제안하는 데 활용될 수 있다.
- **사회적 영향**: LangChain의 유연성 덕분에 작은 규모의 개발팀도 강력한 AI 기반 애플리케이션을 구축할 수 있어, 기술 격차를 줄이는 데 기여할 수 있다.

2. 데이터 인지적 LLM 애플리케이션을 구축하는 데 있어 벡터 데이터베이스와 임베딩의 역할은 무엇이며, LangChain은 이러한 기술을 어떻게 통합하여 활용하는지 구체적인 예를 들어 설명하십시오.
> **벡터 데이터베이스와 임베딩의 역할:**
- **임베딩(Embedding)**: 텍스트 데이터를 의미적으로 표현하기 위한 고차원 벡터로 변환하는 과정이다. 임베딩은 LLM이 텍스트 간의 유사성이나 관계를 이해하는 데 필수
- **벡터 데이터베이스(Vector Database)**: 임베딩된 데이터를 효율적으로 저장하고 검색할 수 있는 데이터베이스, 특히 대규모 데이터셋에서 빠르게 유사한 항목을 찾는 데 유용
>
>**LangChain의 통합 방법:**
LangChain은 벡터 데이터베이스와 임베딩 기술을 활용하여 LLM이 외부 데이터를 효과적으로 이해하고 활용할 수 있도록 지원한다. 예를 들어:
>
>1. **문서 검색 시스템 구축**:
- 사용자가 질문을 하면, LangChain은 해당 질문을 임베딩으로 변환하고, 이를 벡터 데이터베이스에 저장된 문서 임베딩과 비교하여 가장 관련성이 높은 문서를 찾음.
- 이후, 선택된 문서를 LLM에 전달하여 질문에 대한 정확한 답변을 생성한다.
>
>2. **개인화된 추천 시스템**:
- 사용자의 과거 행동 데이터를 임베딩으로 변환하여 벡터 데이터베이스에 저장한다.
- 사용자가 새로운 요청을 할 때마다 LangChain은 해당 행동과 유사한 패턴을 가진 데이터를 검색하고, 이를 기반으로 맞춤형 추천을 제공한다.
>
>**구체적인 예시**:
예를 들어, 의료 분야에서 환자의 증상을 분석하고 진단을 내리는 애플리케이션을 구축한다고 가정하자. 환자의 증상 데이터는 임베딩으로 변환되어 벡터 데이터베이스에 저장된다. LangChain은 환자의 현재 증상을 임베딩으로 변환하고, 이를 벡터 데이터베이스에서 유사한 사례를 검색하여 관련된 진단 정보를 찾아낸다. 이후 LLM은 해당 정보를 바탕으로 적절한 치료 옵션을 제안할 수 있다.

3. LangChain 에이전트의 개념을 설명하고, 실제 시나리오에서 에이전트가 어떻게 자율적으로 작업을 수행하고 문제를 해결할 수 있는지 다양한 응용 사례를 통해 논의하십시오.
> **LangChain 에이전트의 개념:**
LangChain 에이전트는 LLM을 기반으로 한 자율적인 문제 해결 엔진으로, 사용자의 목표를 달성하기 위해 여러 단계의 작업을 계획하고 실행할 수 있다. 에이전트는 외부 도구(API, 데이터베이스, 웹 크롤링 등)와 상호 작용하며, 필요에 따라 동적으로 결정을 내릴 수 있다.
>
>**실제 시나리오 및 응용 사례:**
>
>1. **고객 서비스 자동화**:
- 사용자가 "내 계정에 로그인할 수 없습니다"라는 문의를 남긴다.
- 에이전트는 먼저 사용자의 계정 상태를 확인하기 위해 CRM API를 호출한다.
- 계정이 잠겨 있으면, 에이전트는 사용자에게 비밀번호 재설정 링크를 보내거나 직접 비밀번호를 초기화하는 절차를 수행한다.
>
>2. **마켓 리서치 보고서 작성**:
- 사용자가 "최근 스마트폰 시장 트렌드에 대한 보고서를 작성해주세요"라고 요청한다.
- 에이전트는 웹 크롤링 도구를 사용하여 관련 기사를 수집하고, 이를 분석하여 핵심 트렌드를 요약한다.
- 이후, 수집된 데이터를 기반으로 LLM이 구조화된 보고서를 작성하여 제공한다.
>
>3. **금융 포트폴리오 최적화**:
- 사용자가 자신의 투자 목표와 위험 성향을 입력한다.
- 에이전트는 금융 데이터베이스를 분석하여 최적의 투자 포트폴리오를 설계하고, 이를 시각화하여 제안한다.
- 또한, 시장 상황에 따라 포트폴리오를 자동으로 조정할 수 있다.
>
>**결론:**
LangChain 에이전트는 다양한 도구와 데이터를 통합하여 자율적으로 작업을 수행할 수 있으며, 이를 통해 복잡한 문제를 단순화하고 효율성을 극대화할 수 있다.

4. LangChain 프레임워크의 주요 구성 요소(모델, 프롬프트, 체인, 인덱스, 에이전트)가 서로 어떻게 상호 작용하여 강력하고 유연한 LLM 기반 애플리케이션을 구축하는지 분석하십시오.
> LangChain의 주요 구성 요소는 각각 독립적으로 강력하지만, 서로 결합되었을 때 더욱 큰 가치를 창출한다.
>
>1. **모델(Model)**:
- LLM 자체를 의미하며, 텍스트 생성, 분석, 번역 등 다양한 작업을 수행한다.
- 다른 구성 요소로부터 입력을 받아 적절한 출력을 생성한다.
>
>2. **프롬프트(Prompt)**:
- 사용자 입력과 추가 컨텍스트를 결합하여 모델에게 명확한 지침을 제공한다.
- 예를 들어, "이 문서에서 중요한 내용을 요약해주세요"와 같은 프롬프트는 모델의 작업 범위를 제한하고 정확성을 높인다.
>
>3. **체인(Chain)**:
- 여러 단계의 작업을 순차적으로 연결하여 복잡한 작업을 자동화한다.
- 예를 들어, 문서 요약 → 질문 답변 → 결과 저장이라는 일련의 과정을 하나의 체인으로 구성할 수 있다.
>
>4. **인덱스(Index)**:
- 외부 데이터 소스를 효율적으로 관리하고 검색 가능한 형태로 저장한다.
- 벡터 데이터베이스와 임베딩을 활용하여 빠른 데이터 조회를 지원한다.
>
>5. **에이전트(Agent)**:
- 체인과 인덱스를 포함한 모든 구성 요소를 통합하여 자율적으로 작업을 수행한다.
- 예를 들어, 사용자의 요청에 따라 필요한 데이터를 수집하고, 이를 기반으로 모델을 활용하여 결과를 생성한다.
>
>**상호 작용 예시**:
사용자가 "최근 기술 트렌드에 대한 보고서를 작성해주세요"라고 요청하면:
>1. **프롬프트**가 사용자의 요청을 명확히 정의한다.
>2. **인덱스**는 관련 데이터를 벡터 데이터베이스에서 검색한다.
>3. **체인**은 데이터 수집 → 분석 → 보고서 작성의 단계를 순차적으로 실행한다.
>4. **에이전트**는 전체 과정을 감독하며, 필요 시 외부 도구를 호출한다.
>5. **모델**은 최종 보고서를 생성하여 사용자에게 제공한다.

5. LangChain이 해결하고자 하는 기존 LLM의 한계점은 무엇이며, LangChain의 기능을 활용함으로써 개발자는 어떤 새로운 가능성을 탐색할 수 있는지 심층적으로 논의하십시오.
> **기존 LLM의 한계점:**
>1. **정적 지식**: LLM은 학습된 데이터셋에 기반한 지식만을 활용할 수 있어, 실시간 데이터를 반영하지 못한다.
>2. **맥락 이해 부족**: 복잡한 문제를 해결하기 위해선 여러 단계의 추론이 필요하지만, LLM은 단일 입력에 대한 응답만을 생성하는 데 초점이 맞춰져 있다.
>3. **외부 도구 연동 어려움**: LLM 자체로는 외부 API나 데이터베이스와의 상호 작용이 불가능하다.
>
>**LangChain의 해결책:**
- **외부 데이터 연동**: 벡터 데이터베이스와 임베딩을 통해 LLM이 실시간 데이터를 활용할 수 있게 한다.
- **다단계 문제 해결**: 체인과 에이전트를 통해 복잡한 작업을 단계별로 처리할 수 있다.
- **도구 통합**: 다양한 외부 도구와의 연동을 지원하여 LLM의 기능을 확장한다.
>
>**새로운 가능성:**
>1. **맞춤형 애플리케이션 개발**: 개발자는 특정 산업이나 용도에 맞춘 애플리케이션을 쉽게 구축할 수 있다. 예를 들어, 법률 문서 분석, 의료 진단 지원 등.
>2. **자동화된 워크플로우 구축**: 비즈니스 프로세스를 자동화하여 운영 효율성을 극대화할 수 있다.
>3. **창의적 문제 해결**: 예술, 디자인, 게임 개발 등 창의적인 분야에서도 LLM을 활용한 새로운 접근법을 실험할 수 있다.
>
>**결론적으로**, LangChain은 LLM의 한계를 넘어선 새로운 패러다임을 제시하며, 개발자들이 AI 기술의 잠재력을 최대한 발휘할 수 있도록 지원한다. 

### 1.8 용어 해설
 
1. LLM (Large Language Model): 
    - 방대한 텍스트 데이터셋을 학습하여 인간과 유사한 텍스트를 이해하고 생성할 수 있는 심층 학습 모델입니다.

2. 프레임워크 (Framework): 
    - 소프트웨어 개발을 용이하게 하기 위해 제공되는 재사용 가능한 추상 디자인과 이를 지원하는 도구 및 라이브러리의 모음입니다.

3. API (Application Programming Interface): 
    - 서로 다른 소프트웨어 시스템이 통신하고 데이터를 교환할 수 있도록 하는 인터페이스입니다.

4. 벡터 데이터베이스 (Vector Database): 
    - 텍스트, 이미지, 오디오 등 다양한 형태의 데이터를 벡터 임베딩으로 변환하여 저장하고, 의미론적 유사성을 기반으로 효율적인 검색을 지원하는 데이터베이스입니다.

5. 임베딩 (Embedding): 
    - 텍스트 또는 기타 데이터의 의미론적 의미를 포착하는 숫자 벡터 표현입니다. 유사한 의미를 가진 데이터는 벡터 공간에서 서로 가까이 위치합니다.

6. 프롬프트 (Prompt): 
    - LLM에 제공되는 입력 텍스트로, 특정 응답이나 작업을 유도합니다.

7. 프롬프트 템플릿 (Prompt Template): 
    - 동적 정보를 삽입할 수 있는 재사용 가능한 프롬프트 구조입니다.

8. 체인 (Chain): 
    - LangChain에서 LLM, 프롬프트, 유틸리티 등을 연결하여 특정 작업을 수행하는 파이프라인입니다.

9. 에이전트 (Agent): 
    - LangChain에서 LLM을 사용하여 환경을 인식하고, 도구를 선택하여 작업을 수행하고, 목표를 달성하는 시스템입니다.

10. 토큰 (Token): 
    - 텍스트를 처리하기 위해 분할되는 기본 단위입니다. 단어 또는 단어의 일부가 될 수 있습니다.


## 2. LangChain 프레임워크 7단계 
### 2.1 **개요**
- 본 튜토리얼은 LangChain 프레임워크의 **핵심 철학과 강력한 기능**을 7단계로 세분화하여 상세히 설명하고
- **단순한 LLM 호출을 넘어** 실제 비즈니스 가치를 창출하는 LLM (Large Language Model) 애플리케이션 개발에 필수적인 심층 지식을 제공합니다.
- LangChain은 LLM을 **다양한 외부 데이터 소스 (데이터베이스, API 등)** 및 **정교한 연산 능력 (함수 호출, 다른 LLM 연계 등)**과 유기적으로 결합하여 이전에는 상상하기 어려웠던 **지능적이고 맥락 인식적인** 애플리케이션을 구축할 수 있도록 혁신적으로 지원하는 **모듈화되고 확장 가능한** 오픈 소스 프레임워크입니다.

### 2.2 **1단계: LangChain의 핵심 가치와 혁신적 필요성**

* **LangChain이란?** 
    - 단순한 LLM 래퍼 (wrapper)를 넘어, **복잡한 멀티 스텝 워크플로우**를 효율적으로 설계, 구축 및 관리할 수 있도록 설계된 **포괄적인 생태계**입니다. 
    - LLM을 **단순 질의응답뿐만 아니라, 정보 검색, 의사 결정, 자동화된 액션 수행** 등 다양한 목표 달성에 필요한 외부 데이터, 연산, 그리고 **다른 지능형 에이전트**와 **유기적으로 연결**하여 사용합니다.
* **LangChain의 핵심 장점:**
    * **LLM 활용 패러다임 전환 가속화:** 
        - LLM을 단순 텍스트 생성 도구가 아닌, **지능적인 시스템의 핵심 엔진**으로 활용하는 차세대 프로그래밍으로의 혁신적인 전환을 용이
    * **복잡한 LLM 기반 파이프라인 구축 단순화:** 
        - **모듈화된 컴포넌트와 직관적인 인터페이스**를 제공하여, 데이터 로딩부터 정보 검색, LLM 추론, 최종 출력 생성까지 이어지는 복잡한 LLM 기반 파이프라인 개발 과정을 **놀라울 정도로 간소화**
    * **정교한 에이전트 상호작용 및 협업 지원:**
        -  특히, LangGraph를 통해 **여러 에이전트 간의 복잡하고 동적인 상호작용**을 시각적으로 정의하고 관리할 수 있도록 지원
        -  더욱 **정교하고 자율적인** 시스템 구축이 가능합니다.
        - 이는 인간-에이전트 협업뿐만 아니라, **자율적인 멀티 에이전트 시스템** 개발의 핵심 기반
* **주요 LLM 애플리케이션 유형 심층 분석:**
    * **지능형 챗봇:** 
        - 단순 텍스트 기반 대화를 넘어, **맥락을 이해하고 사용자 의도에 따라 다양한 외부 정보에 접근**하여 맞춤형 답변 및 액션을 제공하는 차세대 챗봇
    * **RAG (Retrieval Augmented Generation) 기반 지식 Q&A 시스템:** 
        - **최신 정보 및 특정 도메인 지식**을 LLM에 실시간으로 주입하여, LLM의 **환각 현상을 줄이고 답변의 정확성과 신뢰성을 획기적으로 향상**시키는 핵심 기술
        -  단순 검색 결과를 보여주는 것을 넘어, **검색된 정보를 바탕으로 논리적인 추론과 답변 생성**이 가능
    * **자율 에이전트 시스템 (다중 에이전트, 인간-에이전트 협업):** 
        - LLM을 기반으로 **스스로 목표를 설정하고, 필요한 도구를 선택하며, 복잡한 작업을 자율적으로 수행**하는 지능형 에이전트 시스템입니다.
        -  여러 에이전트가 **협력하여 하나의 목표를 달성**하거나, 인간과 에이전트가 **상호 작용하며 공동 작업을 수행**하는 등 다양한 시나리오를 지

### 2.3 **2단계: 효율적인 개발 환경 구축 및 LangChain 기본 작동 방식 이해**

* **안전한 API 키 관리:** 
    - OpenAI, Anthropic 등 다양한 LLM 서비스 제공업체의 API 키를 안전하게 확보하고,
    - **`.env` 파일과 같은 환경 변수 관리 도구**를 사용하여 코드와 분리 관리는 **보안 및 유지보수의 기본**
* **필수 라이브러리 설치 및 관리:** 
    - LangChain 프레임워크의 핵심 라이브러리 (`langchain`)뿐만 아니라, 특정 LLM 서비스와의 연동 (`langchain-openai`, `langchain-anthropic`) 
    - 다양한 유틸리티 및 통합 기능을 제공하는 LangChain Community (`langchain-community`) 등 필요한 라이브러리를 **pip와 같은 패키지 관리 도구**를 사용하여 효율적으로 설치하고 관리
* **LLM 연결 및 기본적인 프롬프트 엔지니어링:** 
    - LangChain을 사용하여 특정 LLM 서비스에 **간편하게 연결**하고, 
    - LLM의 능력을 최대한으로 활용하기 위한 **기본적인 프롬프트 작성 기법 (프롬프트 템플릿 활용 등)**을 익혀 LLM을 호출하고 결과를 확인


```bash
# 1. 설치
pip install langchain openai python-dotenv
```

```bash
# 2. .env 파일에 API 키 저장 (OpenAI 기준)
OPENAI_API_KEY=sk-...
```

```python
# 3. 간단한 LLM 호출 예제
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

response = llm.predict("고양이와 강아지의 차이점을 간단히 설명해줘.")
print(response)
```

### 2.4 **3단계: LangChain의 핵심 구성 요소 심층 이해: 체인, 프롬프트, 로더**

* **체인 (Chain):** 
    - 단순히 LLM을 호출하는 것 넘어, **하나의 논리적인 작업 단위를 구성하는 상호 연결된 컴포넌트의 순서**
    - 각 컴포넌트는 **특정 역할**을 수행하며, 이전 컴포넌트의 출력을 다음 컴포넌트의 입력으로 전달하여 **복잡한 데이터 처리 흐름**을 구축
* **핵심 컴포넌트:** 
    - 체인을 구성하는 기본적인 building block으로, 
        - **프롬프트 (LLM에게 지시), LLM (실질적인 텍스트 생성), 출력 파서 (LLM 응답 구조화), 다양한 도구 (외부 API 호출 등), 사용자 정의 함수** 등
    - LangChain에서는 이러한 다양한 컴포넌트들이 **"runnable"**이라는 **통일된 인터페이스**를 통해 추상화되어, **유연하고 일관된 방식으로 연결 및 실행**
* **프롬프트 템플릿 (Prompt Template):** 
    - LLM에게 제공할 **명확하고 효과적인 지침 세트 (프롬프트)**를 **재사용 가능하고 동적으로 생성**할 수 있도록 미리 정의해 둔 **일종의 설계도**
    - 사용자 입력, 외부 데이터 등을 **변수 형태로 삽입**, 상황에 맞는 프롬프트를 생성할 수 있도록 지원
* **문서 로더 (Document Loader):** 
    - **다양한 형식 (텍스트 파일, PDF, 웹 페이지, 데이터베이스 등)**의 데이터를 LangChain이 이해하고 처리할 수 있는 **표준적인 "문서 (Document)" 형태**로 효율적으로 로드하는 역할
    - 각 문서 객체는 **페이지 내용 (page_content)**과 **메타데이터 (metadata)**를 포함.
* **간단한 체인 구축 실습:** 
    - 프롬프트 템플릿을 사용하여 **질의 응답 템플릿**을 생성, LLM과 연결해 **기본적인 질의 응답 체인**을 구축
    - 텍스트 파일 로더를 사용하여 데이터를 로드하고, 로드된 데이터를 LLM에 전달하는 간단한 체인을 만들어 실행하는 과정을 통해 LangChain의 기본적인 작동 방식을 이해

```bash
pip install langchain-core
```

```python
# 프롬프트 템플릿과 LLM 체인
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

template = "너는 여행 전문가야. '{location}'에 가면 추천할 만한 활동은 뭐야?"
prompt = PromptTemplate.from_template(template)

chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run({"location": "제주도"})
print(response)

# 문서 로더 예시는 5단계에서 텍스트와 함께 실습합니다.
```

### 2.5 **4단계: LangChain Expression Language (LCEL) 및 Runnable 프로토콜 심층 분석**

* **LCEL (LangChain Expression Language):** 
    - LangChain의 **핵심적인 추상화 계층**으로, 
    - 기본적인 Runnable 컴포넌트들을 **마치 함수를 연결하듯이 선언적이고 간결한 방식**으로 연결하여 **복잡한 체인을 직관적으로 구축**
    - 파이프라인을 시각적으로 표현하고 이해하는 데 도움을 줍니다.
* **Runnable 프로토콜:** 
    - LangChain의 **모든 컴포넌트가 준수하는 통일된 인터페이스**입니다. 
    - 이를 통해 프롬프트, LLM, 함수, 도구 등 다양한 유형의 컴포넌트들을 **일관된 방식으로 처리하고 연결**
    - Runnable 인터페이스는 **호출 (`invoke`), 배치 처리 (`batch`), 스트리밍 (`stream`), 변환 (`transform`), 구성 (`pipe`)** 등 다양한 작업을 지원
* **핵심 Runnable 객체 상세 분석:**
    * **Runnable Sequence (`|` 연산자):** 
        - **여러 Runnable 컴포넌트를 순차적으로 연결**하여 데이터 처리 파이프라인을 구축
        - 이전 Runnable의 출력이 다음 Runnable의 입력으로 자동 전달됩니다.
    * **Runnable Lambda (파이썬 함수 래핑):** 
        - **기존의 파이썬 함수를 Runnable 컴포넌트로 손쉽게 변환**하여 LangChain 체인 내에서 활용.
        - 데이터 전처리, 후처리 등 다양한 사용자 정의 로직을 통합하는 데 유용합니다.
    * **Runnable Pass Through:** 
        - 입력을 **변경 없이 그대로 다음 단계로 전달**하거나, **추가적인 키-값 쌍을 출력에 병합**하는 역할
        -  체인 내에서 중간 결과를 유지하거나, 추가적인 컨텍스트 정보를 제공하는 데 활용
    * **Runnable Parallel ( `RunnableParallel`):** 
        - **여러 Runnable 컴포넌트를 동시에 병렬로 실행**하여 전체 처리 시간을 단축
        - 각 분기의 결과를 **딕셔너리 형태로 병합**하여 다음 단계로 전달합니다.
* **실제 코드 예제를 통한 LCEL 및 Runnable 활용:** 
    - Runnable Lambda를 사용하여 간단한 데이터 변환 함수를 체인에 통합하는 방법,
    - Runnable Pass Through를 활용하여 중간 결과를 유지하는 방법, 
    - Runnable Parallel을 사용하여 여러 작업을 동시에 처리하고 결과를 결합하는 방법 

```python
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

# RunnableLambda: 입력에 '님' 붙이기
add_suffix = RunnableLambda(lambda x: x + "님")

# RunnablePassthrough: 입력 그대로 출력
passthrough = RunnablePassthrough()

# RunnableParallel: 두 개 동시에 실행
parallel = RunnableParallel({
    "원본": passthrough,
    "존칭": add_suffix
})

print(parallel.invoke("철수"))
```

출력 예시:
```python
{'원본': '철수', '존칭': '철수님'}
```

### 2.6 **5단계: 외부 지식 활용의 핵심: 텍스트 분할 및 효율적인 검색**

* **검색기 (Retriever):** 
    - 사용자 쿼리와 **의미적으로 관련된 문서를 효율적으로 찾아 반환**하는 추상화된 인터페이스입니다.
    - **벡터 스토어**와 같은 인덱싱 기술을 기반으로 작동하며, 단순히 키워드 매칭이 아닌 **의미론적 유사성**을 기반으로 검색
* **텍스트 분할기 (Text Splitter):** 
    - 대용량의 텍스트 데이터를 LLM의 컨텍스트 윈도우 한계 내에서 처리할 수 있도록 **의미론적 단위 (문장, 단락 등)를 유지하면서 작은 청크 (chunk)로 분할**하는 중요한 역할    
    - RecursiveTextSplitter (일반적인 텍스트), HTML Splitter, Markdown Splitter 등 **데이터 형식에 최적화된 다양한 분할 방식**을 제공
* **벡터 스토어 (Vector Store):** 
    - 분할된 텍스트 청크들을 **고차원 벡터 형태로 임베딩 (embedding)**하여 저장하고, **빠르고 효율적인 유사도 검색**을 지원하는 특수한 데이터베이스
    -  Redis, Pinecone, Chroma, FAISS 등 다양한 옵션을 제공
* **실습:** 
    - **Redis 벡터 스토어를 직접 구축**하고, 
    - 여러 텍스트 분할기 (예: RecursiveCharacterTextSplitter)를 사용, **샘플 데이터를 청크 단위로 분할** 
    - **벡터 임베딩 모델을 통해 벡터화하여 Redis 벡터 스토어에 저장**하는 과정을 실습
    - 구축된 Redis 기반 검색기를 활용하여 **실제 쿼리를 던져 관련 문서를 검색**

```bash
pip install chromadb langchain-community tiktoken
```

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

text = "LangChain은 LLM 애플리케이션을 쉽게 만들 수 있게 도와주는 프레임워크입니다. 다양한 구성 요소를 포함합니다."
splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=10)
docs = splitter.create_documents([text])

# 임베딩 및 벡터 저장소 구축
embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding=embedding)

# 검색기 만들기
retriever = vectorstore.as_retriever()
retrieved_docs = retriever.get_relevant_documents("LangChain은 무엇인가요?")
print(retrieved_docs[0].page_content)
```

### 2.7 **6단계: LLM 지식 확장의 핵심 기술: RAG (Retrieval Augmented Generation)**

* **RAG (Retrieval Augmented Generation)의 핵심 개념:** 
    - LLM이 가지고 있는 일반적인 지식 외에 **외부의 전문 지식이나 최신 정보를 실시간으로 검색하여 답변 생성 과정에 활용**
    - 답변의 **정확성, 관련성, 최신성**을 획기적으로 향상시키는 강력한 기술입니다.
    - LLM의 **생성 능력**과 외부 정보 **검색 능력**을 결합하여 **환각 현상을 줄이고 신뢰할 수 있는 답변**을 생성
* **RAG 애플리케이션 구축의 핵심 단계:**
    1.  **인덱싱 (Indexing):** 
        - 외부의 다양한 데이터 소스를 로드하고, 텍스트 분할기를 사용하여 의미 단위로 분할한 후, 벡터 임베딩 모델을 통해 벡터화하여 벡터 스토어에 저장하는 과정
        - **데이터 전처리 및 벡터 임베딩 전략**이 전체 시스템 성능에 큰 영향
    2.  **검색 및 생성 (Retrieval and Generation):** 
        - 사용자 쿼리가 들어오면, **검색기를 사용하여 벡터 스토어에서 쿼리와 의미적으로 가장 관련 있는 문서를 검색**하고, 
        - 검색된 문서를 **프롬프트에 컨텍스트로 포함**하여 LLM에게 전달합니다. 
        - LLM은 주어진 컨텍스트를 바탕으로 **사용자 쿼리에 대한 답변을 생성**합니다.
* **검색기의 중요성 심층 분석:** 
    - 단순히 많은 문서를 검색하는 것을 넘어, **실제 질문과 가장 관련성이 높고 유용한 정보를 정확하게 찾아내는 것**이 RAG 시스템의 핵심입니다. 
    - 검색 결과의 **정확도 (precision)**와 **완전성 (recall)**을 최적화하는 것은 챗봇의 **전환율 향상 및 사용자 만족도 증대**에 직접적인 영향을 미칩니다. 
    - 검색기를 단순히 추상화하는 것만으로는 부족하며, **데이터 특성과 쿼리 유형에 맞는 검색 전략 및 파인튜닝**이 중요
* **RAG 체인 구축 실습:** 
    - 이전 단계에서 구축한 Redis 기반 검색기를 활용하여 **질문 템플릿, 검색기, LLM을 연결하는 RAG 체인을 직접 구축**하고, 
    - 실제 질문을 통해 **외부 정보를 기반으로 답변을 생성**하는 과정을 실습합니다. 

```python
from langchain.chains import RetrievalQA

# RAG 체인 구성
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

response = rag_chain.run("LangChain은 무엇인가요?")
print(response)
```


### 2.8 **7단계: LLM의 지능적 행동 구현: 도구 (Tools) 및 에이전트 (Agents)**

* **도구 (Tool):** 
    - 에이전트, 체인, 또는 LLM이 **외부 세계와 상호작용하고 특정 작업을 수행**하기 위해 사용하는 **플러그인**
    - 웹 검색, 데이터베이스 쿼리, 계산기, 외부 API 호출 등 다양한 형태를 가짐.
* **도구 키트 (Toolkit):** **특정 목적이나 작업을 해결하기 위해 논리적으로 그룹화된 도구들의 모음**입니다. 
    - 예를 들어, 웹 검색 및 URL 추출 도구 키트, 데이터베이스 조작 도구 키트 등
* **도구 사용 방법:**
    1.  **직접 도구 실행:** 
        - 개발자가 필요에 따라 도구를 직접 호출하여 데이터를 가져오거나 특정 작업을 수행
    2.  **LLM에 도구 바인딩:** 
        - LangChain의 기능을 사용하여 **LLM이 필요에 따라 도구를 선택하고 실행**할 수 있도록 LLM과 도구를 연결
        -  이를 통해 LLM은 외부 정보를 활용하거나 특정 기능을 수행하는 능력을 보유
* **에이전트 (Agent):** 
    - LLM을 **추론 엔진**으로 사용하여 **주어진 목표를 달성하기 위해 어떤 도구를 사용할지, 언제 사용할지, 그리고 어떤 순서로 사용할지를 스스로 결정하고 실행**하는 지능적인 시스템
    - 미리 정의된 작업 흐름을 따르는 체인과 달리, 에이전트는 **동적으로 행동 계획을 수립**
* **에이전트와 체인의 근본적인 차이점:** 
    - 체인은 **작업 시퀀스가 개발자에 의해 하드코딩**되어 있는 
    - 에이전트는 **LLM의 추론 능력**을 활용하여 **현재 상황과 목표에 따라 다음에 수행할 작업을 스스로 결정**
    - 이는 에이전트에게 훨씬 더 큰 **유연성과 자율성**을 부여합니다.

```bash
pip install duckduckgo-search
```

```python
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchRun

# 검색 도구 등록
search = DuckDuckGoSearchRun()
tools = [
    Tool(name="Web Search", func=search.run, description="웹에서 정보를 검색함")
]

# 에이전트 초기화
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 실행
agent.run("2024년 파리 올림픽 일정 알려줘")
```

### 2.9 **실전 예제: YouTube 요약 에이전트 구축:** 
- **YouTube 검색 도구**를 사용하여 특정 키워드로 YouTube 동영상을 검색하고, 
- **텍스트 변환 도구** (예: Transcriber)를 사용하여 동영상 내용을 텍스트로 추출한 후,
- 이를 LLM에 전달하여 **YouTube 채널의 주요 주제를 요약**하는 에이전트를 직접 만들어봅니다. 

**1. 준비 사항**

```bash
pip install langchain openai youtube-search-python youtube-transcript-api
```

```bash
# `.env` 파일에 OpenAI API 키 저장:
OPENAI_API_KEY=sk-xxxx
```

**2. YouTube 검색 도구 구현**

```python
from youtubesearchpython import VideosSearch

def search_youtube_videos(keyword, max_results=1):
    results = VideosSearch(keyword, limit=max_results).result()
    video_urls = [video['link'] for video in results['result']]
    return video_urls
```

**3. 자막 텍스트 추출 도구 구현**

```python
from youtube_transcript_api import YouTubeTranscriptApi
import re

def get_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

def extract_transcript(video_url):
    video_id = get_video_id(video_url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([t["text"] for t in transcript])
    return text
```

**4. LangChain 에이전트 설정 및 요약 실행**

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")

# 도구 1: YouTube 검색
youtube_search_tool = Tool(
    name="YouTube Search",
    func=lambda q: search_youtube_videos(q)[0],  # 첫 번째 영상 URL 반환
    description="유튜브에서 동영상 검색"
)

# 도구 2: 자막 추출기
transcript_tool = Tool(
    name="Transcript Extractor",
    func=extract_transcript,
    description="유튜브 URL에서 자막 텍스트 추출"
)

# 에이전트 정의
tools = [youtube_search_tool, transcript_tool]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 실행 예시: 키워드를 기반으로 동영상 요약 생성
keyword = "ChatGPT 사용법"
video_url = search_youtube_videos(keyword)[0]
transcript = extract_transcript(video_url)

summary_prompt = f"""
다음은 유튜브 영상의 전체 스크립트입니다:

{text if len(transcript) < 3000 else transcript[:3000]} 

이 영상의 핵심 내용을 간략하게 요약해 주세요.
"""

summary = llm.predict(summary_prompt)
print("🔎 요약 결과:")
print(summary)
```

**5. 최종 결과 예시**

```
🔎 요약 결과:
이 영상은 ChatGPT를 사용하는 기본적인 방법을 설명합니다. 사용자는 프롬프트를 입력하여 정보를 얻거나 콘텐츠를 생성할 수 있으며, 다양한 팁과 예시를 통해 효율적으로 활용하는 방법을 배울 수 있습니다...
```

### 2.10 **추가 정보**

* **LangSmith:** 
    - LLM 애플리케이션의 **실험, 디버깅, 모니터링, 평가**를 위한 통합 플랫폼으로, 
    - 개발 워크플로우를 혁신적으로 개선합니다. (별도 비디오에서 상세히 다룰 예정입니다.)
* **LangServe:** 
    - 개발된 LangChain 체인 및 에이전트를 **REST API 형태로 쉽게 배포**하여 
    - 다른 애플리케이션과 통합할 수 있도록 지원하는 도구입니다. (별도 비디오에서 상세히 다룰 예정)

---

### 2.10 학습 퀴즈 
1. LangChain의 주요 목표는 무엇이며, 개발자가 LLM 애플리케이션을 구축하는 데 어떻게 도움을 주나요?
> LangChain의 주요 목표는 LLM(Large Language Model)을 외부 데이터 및 계산 소스와 결합하여 더욱 강력하고 다양한 애플리케이션을 개발할 수 있도록 지원하는 것입니다. LangChain은 프롬프트 관리, 체인 구성, 데이터 로딩, 도구 사용, 에이전트 구축 등 LLM 애플리케이션 개발에 필요한 다양한 모듈과 추상화를 제공하여 개발 과정을 단순화합니다.

2. RAG (Retrieval-Augmented Generation) 애플리케이션의 기본적인 작동 방식을 간략하게 설명하고, LangChain이 이 과정에서 어떤 역할을 하는지 언급하세요.
> RAG 애플리케이션은 LLM이 답변을 생성할 때 외부 데이터 소스를 검색하여 활용하는 방식입니다. 사용자의 질문에 따라 관련된 문서를 벡터 스토어에서 검색하고, 검색된 컨텍스트와 질문을 함께 LLM에 제공하여 LLM이 더욱 정확하고 풍부한 답변을 생성할 수 있도록 돕습니다. LangChain은 데이터 로딩, 분할, 벡터화, 검색 및 LLM과의 통합을 위한 다양한 도구를 제공하여 RAG 애플리케이션 구축을 용이하게 합니다.

3. LangChain에서 "체인(Chain)"이란 무엇이며, 일반적인 구성 요소에는 어떤 것들이 포함되나요?
> LangChain에서 "체인"은 일련의 연결된 컴포넌트(예: 프롬프트, LLM, 출력 파서, 도구)로, 사용자 쿼리를 처리하여 가치 있는 출력을 생성하는 파이프라인입니다. 일반적인 구성 요소로는 사용자 입력을 LLM이 이해할 수 있는 형식으로 변환하는 프롬프트 템플릿, 실제 언어 모델인 LLM 또는 Chat 모델, LLM의 출력을 원하는 형식으로 변환하는 출력 파서, 외부 정보나 기능을 활용하는 도구 등이 있습니다.

4. LangChain Expression Language (LCEL)의 주요 특징과 이점을 설명하고, 파이프 연산자(|)가 LCEL에서 어떤 역할을 하는지 간략히 언급하세요.
> LCEL은 LangChain에서 복잡한 체인을 기본 컴포넌트로부터 쉽고 선언적으로 구축할 수 있도록 하는 언어입니다. 주요 특징은 파이프 연산자를 사용하여 Runnable 객체들을 순차적으로 연결하여 데이터 흐름을 정의할 수 있다는 점이며, 이를 통해 코드를 간결하고 가독성 높게 유지할 수 있습니다. 파이프 연산자는 이전 컴포넌트의 출력을 다음 컴포넌트의 입력으로 전달하는 역할을 합니다.

5. LangChain에서 "Runnable" 인터페이스의 개념을 설명하고, RunnableSequence와 RunnableLambda의 주요 차이점을 간략하게 설명하세요.
> Runnable 인터페이스는 LangChain에서 호출(invoke), 배치(batch), 스트리밍(stream), 변환(transform), 구성(compose)될 수 있는 작업 단위를 정의하는 개념입니다. RunnableSequence는 여러 개의 Runnable 컴포넌트를 순차적으로 연결하여 하나의 실행 가능한 파이프라인을 만드는 클래스인 반면, RunnableLambda는 파이썬의 호출 가능한 객체(예: 함수)를 Runnable 컴포넌트로 래핑하여 LangChain 체인 내에서 임의의 파이썬 코드를 실행할 수 있도록 합니다.

6. LangChain에서 텍스트 분할기(Text Splitter)의 역할은 무엇이며, 텍스트를 작은 조각으로 나누는 이유는 무엇인가요?
> LangChain에서 텍스트 분할기의 역할은 긴 텍스트 문서를 LLM의 컨텍스트 창 크기에 맞게 더 작고 의미 있는 청크(chunk)로 나누는 것입니다. 텍스트를 작은 조각으로 나누는 이유는 LLM이 처리할 수 있는 입력 텍스트의 길이에 제한이 있기 때문이며, 또한 관련성 높은 정보를 더 효과적으로 검색하고 LLM에 제공하기 위함입니다.

7. LangChain에서 "Retriever"와 "Vector Store"의 관계를 설명하고, Retriever가 벡터 스토어와 어떻게 상호작용하여 관련 문서를 검색하는지 간략히 언급하세요.
> LangChain에서 "Retriever"는 사용자 쿼리에 따라 관련 문서를 반환하는 인터페이스이며, "Vector Store"는 텍스트 데이터를 벡터 임베딩 형태로 저장하고 유사성 검색을 효율적으로 수행할 수 있도록 하는 데이터베이스입니다. Retriever는 일반적으로 Vector Store를 기반으로 작동하며, 사용자 쿼리를 벡터화하여 Vector Store에 유사성 검색을 요청하고, 가장 유사한 문서를 검색하여 반환합니다.

8. LangChain에서 "Tool"의 개념을 설명하고, LLM 기반 에이전트가 Tool을 사용하는 주요 이유는 무엇인가요?
> LangChain에서 "Tool"은 LLM 에이전트나 체인이 외부 세계와 상호작용하거나 특정 작업을 수행하기 위해 사용할 수 있는 인터페이스입니다. Tool은 검색 엔진, 계산기, 데이터베이스 쿼리 도구 등 다양한 형태를 가질 수 있으며, LLM 기반 에이전트가 Tool을 사용하는 주요 이유는 LLM 자체의 지식이나 능력만으로는 해결하기 어렵거나 불가능한 작업을 수행하고 외부 정보에 접근할 수 있도록 확장하기 위함입니다.

9. LangChain에서 "Agent"와 "Chain"의 주요 차이점을 설명하고, Agent가 작업을 수행하기 위한 액션 순서를 어떻게 결정하는지 간략히 언급하세요.
> LangChain에서 "Agent"는 LLM을 사용하여 수행할 액션의 순서를 결정하고, 필요한 경우 Tool을 호출하여 작업을 완료하는 프로그램입니다. 반면, "Chain"은 액션의 순서가 미리 정의되어 있는 고정된 파이프라인입니다. Agent는 LLM의 추론 능력을 활용하여 현재 상황에 가장 적합한 액션을 동적으로 선택하고 실행하는 반면, Chain은 항상 동일한 순서로 단계를 거칩니다.

10. LangGraph의 주요 목표와 LangChain 생태계 내에서 어떤 종류의 애플리케이션을 구축하는 데 사용될 수 있는지 간략하게 설명하세요.
> LangGraph의 주요 목표는 LangChain에서 멀티 에이전트 시스템과 에이전트-인간 상호작용과 같은 복잡한 상호작용 패턴을 정의하고 실행할 수 있도록 하는 것입니다. LangGraph는 에이전트 간의 통신, 조건부 실행, 루핑 등 고급 제어 흐름을 구축하는 데 사용될 수 있으며, 분산된 작업 처리, 전문화된 에이전트 협업 등 다양한 시나리오에 적용될 수 있습니다.

### 2.11 논술형 문제 
1. LangChain 프레임워크의 주요 구성 요소(예: Models, Prompts, Chains, Data Loaders, Indexes, Memory, Agents, Callbacks)를 설명하고, 각 구성 요소가 LLM 애플리케이션 개발에서 어떤 중요한 역할을 하는지 구체적인 예시와 함께 논하세요.
> LangChain 프레임워크는 LLM 애플리케이션 개발을 위한 다양한 구성 요소(Models, Prompts, Chains, Data Loaders, Indexes, Memory, Agents, Callbacks)를 제공한다. **Models**은 LLM과의 상호작용을 관리하며, 예를 들어 GPT-4를 호출해 텍스트 생성을 수행한다. **Prompts**는 모델 입력을 구조화해 특정 작업(예: 요약, 번역)을 최적화하고, **Chains**는 여러 작업 단계를 연결해 복잡한 워크플로우(예: 텍스트 분류 후 요약)를 구현한다. **Data Loaders**, **Indexes**, **Memory**, **Agents**, **Callbacks**는 각각 외부 데이터 로딩, 검색 최적화, 대화 기록 유지, 자율적 작업 수행, 실행 모니터링을 지원해, 예를 들어 고객 지원 챗봇이 과거 대화를 참조하며 실시간으로 응답하도록 돕는다.

2. RAG (Retrieval-Augmented Generation) 파이프라인을 구축하는 주요 단계(데이터 로딩 및 전처리, 텍스트 분할, 벡터 임베딩 생성 및 저장, 검색, 생성)를 상세히 설명하고, 각 단계에서 LangChain이 제공하는 기능과 이점을 논하세요. 또한, RAG 시스템의 성능을 향상시키기 위한 잠재적인 전략들을 제시하고 설명하세요.
> RAG 파이프라인은 **데이터 로딩 및 전처리**(예: PDF 문서 로딩 후 정제), **텍스트 분할**(긴 문서를 작은 청크로 분할), **벡터 임베딩 생성 및 저장**(텍스트를 벡터로 변환해 Pinecone에 저장), **검색**(질문과 관련된 문서 검색), **생성**(검색된 문서를 바탕으로 답변 생성)의 단계로 구성된다. LangChain은 `DocumentLoader`, `TextSplitter`, `VectorStore`, `Retriever`, `ChatModel`과 같은 도구를 제공해 각 단계를 간소화하며, 예를 들어 `FAISS`를 사용한 빠른 벡터 검색을 지원한다. 성능 향상 전략으로는 고품질 임베딩 모델 사용, 검색 결과 필터링 강화, 프롬프트 최적화가 있으며, 이는 답변 정확도와 속도를 개선한다.

3. LangChain Expression Language (LCEL)을 사용하여 복잡한 LLM 애플리케이션 워크플로우를 구축하는 방법을 설명하고, LCEL의 주요 이점(예: 스트리밍, 비동기 지원, 로깅, 디버깅 용이성)을 구체적인 코드 예시와 함께 논하세요. 또한, LCEL을 사용할 때 고려해야 할 잠재적인 어려움이나 제한 사항들을 제시하고 설명하세요.
> LCEL은 파이프라인 연산자(`|`)를 사용해 LLM 워크플로우를 구성하며, 예를 들어 `PromptTemplate | ChatModel | StrOutputParser`로 질문-답변 체인을 만든다. 주요 이점은 **스트리밍**으로 실시간 출력, **비동기 지원**으로 병렬 처리, **로깅**과 **디버깅**으로 오류 추적이 용이하다는 점이다. 하지만 복잡한 워크플로우에서 구성 요소 간 호환성 문제나 성능 최적화의 어려움이 발생할 수 있으며, 이를 해결하려면 명확한 문서화와 테스트가 필요하다.

4. LangChain에서 에이전트의 개념과 다양한 유형의 에이전트(예: Action Agent, Plan-and-Execute Agent)를 비교 분석하고, 에이전트가 Tool을 활용하여 복잡한 작업을 자율적으로 수행하는 과정을 구체적인 시나리오와 함께 설명하세요. 또한, 에이전트 시스템을 설계하고 배포할 때 고려해야 할 윤리적 및 안전성 문제들을 논하세요.
> LangChain의 에이전트는 **Action Agent**(단일 작업 즉시 실행)와 **Plan-and-Execute Agent**(계획 수립 후 작업 수행)로 나뉜다. 예를 들어, Action Agent는 검색 툴을 사용해 즉시 질문에 답하고, Plan-and-Execute Agent는 복잡한 여행 계획을 단계별로 처리한다. 에이전트는 웹 검색, 계산기 등 툴을 활용해 자율적으로 작업하며, 이를 통해 시장 분석 같은 작업을 수행한다. 하지만 잘못된 정보 생성, 편향된 출력, 오작동 가능성 같은 윤리적·안전성 문제를 고려해, 엄격한 출력 검증과 사용자 피드백 루프를 설계해야 한다.


5. LangChain 생태계 내에서 LangGraph의 역할과 중요성을 설명하고, LangGraph를 활용하여 멀티 에이전트 시스템 또는 인간-에이전트 협업 워크플로우를 구축하는 방법을 구체적인 예시와 함께 논하세요. 또한, LangGraph가 기존의 LangChain 기능과 어떻게 통합되고 확장되는지 분석하고, 향후 LangGraph의 발전 방향에 대한 서술하시오
> LangGraph는 LangChain 내에서 복잡한 멀티 에이전트 워크플로우를 그래프 기반으로 설계하는 도구로, 예를 들어 고객 지원 에이전트와 데이터 분석 에이전트가 협업해 사용자 요청을 처리한다. `langgraph` 모듈은 노드와 엣지를 정의해 워크플로우를 시각화하며, LangChain의 `Agent`와 `Tool`을 통합해 확장성을 높인다. 인간-에이전트 협업 시나리오에서는 사용자가 중간 피드백을 제공해 워크플로우를 조정할 수 있다. 향후 LangGraph는 더 직관적인 그래프 편집 UI와 강화된 병렬 처리 기능을 통해 복잡한 AI 시스템 개발을 더욱 가속화할 가능성이 크다.

### 2.12 용어집

1. LLM (Large Language Model): 
    - 방대한 양의 텍스트 데이터를 학습하여 인간과 유사한 텍스트를 이해하고 생성할 수 있는 인공 신경망 모델.

2. LangChain: 
    - LLM을 기반으로 하는 애플리케이션 개발을 위한 오픈 소스 프레임워크로, 다양한 구성 요소와 도구를 제공하여 LLM과의 상호작용을 단순화하고 확장합니다.

3. Chain: 
    - LangChain에서 사용자 쿼리를 처리하기 위해 연결된 일련의 컴포넌트(예: 프롬프트 템플릿, LLM, 출력 파서, 도구)를 의미하는 파이프라인.

4. Prompt Template: 
    - LLM에 제공될 프롬프트의 구조와 형식을 정의하는 템플릿으로, 변수를 사용하여 동적으로 프롬프트를 생성할 수 있습니다.

5. Output Parser: 
    - LLM의 응답을 특정 형식으로 추출하거나 변환하는 컴포넌트.

6. Runnable: 
    - LangChain에서 invoke, batch, stream, transform, compose 등의 작업을 수행할 수 있는 실행 가능한 작업 단위를 정의하는 인터페이스.

7. LCEL (LangChain Expression Language): 
    - Runnable 객체들을 파이프 연산자를 사용하여 선언적으로 연결하여 복잡한 체인을 구축할 수 있도록 하는 언어.

8. Data Loader: 
    - 다양한 소스(예: 텍스트 파일, 웹 페이지, 데이터베이스)로부터 데이터를 LangChain이 처리할 수 있는 형식(Document)으로 로드하는 컴포넌트.

9. Document: 
    - LangChain에서 텍스트 내용(page_content)과 관련 메타데이터(metadata)를 포함하는 데이터 구조.

10. Text Splitter: 
    - 긴 텍스트를 LLM의 컨텍스트 창 크기에 맞게 더 작고 의미 있는 청크(chunk)로 분할하는 컴포넌트.

11. Embedding: 
    - 텍스트나 다른 데이터를 의미론적 유사성을 반영하는 숫자 벡터로 변환하는 과정 또는 그 결과물.

12. Vector Store: 
    - 텍스트 임베딩을 효율적으로 저장하고 유사성 검색을 수행할 수 있도록 하는 특수 데이터베이스.

13. Retriever: 
    - 사용자 쿼리와 관련된 문서를 벡터 스토어 또는 다른 데이터 소스로부터 검색하는 인터페이스.

14. RAG (Retrieval-Augmented Generation): 
    - LLM이 답변을 생성할 때 외부 데이터 소스를 검색하여 활용하는 기술.

15. Tool: 
    - LLM 에이전트나 체인이 외부 세계와 상호작용하거나 특정 작업을 수행하기 위해 사용할 수 있는 기능 또는 인터페이스 (예: 검색, 계산).

16. Agent: 
    - LLM을 사용하여 수행할 액션의 순서를 결정하고, 필요한 경우 Tool을 호출하여 작업을 완료하는 자율적인 프로그램.

17. LangGraph: 
    - LangChain에서 멀티 에이전트 시스템 및 복잡한 상호작용 워크플로우를 구축하기 위한 프레임워크.