---
title: 25차시 8:IBM TECH(종합 내용)
layout: single
classes: wide
categories:
  - IBM TECH(종합 내용)
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

## 71. GenAI 기반 애플리케이션 개발
- 출처: [From Idea to AI: Building Applications with Generative AI](https://www.youtube.com/watch?v=dFSnam97YbQ)

### **70.1 배경**

*   Gartner에 따르면 2026년까지 기업의 80%가 모델 또는 API 형태로 생성형 AI를 활용할 것으로 예상됨  
    *   이는 기업들이 비즈니스 프로세스를 자동화하고, 고객 경험을 향상시키며, 새로운 수익원을 창출하기 위해 AI를 적극적으로 도입하려는 움직임을 반영함.  
*   개발자는 AI 기반 애플리케이션 구축 경험 부족에 대한 우려가 있을 수 있음  
    *   특히, 생성형 AI는 전통적인 소프트웨어 개발과는 다른 접근 방식과 기술 스택을 요구하기 때문에 초기 학습 곡선이 높을 수 있음.  

### **70.2 목표**

*   생성형 AI(GenAI) 기반 애플리케이션 개발 시작 방법 안내  
    *   초보자도 쉽게 따라할 수 있는 단계별 가이드를 통해 개발자가 첫 발을 내딛을 수 있도록 돕고자 함.  
*   AI 기반 애플리케이션 구축 및 실행 방법 제시  
    *   모델 선택부터 데이터 준비, 애플리케이션 통합까지 전체 과정을 명확하게 설명하여 실질적인 개발 경험을 제공.  
*   개발자가 애플리케이션 구축, 실행, 테스트 과정에서 활용할 수 있는 오픈 소스 도구 및 기술 소개  
    *   다양한 오픈 소스 생태계를 활용하여 비용 효율적이고 유연한 개발 환경을 조성하는 방법을 제안.  

### **70.3 GenAI 개발 여정 3단계**

*   **아이디어 구상 및 실험:**  
    *   특정 용도에 맞는 모델 선택  
        *   용도에 따라 모델의 크기, 성능, 비용 등을 고려해야 하며, 이를 위해 충분한 사전 조사와 비교 분석이 필요함.  
    *   Hugging Face 등 모델 리포지토리에서 모델 성능 및 크기 비교  
        *   Hugging Face와 같은 플랫폼은 다양한 모델의 성능 지표, 사용 사례, 커뮤니티 피드백을 제공하여 모델 선택을 지원함.  
    *   모델 성능 벤치마킹 도구 활용  
        *   벤치마킹 도구를 통해 모델의 정확도, 속도, 리소스 사용량 등을 평가하여 최적의 모델을 선정할 수 있음.  
    *   **모델 선택 고려 사항:**  
        *   자체 호스팅(self-hosting)은 클라우드 기반 서비스보다 저렴  
            *   자체 호스팅은 초기 설정 비용과 유지보수 노력이 필요하지만, 장기적으로 비용 절감 효과를 가져올 수 있음.  
        *   Small Language Model(SLM)은 Large Language Model(LLM)보다 짧은 지연 시간과 특정 작업에 특화된 성능 제공  
            *   SLM은 LLM에 비해 리소스 요구가 적고 응답 속도가 빠르며, 특정 도메인에서 더 나은 성능을 발휘할 수 있음.  
    *   **프롬프트 엔지니어링:**  
        *   Zero-shot 프롬프팅: 예시 없이 질문  
            *   모델이 기본적인 이해 능력을 바탕으로 답변을 생성하도록 요청하는 방식으로, 간단한 질문에 적합함.  
        *   Few-shot 프롬프팅: 원하는 응답 방식 예시 제공  
            *   몇 가지 예시를 제공하여 모델이 특정 형식이나 스타일로 답변하도록 유도하는 방법으로, 정교한 결과를 얻을 수 있음.  
        *   Chain-of-thought: 모델의 사고 과정 단계별 설명 요청  
            *   복잡한 문제 해결을 위해 모델이 중간 단계를 설명하면서 답을 도출하도록 유도하는 방식으로, 논리적 일관성을 강화함.  
    *   데이터 실험을 통해 모델의 잠재적 문제점 파악  
        *   다양한 데이터 세트를 활용한 실험을 통해 모델의 한계나 오류를 미리 발견하고 해결책을 마련할 수 있음.  

*   **애플리케이션 구축:**  
    *   로컬 환경에서 AI 모델을 실행하고 API 요청 가능  
        *   로컬 환경에서 테스트를 진행하면 초기 개발 단계에서 발생할 수 있는 문제를 신속히 파악하고 수정할 수 있음.  
    *   **데이터 활용 방법:**  
        *   Retrieval Augmented Generation(RAG): LLM에 관련 데이터를 추가하여 응답 정확도 향상  
            *   외부 데이터베이스나 문서를 참조하여 모델의 지식을 확장함으로써 최신 정보나 특정 도메인 지식을 반영할 수 있음.  
        *   Fine-tuning: LLM에 데이터를 통합하여 특정 스타일 및 직관 학습  
            *   특정 작업이나 스타일에 맞게 모델을 미세 조정하여 더 정확하고 유의미한 결과를 획득
    *   LangChain과 같은 도구 및 프레임워크 활용  
        *   LangChain은 AI 모델을 애플리케이션에 쉽게 통합할 수 있도록 설계된 프레임워크로, 개발자의 생산성을 크게 향상시킴.  
    *   복잡한 작업 분할 및 모델 호출 흐름 평가  
        *   하나의 큰 작업을 여러 작은 작업으로 분할하고, 각 작업에 적합한 모델을 호출하는 전략을 통해 효율성을 극대화할 수 있음.  

*   **개발 및 운영(MLOps):**  
    *   AI 기반 애플리케이션의 프로덕션 배포 및 확장  
        *   MLOps는 AI 모델의 지속적인 관리와 업데이트를 위한 체계적인 접근 방식을 제공함으로써 프로덕션 환경에서의 안정성을 보장함.  
    *   **인프라:** 컨테이너 및 Kubernetes와 같은 기술을 사용하여 효율적인 모델 배포 및 확장  
        *   컨테이너화된 모델은 다양한 환경에서 일관되게 실행될 수 있으며, Kubernetes를 통해 자동 확장 및 리소스 관리가 가능함.  
    *   **운영:** VLLM과 같은 프로덕션 환경 런타임 사용  
        *   VLLM은 대규모 언어 모델을 효율적으로 실행할 수 있는 런타임으로, 높은 처리량과 낮은 지연 시간을 제공함.  
    *   **모델 및 인프라:** 온프레미스와 클라우드 인프라 조합을 통한 하이브리드 접근 방식 고려  
        *   민감한 데이터를 처리하거나 규제 준수를 위해 온프레미스를 사용하고, 클라우드를 활용하여 확장성을 보완할 수 있음.  
    *   **모니터링:** 애플리케이션 예외 처리, 벤치마킹 및 모니터링 필요  
        *   모델의 성능 저하나 이상 징후를 실시간으로 감지하고, 필요한 경우 재훈련하거나 업데이트할 수 있는 체계가 필요함.  
    *   DevOps와 유사한 ML Ops를 통해 모델의 원활한 프로덕션 배포 보장  
        *   CI/CD 파이프라인을 확장하여 AI 모델의 지속적인 통합, 테스트, 배포를 자동화함으로써 운영 효율성을 높일 수 있음.  

### **70.4 결론**

*   AI는 또 다른 도구일 뿐이며, GenAI를 활용하여 아이디어 구상, 구축, 배포를 통해 실질적인 결과 창출 가능  
    *   AI는 인간의 창의성과 문제 해결 능력을 보완하는 강력한 도구로서, 적절한 전략과 접근 방식을 통해 혁신적인 제품과 서비스를 만들어낼 수 있음.  
    *   GenAI를 통해 기존 비즈니스 프로세스를 개선하거나 새로운 시장을 개척할 수 있는 무한한 가능성이 열려있음.  

## 72. 전자 제품 매장 운영 & LLM 최적화
- 출처: [Context Optimization vs LLM Optimization: Choosing the Right Approach](https://www.youtube.com/watch?v=pZjpNS9YeVA)


### **72.1 상황 설정: 전자 제품 매장 운영 & 직원 교육**

* 신규 전자 제품 매장 오픈, 직원 채용 - 다양한 경험과 배경을 가진 인력 구성으로 교육 필요성 증가
* 직원 응대 표준화 필요 (고객 응대, 제품 정보 제공 등) - 브랜드 이미지 확립과 고객 만족도 향상 위한 요소
* 새로운 기술 정보 습득 어려움 발생 - 빠르게 변화하는 전자제품 시장에서 최신 정보 유지의 어려움 직면

### **72.2 LLM 최적화 기본 개념**

* **Context Optimization (컨텍스트 최적화):** 
    - LLM이 텍스트 생성 시 고려하는 텍스트 범위 (창) 최적화 
    - 모델이 참조할 수 있는 정보의 양과 질을 향상시켜 더 정확하고 관련성 높은 응답 생성
* **Model Optimization (모델 최적화):** 
    - 특정 요구사항에 맞춰 LLM 모델 업데이트 
    - 모델 자체의 능력과 특성을 조정하여 특정 도메인이나 작업에 더 효과적으로 대응하도록 조정

### **72.3 직원 교육 & LLM 최적화 방법 비교**

| **상황**                 | **직원 교육**                                                                    | **LLM 최적화**                                                                                                         |
| :----------------------- | :----------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| **기본 응대 가이드라인**      | 인사, 정중한 태도, 질문 기반 상위 3개 옵션 제시, 프로모션 안내 등 - 기본적인 고객 응대 프로토콜과 상품 추천 방식 표준화                                                 | **Prompt Engineering (프롬프트 엔지니어링):** 명확한 지침 제공 (텍스트, 예시, Chain of Thought 등) - 특정 상황에 맞는 응답을 생성하도록 정교한 지시문 설계                                 |
| **새로운 정보 습득 어려움**   | 매뉴얼 제작 및 제공, 질문 기반 매뉴얼 관련 페이지 발췌 및 제공 - 체계적인 정보 관리와 접근 방법을 통한 학습 장벽 감소                                                      | **RAG (Retrieval Augmented Generation):** LLM을 데이터 소스에 연결, 정확한 답변 제공 (매뉴얼 = 데이터 소스) - 최신 제품 정보 데이터베이스와 연동하여 실시간 정확한 정보 제공                            |
| **직원 증가 및 전문성 필요** | 판매/기술 교육 실시 - 제품 지식과 판매 기술 향상을 위한 체계적인 교육 프로그램 운영                                                              | **Fine-tuning (파인 튜닝):** 모델 파라미터 업데이트, 특정 도메인 특화 (교육 = 데이터 학습) - 전자제품 분야에 특화된 지식과 응대 패턴을 모델에 학습시켜 전문성 강화                                        |

### **72.4 LLM 최적화 방법 요약**

* **Prompt Engineering (PE):** 
    - 모델에게 원하는 결과에 대한 명확한 지침 제공 
    - 정확한 질문 구조와 맥락 제공으로 효과적인 응답 유도
* **RAG:** 
    - LLM을 외부 데이터 소스에 연결하여 최신 정보 활용 및 답변 정확도 향상 (환각 현상 방지) 
    - 실시간으로 업데이트되는 제품 사양, 재고 상태, 가격 정보를 정확히 반영
* **Fine-tuning:** 
    - 모델 파라미터를 업데이트하여 특정 도메인에 특화된 모델 생성 및 행동 제어 
    - 전자제품 관련 전문 용어와 브랜드 톤앤매너에 최적화된 응답 패턴 학습

### **72.5 LLM 최적화 시 고려 사항**

* **Context Optimization (RAG, PE):** 
    - 모델이 사전에 알아야 할 정보를 전달, 추론 및 텍스트 생성 유도 
    - 고객 질문 유형별 적절한 정보 제공 범위와 깊이 설정
* **Model Optimization (Fine-tuning):** 
    - 모델을 최적화하여 원하는 행동 및 응답 유도 
    - 매장의 고유한 판매 철학과 고객 응대 방식을 반영한 모델 구축
* **실시간 데이터 접근 및 정확성 확보:** 
    - RAG를 통해 실시간 데이터 기반 답변 제공 및 정확성 확보 
    - 가격 변동, 프로모션, 재고 상태 등 변동성 높은 정보의 정확한 실시간 반영

### **72.6 LLM 최적화 전략**

* **Additive (상호 보완적):** 
    - PE, RAG, Fine-tuning은 상호 보완적으로 작용 
    - 각 기술의 장점을 결합하여 시너지 효과 창출
* **시작은 PE:** 
    - PE를 통해 LLM 솔루션 적합성 판단, baseline 모델 정확도 확인, Fine-tuning 데이터 확보 
    - 초기 투자 비용과 리소스 최소화로 빠른 검증 가능
* **정확도 우선:** 
    - Context Window 최적화보다 답변 정확도에 집중, 정확도 확보 후 최적화 전략 수립 
    - 잘못된 제품 정보 제공으로 인한 고객 신뢰 하락 방지
* **데이터 품질 우선:** 
    - Fine-tuning 시 데이터 양보다 품질에 집중 (100개 고품질 예시 활용 가능) 
    - 실제 고객 상호작용에서 수집된 우수 사례 중심의 학습 데이터 구성
* **성공 측정 및 Baseline 설정:** 
    - 정확도, 정밀도 등 객관적인 지표를 통해 성공 측정 및 Baseline 설정 
    - 고객 만족도 조사와 판매 전환율 등 비즈니스 KPI와 연계한 성과 측정

### **72.7 LLM 최적화 기술 비교**

| 특징                                      | Prompt Engineering (PE)                                          | RAG (Retrieval Augmented Generation)                                           | Fine-tuning                                                                 |
| :---------------------------------------- | :------------------------------------------------------------- | :----------------------------------------------------------------------------- | :-------------------------------------------------------------------------- |
| **공통점**                                  | 정확도 향상, 환각 현상 감소 - LLM의 신뢰성과 실용성 향상을 위한 핵심 접근법                                                                 |                                                                               |                                                                             |
| **핵심**                                   | 빠른 반복 작업, 솔루션 적합성 확인 - 적은 비용으로 다양한 응대 시나리오 테스트 가능                                                               | 외부 데이터 소스 연결, 가이드라인 제공 - 매장 내 제품 카탈로그, 재고 시스템과 실시간 연동                                                                  | 모델 행동 변경, 특정 도메인 전문화 - 전자제품 전문 용어와 판매 기법에 최적화된 응답 패턴 학습                                                         |
| **특징**                                   | 구현 용이성, 빠른 결과 확인 - 기술적 진입장벽이 낮고 즉각적인 효과 측정 가능                                                                | Context Window 최적화 (제약 존재) - 대용량 제품 매뉴얼과 기술 사양 정보 활용 가능                                                                 | 모델 추론 (PE: 지침 제공, Fine-tuning: 보장), 모델 행동 제어 - 브랜드 가치와 판매 철학에 맞는 일관된 응대 스타일 구현                                              |
| **비유**                                   | 명확한 업무 지시서 - 상황별 대응 방식과 권장 표현을 상세히 안내                                                                | 단기 기억 - 필요한 정보를 즉시 참조하여 활용하는 능력                                                                        | 장기 기억 - 반복적인 학습을 통해 체화된 지식과 행동 패턴                                                                     |

### **72.8 최종 요약**

* Context Optimization은 쉽고 빠른 LLM 최적화 방법 
    - 적은 초기 투자로 빠른 성과 달성 가능
* Fine-tuning은 사용자가 증가하고 Latency 문제가 발생할 경우 고려, 모델 전문화 및 행동 제어 가능 
    - 대규모 체인점이나 높은 수준의 전문성이 요구되는 프리미엄 매장에 적합
* PE와 RAG를 먼저 활용하고, 필요에 따라 Fine-tuning 적용 
    - 단계적 접근을 통한 비용 효율적인 최적화 전략 구현

### **72.9 결론**

- LLM 최적화는 고객 경험 향상 및 비즈니스 성장을 위한 필수 전략입니다. 
- Prompt Engineering, RAG, Fine-tuning을 적절히 활용하여 목표를 달성하십시오. 
- Context Optimization과 Model Optimization의 균형을 맞추는 것이 중요합니다. 
- 매장 규모, 제품 복잡성, 고객층 특성에 따라 최적의 조합을 찾아 적용하는 것이 성공의 열쇠입니다.

## 73. AI 추론(Inferencing)
- 출처: [AI Inference: The Secret to AI's Superpowers](https://www.youtube.com/watch?v=XtT5i0ZeHHE)


### **73.1 추론이란 무엇인가?**

*   AI 모델이 훈련을 통해 학습한 정보를 바탕으로 예측하거나 작업을 수행하는 **실질적인 활용 단계**  
    - 즉, 추론은 AI가 학습한 지식을 실세계 문제에 적용하는 과정으로, 단순히 이론에 머무르지 않고 구체적인 결과를 만들어내는 단계입니다. 예를 들어, 이미지 인식 모델이 고양이 사진을 보고 "고양이"라고 판단하는 것이 추론입니다.
*   **비용 효율성과 속도**가 중요  
    - 추론은 실시간으로 이루어져야 하므로 빠른 처리 속도가 필수이며, 동시에 대규모 사용자 요청을 처리하기 위해 자원을 효율적으로 사용하는 것이 핵심 과제입니다.

### **73.2 AI 모델의 두 가지 주요 단계**

*   **훈련(Training) 단계**: 모델이 학습하는 단계 (데이터 간의 관계 파악 및 모델 가중치(weight)에 저장)  
    - 이 단계에서는 모델이 대량의 데이터를 분석하며 패턴을 찾아내고, 이를 기반으로 가중치라는 수치 데이터를 업데이트합니다. 예를 들어, 강아지와 고양이를 구분하는 모델이라면 털 색깔, 귀 모양 등의 특징을 학습합니다.
*   **추론(Inferencing) 단계**: 모델이 실시간 데이터에 대해 예측/판단하는 단계 (학습한 내용을 실제 적용)  
    - 훈련이 끝난 모델이 새로운 데이터를 입력받아 학습 결과를 바탕으로 결론을 내리는 과정입니다. 예를 들어, 사용자가 업로드한 사진을 보고 "강아지"인지 "고양이"인지 판단합니다.

### **73.3 추론 과정**

1.  **실시간 데이터 입력**: 사용자의 쿼리 또는 새로운 데이터 입력  
    - 예를 들어, 사용자가 "오늘 날씨 어때?"라고 질문하거나, 새로운 이메일이 시스템에 도착하는 경우가 이에 해당합니다.
2.  **비교 및 일반화**: 모델이 훈련 과정에서 처리된 정보 및 저장된 가중치를 기반으로 입력 데이터와 비교/분석  
    - 모델은 과거 학습 데이터를 기준으로 새로운 데이터를 해석합니다. 이때 단순히 기억하는 것이 아니라, 일반화된 패턴을 통해 판단. 예: 이메일에 "무료"라는 단어가 많다면 스팸일 가능성을 높게 봅니다.
3.  **결과 도출**: 학습된 내용을 바탕으로 새로운 데이터 해석 및 **실행 가능한 결과(예: 스팸 메일 분류)** 생성  
    - 최종적으로 모델은 예측 결과를 내놓고, 이를 실제 행동으로 연결합니다. 스팸 메일이면 삭제하거나 분류하는 식입니다.

### **73.4 추론 예시: 스팸 메일 탐지 모델**

*   **훈련**: 스팸/정상 메일로 레이블링된 대량의 데이터를 모델에 학습 (특정 키워드, 발신 주소, 과도한 느낌표 사용 등 스팸 메일의 특징 학습)  
    - 수십만 개의 이메일을 분석하며 "광고성 단어(예: 할인, 무료)"나 "의심스러운 발신자(예: 알 수 없는 도메인)" 같은 특징을 학습합니다.
*   **추론**: 새로운 메일이 수신되면, 모델이 학습한 스팸 패턴과 비교하여 스팸 여부를 예측하고 확률 점수(probability score)로 결과를 제공  
    - 예를 들어, 새 메일에 "당첨"이라는 단어가 반복되고 발신자가 불명확하면 모델은 이를 스팸으로 간주할 가능성을 계산합니다.
*   **실행 가능한 결과**:  
    *   90% 스팸 확률 → 스팸 메일함으로 이동  
        - 확률이 높으니 사용자가 보지 않도록 자동으로 분류합니다.  
    *   50% 스팸 확률 → 사용자에게 판단을 맡기도록 메일함에 표시  
        - 확률이 모호할 경우, 사용자가 직접 확인할 수 있도록 중립적으로 처리합니다.

### **73.5. 추론 비용**

*   모델 훈련 비용보다 추론 비용이 훨씬 더 높음 (모델 생애 주기 동안 추론에 90% 정도 소요)  
    - 훈련은 한 번에 끝나지만, 추론은 모델이 배포된 후 매일 수많은 요청을 처리하며 계속 자원을 소모.
*   추론 비용 발생 요인:  
    *   **규모**: 훈련은 한 번이지만, 추론은 수백만/수십억 번 발생  
        - 예를 들어, 구글 검색은 하루에도 수십억 번의 추론을 처리합니다.  
    *   **속도**: 실시간 데이터 처리 및 빠른 응답 필요 (GPU 등 고성능 하드웨어 필요)  
        - 사용자가 기다리지 않도록 0.1초 안에 결과를 내야 하므로 강력한 하드웨어가 필수입니다.  
    *   **모델 복잡성**: 복잡한 작업을 처리하기 위해 모델 크기가 커지면서 더 많은 계산 자원 필요  
        - 예: 단순 텍스트 분류보다 이미지나 음성을 분석하는 모델은 훨씬 더 많은 연산이 필요합니다.  
    *   **인프라**: 데이터 센터 유지/관리, 네트워크 연결, 에너지 소비 등 지속적인 비용 발생  
        - 서버를 24시간 가동하고 냉각 시스템을 유지하는 데도 큰 비용이 듭니다.

### **73.6 추론 속도 향상 방법 (The Stack)**

*   **하드웨어(Hardware)**: AI 연산에 최적화된 **AI 가속기** (CPU, GPU보다 효율적) 개발  
    - 예: 구글의 TPU나 엔비디아의 A100 같은 장치는 일반 CPU보다 AI 작업을 훨씬 빠르게 처리.
*   **소프트웨어(Software)**: 모델 압축 기술 활용  
    *   **가지치기(Pruning)**: 모델 정확도에 큰 영향 없이 불필요한 가중치를 제거하여 모델 크기 축소  
        - 마치 나무 가지를 쳐내듯, 모델에서 중요하지 않은 부분을 잘라내 가볍게 만듭니다.  
    *   **양자화(Quantization)**: 모델 가중치 정밀도를 낮춰 계산 속도 향상 및 메모리 요구량 감소 (예: 32비트 부동 소수점 → 8비트 정수)  
        - 숫자를 덜 정밀하게 표현해도 결과가 비슷하다면, 계산량을 줄여 속도를 높입니다.  
*   **미들웨어(Middleware)**: 하드웨어와 소프트웨어 연결, 추론 가속화 기능 제공  
    *   **그래프 융합(Graph Fusion)**: 통신 그래프 내 노드 수를 줄여 CPU와 GPU 간 통신 횟수 최소화  
        - 작업 단계를 통합해 중간 과정에서의 데이터 이동을 줄입니다.  
    *   **병렬 텐서(Parallel Tensors)**: 모델의 계산 그래프를 여러 조각으로 분할하여 여러 GPU에서 동시에 실행 (대용량 모델 처리 가능)  
        - 대규모 모델을 쪼개 여러 장치에서 동시에 돌리면 처리 시간이 단축됩니다.

## 74. AI 어시스턴트 vs. AI 에이전트
- 출처: [AI Agents and AI Assistants: A Contrast in Function](https://www.youtube.com/watch?v=IivxYYkJ2DI)


### **74.1 AI 어시스턴트**

*   **특징:**
    *   사용자의 명령(프롬프트)에 **반응적**으로 작동  
        - 명확한 지시가 있어야만 작동하며, 능동적인 행동은 하지 않음
    *   자연어 이해 능력 활용  
        - 사용자의 언어를 해석하여 적절한 작업을 수행
    *   잘 정의된 명령 필요  
        - 애매한 요청에는 오작동하거나 결과가 만족스럽지 않을 수 있음
    *   프롬프트 튜닝, 미세 조정을 통해 성능 향상 가능  
        - 다양한 문장 표현을 실험하거나 파인튜닝 모델을 통해 성능을 개선할 수 있음

*   **기능:**
    *   정보 정리  
        - 예: 복잡한 이메일 내용을 요약하거나 회의록을 작성
    *   고객 응대  
        - 챗봇을 통해 기본적인 고객 질문 자동 응답
    *   추천, 정보 검색, 콘텐츠 생성  
        - 예: 제품 추천, 글쓰기 보조, 뉴스 요약 등

*   **예시:**  
    - Siri, Alexa, ChatGPT 등은 주로 사용자의 명령을 받아 특정 작업을 수행

*   **활용 분야:**
    *   고객 서비스 (챗봇)  
        - 기본 문의 응답, 주문 상태 확인 등
    *   코드 생성  
        - 개발자가 작성한 요구사항을 바탕으로 코드 스니펫 제공

*   **장점:**
    *   반복적인 작업 시간 단축  
        - 반복 업무 자동화로 효율성 향상

*   **한계:**
    *   프롬프트의 작은 변화에도 오류 발생 가능 (brittleness)  
        - “정확히 어떻게 말하느냐”에 따라 응답 품질이 크게 달라질 수 있음

### **74.2 AI 에이전트**

*   **특징:**
    *   초기 프롬프트만으로 **자율적**으로 목표 달성  
        - 사용자가 모든 단계를 지시하지 않아도 스스로 작업을 분해하고 수행
    *   자체적으로 워크플로우 설계  
        - 예: 목표 달성을 위해 필요한 절차를 계획하고 실행
    *   외부 데이터, 도구 활용  
        - 웹 검색, API 호출, 코드 실행 등 다양한 리소스를 자유롭게 사용
    *   지속적인 기억을 통해 의사 결정 개선  
        - 과거 작업 결과나 피드백을 바탕으로 더 나은 선택 가능

*   **기능:**
    *   전략 수립 및 실행  
        - 예: 마케팅 캠페인 계획부터 실행까지 전 과정 처리
    *   최적의 의사 결정  
        - 여러 옵션을 비교하고 가장 효과적인 방안을 선택

*   **활용 분야:**
    *   자동화된 거래 (금융)  
        - 시장 변화에 따라 실시간으로 투자 전략 조정 및 실행
    *   네트워크 모니터링  
        - 이상 탐지, 대응 조치까지 자동 처리

*   **장점:**
    *   복잡하고 모호한 문제 해결에 적합  
        - 명확한 지시가 없어도 스스로 판단해 목표를 달성
    *   인간의 개입 없이 여러 작업 동시 처리 가능  
        - 복수의 태스크를 동시에 수행하며 효율성을 극대화

*   **한계:**
    *   피드백 루프에 갇힐 수 있음  
        - 반복적 사고나 비효율적 루틴을 스스로 계속 수행할 가능성
    *   높은 컴퓨팅 자원 필요 (비용 발생)  
        - 성능을 위해 GPU, 클라우드 자원 등을 많이 소모함

### **74.3 공통 기반**

*   대규모 언어 모델(LLM) 기반  
    - GPT, Claude, Gemini 등과 같은 언어 모델을 중심으로 작동하며, 인간 언어를 이해하고 처리하는 능력을 바탕으로 함


### **74.4 결론**

*   AI 어시스턴트와 에이전트는 상호 보완적인 관계  
    - 단순 요청에는 어시스턴트, 복잡한 문제 해결에는 에이전트가 적합
*   기술 발전으로 두 유형의 시너지 효과 기대  
    - 에이전트가 어시스턴트를 활용하는 등 협력 구조 가능
*   더욱 복잡한 문제 해결에 기여할 것으로 예상  
    - AI 시스템이 사람처럼 판단하고 실행하는 방향으로 발전 중


## 75. 데이터 기반 질문 응답 시스템 구축
- 출처: [AI for BI Automation](https://www.youtube.com/watch?v=sfyNLcHHDOM)

### 75.1 문제 상황

* 다양한 데이터 소스 (데이터베이스, 데이터 웨어하우스) 존재
  * 기업 내 Oracle, MySQL, PostgreSQL, Snowflake, BigQuery 등 이기종 시스템 산재
  * 데이터 소스별로 접근 방식과 쿼리 언어의 미묘한 차이 존재
* 수백 개의 테이블과 수많은 컬럼 존재
  * 일반적인 엔터프라이즈 환경에서 300-500개 테이블, 테이블당 평균 30-50개 컬럼 보유
  * 관계 및 종속성이 복잡하게 얽혀 있어 데이터 구조 이해에 전문성 필요
* "고객 만족도가 지난달 매출에 미치는 영향"과 같은 복잡한 질문에 대한 일관된 답변 필요
  * 다중 테이블 조인, 복잡한 집계, 시계열 분석 등이 요구됨
  * 비즈니스 도메인 지식과 데이터 구조에 대한 깊은 이해 필요

### 75.2 목표

* **확장성 (Scalable):** 대용량 데이터 처리 가능
  * 페타바이트 규모의 데이터도 효율적으로 처리할 수 있는 아키텍처
  * 동시 사용자 증가에도 성능 저하 없이 운영 가능
  * 새로운 데이터 소스 추가에 유연하게 대응
* **정확성 (Accurate):** 비즈니스 정의 및 어휘에 맞는 정확한 답변
  * 산업별, 부서별 특수 용어와 비즈니스 규칙 준수
  * 계산식, KPI 정의에 일관성 유지
  * 데이터 품질 및 신뢰성 보장 메커니즘 포함
* **일관성 (Consistent):** 동일하거나 유사한 질문에 대해 동일한 답변 제공
  * 질문 형식이나 표현이 달라도 의미적으로 동일한 질문에는 같은 결과 제공
  * 시간에 따른 데이터 업데이트에도 계산 로직 일관성 유지
  * 여러 사용자, 부서 간 공통된 이해와 의사결정 기반 제공

### 75.3 기존 방식의 문제점

* 모든 데이터를 데이터베이스에서 추출하여 LLM에 전달하는 방식은 데이터 양이 많을 경우 토큰 제한으로 인해 실패
  * 대규모 테이블(수백만~수억 행)의 경우 LLM 컨텍스트 윈도우 한계(8K-128K 토큰) 초과
  * 데이터 전처리 및 필터링 없이 원시 데이터 전달 시 처리 시간 및 비용 급증
  * 실시간 응답이 어려워 사용자 경험 저하
* LLM이 비즈니스 정의, 어휘, 특정 계산 방식을 이해하지 못함
  * 도메인 특화 용어와 계산 방식에 대한 학습 데이터 부족
  * 기업 내부 정의된 비즈니스 규칙과 KPI 해석 오류 발생
  * 의미적 모호성으로 인한 잘못된 테이블/컬럼 선택 위험

### 75.4 개선된 접근 방식

1. **SQL 생성 단계 추가:**
   * LLM이 테이블 정보를 바탕으로 질문에 필요한 데이터만 추출하는 SQL 쿼리 생성
     * 스키마 메타데이터와 샘플 데이터를 활용한 최적화된 쿼리 구성
     * 복잡한 조인, 서브쿼리, 윈도우 함수 등 고급 SQL 패턴 자동 적용
     * 쿼리 실행 계획 분석을 통한 성능 최적화
   * 데이터 양을 줄여 확장성 확보
     * 필요한 컬럼과 레코드만 선별적으로 추출하여 처리 효율성 증대
     * 데이터 필터링 및 집계를 데이터베이스 엔진에 위임하여 성능 향상
     * 점진적 쿼리 실행으로 대용량 데이터 처리 가능

2. **시맨틱 레이어 도입:**
   * 테이블 및 컬럼에 대한 비즈니스 정의 및 어휘를 담은 아티팩트
     * 테이블 간 관계, 데이터 계보, 품질 지표 등 메타데이터 포함
     * 도메인별 용어집과 동의어 사전 구축
     * 컬럼별 데이터 유형, 제약조건, 허용 범위 정의
   * LLM이 비즈니스 맥락을 이해하고 정확한 답변을 생성하도록 지원
     * 질문에서 비즈니스 용어 식별 및 관련 데이터 요소 매핑
     * 비즈니스 규칙에 따른 데이터 해석 및 변환 가이드 제공
     * 결과 검증 및 신뢰도 평가 메커니즘 포함

3. **메트릭 정의 활용:**
   * KPI, 목표 추적, 예측 값, 기간별 계산 방식 등을 정의
     * 산업 표준 및 기업 고유의 성과 지표 표준화
     * 시간 단위(일별, 월별, 분기별) 집계 규칙 명시
     * 이상치 처리 및 데이터 정규화 방법론 포함
   * 질문과 관련된 컬럼과 테이블의 범위를 좁혀 정확성을 높임
     * 메트릭 관련 데이터 소스 우선순위 설정
     * 연관 지표 간의 상관관계 및 의존성 정보 활용
     * 특정 비즈니스 질문에 최적화된 데이터 패스 구성
   * 중앙 집중식 메트릭 정의 카탈로그를 통해 사용자 간 일관성 확보
     * 전사적 메트릭 거버넌스 체계 구축
     * 부서 간 일관된 정의와 계산 방식 공유
     * 메트릭 변경 이력 및 버전 관리를 통한 추적성 확보

### 75.5 결론

- 시맨틱 레이어와 메트릭 정의를 LLM과 결합하여 데이터 기반 질문 응답 시스템의 확장성, 정확성, 일관성을 향상시킬 수 있습니다. 이러한 접근 방식은 데이터 양과 복잡성이 증가하는 환경에서도 신뢰할 수 있는 분석 결과를 제공하며, 기업 전체의 데이터 기반 의사결정 문화를 촉진합니다.

## 76. Generative AI의 영향과 편향 문제 해결 방안
- 출처: [Unraveling AI Bias: Principles & Practices](https://www.youtube.com/watch?v=ZsjDvyuxxgg)

### **76.1 Generative AI의 영향**

*   **경제적 영향:** Generative AI는 산업 전반에 걸쳐 혁신을 가져오며, 예를 들어 법률 문서 요약을 통해 변호사의 업무 효율성을 높이고, 고객 참여를 증진시켜 기업의 마케팅 효과를 극대화하며, 자동화로 비용을 절감하는 등 경제적 파급력이 크다. 이는 새로운 시장 창출과 기존 산업의 구조적 변화를 동반한다.
*   **장점:** 복잡한 데이터 분석이나 창작 작업을 인간보다 빠르게 수행할 수 있으며, 생산성 향상을 통해 기업의 경쟁력을 강화하고, 제품 및 서비스의 개발부터 출시까지 걸리는 시간을 단축하여 시장 대응 속도를 높인다.

### **76.2 Generative AI의 위험**

*   **새로운 위험:**
    *   다운스트림 기반 모델 재학습 시 문제 발생 가능성: 다른 모델의 출력 데이터를 학습에 재사용하면서 오류나 편향이 증폭될 수 있다.
    *   저작권 침해: 생성된 콘텐츠가 기존 저작물과 유사할 경우 법적 분쟁으로 이어질 가능성이 있다.
*   **기존 위험:**
    *   개인 정보 유출: 학습 데이터에 포함된 민감한 정보가 의도치 않게 노출될 수 있다.
    *   결과에 대한 모델 설명 부족 (투명성 부족): AI가 왜 특정 결정을 내렸는지 사용자가 이해하기 어려운 "블랙박스" 문제가 발생한다.
*   **가장 큰 위험:** 편향 (Bias): AI가 공정하지 않은 결정을 내리며 사회적 문제를 악화시킬 가능성이 가장 큰 위험으로 꼽힌다.

### **76.3 AI 편향 (Bias) 이란?**

*   AI 시스템이 편향된 결과를 생성하는 현상: 학습 데이터나 설계 과정에서 비롯된 불공정성이 결과물에 반영되는 것을 의미한다.
*   사회적 불평등을 반영하고 강화하는 결과: 예를 들어, 인종, 성별, 사회경제적 지위와 같은 요소에서 차별을 재생산하며, 이는 사회적 갈등을 심화시킬 수 있다.

### **76.4 AI 편향의 종류**

*   **알고리즘 편향:** AI 시스템의 설계나 작동 방식에 내재된 체계적 오류로, 특정 집단에 대해 지속적으로 불공정한 결과를 낳는다. 
    - 예: 채용 AI가 특정 성별을 선호하도록 설계될 수 있다.
*   **인지 편향:** AI 설계자가 자신의 주관적 경험에 기반해 시스템을 설계하면서 생기는 편향. 
    - 예를 들어, 최근성 편향은 최근 사건에 과도하게 영향을 받아 오래된 데이터를 무시하는 경우를 말한다.
*   **확증 편향:** AI가 설계자나 사용자의 기존 믿음을 강화하는 방향으로 데이터를 해석하는 경향. 
    - 예: 특정 정치적 성향을 지지하는 데이터만 강조될 수 있다.
*   **외집단 동질성 편향:** 특정 집단 외부의 사람들을 모두 유사하다고 간주하며 다양성을 무시하는 편향. 
    - 예: 소수 집단의 특성을 과소평가하거나 일반화할 수 있다.
*   **편견:** 사회적 고정관념에서 비롯된 오류로, 
    - 예를 들어 "간호사는 모두 여성이다"와 같은 잘못된 가정을 AI가 학습할 수 있다.
*   **배제 편향:** 데이터 수집 과정에서 의도치 않게 특정 집단이나 중요한 변수를 누락시키는 경우. 
    - 예: 저소득층 데이터를 포함시키지 않아 결과가 왜곡될 수 있다.

### **76.5 AI 편향 해결을 위한 AI 거버넌스**

*   AI 활동을 감독, 관리, 모니터링하는 방법: 
    - AI 시스템의 개발부터 배포, 운영까지 전 과정을 체계적으로 관리
*   책임감 있는 AI 개발을 위한 정책, 관행, 프레임워크: 
    - 윤리적 가이드라인과 법적 기준을 마련 개발자가 준수
*   공정성, 형평성, 포용성을 탐지하는 도구와 기술 활용: 
    - 편향을 식별하고 수정할 수 있는 알고리즘이나 소프트웨어를 도입한다.
*   기업, 직원, 고객에게 혜택을 제공하는 데 목표: 
    - AI가 사회적 가치를 창출하고 신뢰를 구축하도록 설계한다.

### **76.6 AI 편향 회피 방법**

*   **학습 모델 선택:**
    *   비즈니스 기능에 적합하고 확장 가능한 모델 선택: 
        - 기업의 목표와 규모에 맞는 모델을 선택해 효율성과 공정성을 동시에 확보한다.
    *   지도 학습 모델: 
        - 다양한 이해관계자(예: 전문가, 사용자, 소수 집단 대표)가 학습 데이터 선정에 참여해 편향을 감소
    *   비지도 학습 모델: 
        - 편향 탐지 도구(예: Google의 What-If Tool, IBM의 AI Fairness 360)를 사용해 결과를 검증
*   **균형 잡힌 AI 팀 구성:**
    *   인종, 경제적 지위, 교육 수준, 성별 등 다양한 배경의 팀원 구성: 
        - 서로 다른 관점을 반영해 편향을 최소화한다.
    *   혁신가, AI 개발자, AI 사용자 모두 포함: 
        - 기술적 전문성과 실사용 경험을 결합해 현실적인 해결책을 도출한다.
*   **데이터 처리:**
    *   전처리, 인라인 처리, 후처리 단계에서 편향이 발생하지 않도록 주의: 
        - 데이터 정제(전처리), 모델 학습 중 조정(인라인 처리), 결과 보정(후처리)을 통해 공정성을 유지
*   **모니터링:**
    *   실제 데이터와 트렌드를 지속적으로 모니터링하여 AI 시스템을 업데이트: 
        - 사회 변화나 새로운 데이터 패턴을 반영해 모델을 최신 상태로 유지한다.


## 77. Llama 모델 활용
- 출처: [Llama in Action: Conversational AI, Language Generation, and More!](https://www.youtube.com/watch?v=ucGfGWo_duE)

### **77.1 Llama 소개**

* **가능성 제시:**  
  - Llama 모델은 텍스트와 이미지를 동시에 이해하고 생성하는 멀티모달 AI로,  
  - 일상적인 업무 자동화(예: 휴가 계획 짜기),  소셜 미디어 콘텐츠 생성 및 분석, 이미지 속 사물 인식 및 설명 등 다양한 분야에 적용 가능함. 개인 사용자부터 기업, 공공기관까지 활용 범위가 넓어짐.

* **목표:**  
  - Llama는 대규모 언어 모델(LLM)의 성능을 누구나 쉽게 활용할 수 있도록 설계되었으며,  
  - 이를 통해 작업의 효율성을 높이고, 창의적인 문제 해결을 유도하며, 새로운 비즈니스 기회와 혁신을 창출하는 것이 궁극적인 목표임.

### **77.2 Llama 3.2 최신 버전** (2024년 9월 출시)

* **이미지 추론 모델:**  
  - 110억 ~ 900억 파라미터 규모의 고성능 이미지 추론 모델 출시.  
  - 이미지 내의 물체 식별, 장면 분석, 복잡한 시각적 질문 응답 등에서 뛰어난 성능을 발휘.  
  - 특히, 비정형 데이터(사진, 스캔 문서 등)를 다루는 분야에서 활용 가치가 큼.

* **경량 텍스트 모델:**  
  - 10억, 30억 파라미터로 구성된 경량 모델은  
  - 스마트폰, 태블릿 등의 온디바이스 환경에서도 작동 가능하여 프라이버시 보호가 중요한 개인용 AI 애플리케이션에 적합함.  
  - 예: 로컬에서 작동하는 AI 비서, 메시지 자동 요약 앱 등.

* **Llama Stack:**  
  - 모델 배포, 파인튜닝, API 통합 등을 간소화할 수 있는 개발 스택 제공.  
  - 개발자는 별도의 복잡한 설정 없이 Llama 모델을 다양한 서비스에 쉽게 통합 가능.  
  - 특히 스타트업이나 중소기업의 AI 도입 장벽을 크게 낮춤.

### **77.3 Llama 활용 사례**

* **이미지 이해:**
  * **문서 이해:**  
    - 문서 내 차트나 표를 분석하고 의미를 추론하여 질문에 응답.  
    - 예: "이 보고서에서 수익이 증가한 이유는 무엇인가?"와 같은 질문에 답변 가능.
  * **시각적 질문 응답:**  
    - 이미지 기반의 맥락을 이해하고 적절한 응답 제공.  
    - 예: 스포츠 사진을 보고 "이 선수는 어떤 종목에 출전 중인가?"에 대한 답변 가능.
  * **이미지 캡셔닝:**  
    - 이미지의 내용을 자동으로 설명하는 문장 생성.  
    - 예: SNS 게시물 자동 설명, 이미지 분류기용 라벨 생성 등에 활용 가능.

* **언어 생성 및 요약:**
  * **언어 생성:**  
    - 블로그 글, 자기소개서, 대화 스크립트 등 다양한 텍스트 생성이 가능하며,  
    - 콘텐츠 제작의 생산성을 크게 향상시킴.
  * **요약:**  
    - 긴 회의나 대화 내용을 간결한 포인트로 요약.  
    - 예: 회의록을 3~4개의 핵심 내용으로 자동 정리하여 보고 시간 절약.
  * **온디바이스 활용:**  
    - 모바일 기기에서 바로 작동하는 개인화된 요약/재작성 기능 지원.  
    - 예: 문자 메시지를 더 정중하게 바꾸거나, 일정 요약 알림 생성 등.

* **대화형 AI:**
  * **챗봇/가상 비서:**  
    - 고객 서비스용 챗봇부터 개인 일정 관리 비서까지,  
    - 다양한 대화형 에이전트로 활용 가능. 사용자의 질문에 정확하고 자연스러운 응답 제공.
  * **온디바이스 가상 비서:**  
    - 인터넷 연결 없이도 작동 가능한 스마트 AI 비서.  
    - 텍스트 요약, 일정 확인, 알림 설정 등을 빠르게 처리함.

* **언어 번역:**
  * **일반 언어 번역:**  
    - 다양한 언어 간 번역을 실시간으로 수행할 수 있으며,  
    - 특히 여행, 교육, 비즈니스 커뮤니케이션에서 활용도 높음.
  * **코드 번역:**  
    - 프로그래밍 언어 간 코드 변환 지원.  
    - 예: 파이썬으로 작성된 알고리즘을 자바나 C++로 자동 변환하거나,  
    - 자연어 명령으로 코드 생성도 가능함.

### **77.4 Llama 모델 접근 방법**

* 소셜 미디어 플랫폼(Facebook AI Research 등), 모델 공유 플랫폼(Hugging Face),  
  다양한 생성 AI 플랫폼을 통해 누구나 모델을 다운로드하거나,API를 통해 직접 활용 가능.  
- 개발자는 Llama 기반 서비스를 빠르게 구축할 수 있으며, 일반 사용자도 노코드 도구를 통해 손쉽게 체험


## 78. Generative AI 모델 특화 튜닝
- 출처: [Fine Tuning Large Language Models with InstructLab](https://www.youtube.com/watch?v=pu3-PeBG0YU)

### **78.1 문제점**
* 일반적인 LLM은 특정 분야에 대한 전문 지식이 부족하여 유용한 답변을 얻기 어려움. 예를 들어, 의학이나 법률과 같은 전문 분야에서는 일반 모델이 정확하고 깊이 있는 정보를 제공하지 못하는 경우가 많음.
* 모델에게 원하는 행동 양식을 예시로 제공하는 방식은 비효율적임. 매번 프롬프트에 상세한 지시사항과 예시를 포함해야 하므로 토큰 소모가 크고, 일관된 결과를 얻기 어려움.

### **78.2 해결책**
* 오픈 소스 LLM을 활용하여 특정 분야에 대한 지식을 내재화하는 **미세 조정(Fine-tuning)**을 수행. 이는 사전 훈련된 모델에 특정 도메인의 데이터로 추가 학습을 진행하여 해당 분야에 특화된 모델을 만드는 과정임.

### **78.3 미세 조정의 장점**
* 더 짧은 프롬프트로 더 나은 응답 가능 
    - 전문 지식이 모델 가중치에 직접 반영되어 복잡한 프롬프트 엔지니어링 없이도 정확한 답변 생성
* 더 빠른 추론 및 낮은 계산 비용 
    - 긴 컨텍스트 윈도우를 사용하지 않아도 되므로 inference 과정에서 자원 사용량 감소
* 특정 분야를 더 잘 이해하는 모델 생성 
    - 해당 분야의 전문 용어, 맥락, 지식 체계를 내재화하여 더 전문적인 답변 제공

### **78.4 InstructLab 프로젝트 소개**
* AI 모델에 대한 커뮤니티 기반 기여를 장려하고 접근성을 높이기 위한 연구 기반 프로젝트. 기술적 장벽을 낮추어 다양한 분야의 전문가들이 AI 모델 개발에 참여할 수 있도록 함.
* 개인 노트북에서도 미세 조정 가능. 고성능 GPU나 대규모 컴퓨팅 자원 없이도 효율적인 파라미터 튜닝 방식을 통해 소규모 환경에서 실행 가능.

### **78.5 미세 조정 3단계**
1. **데이터 선별(Curation):** 
    - 모델이 학습할 데이터 수집 및 구성. 
    - 특정 도메인의 지식을 담은 질문-답변 쌍, 전문 문서, 사례 연구 등 고품질 데이터를 선별하는 과정으로, 최종 모델의 품질을 결정하는 중요한 단계임.

2. **합성 데이터 생성:** 
    - 로컬에서 실행되는 LLM을 사용하여 초기 예제에서 합성 데이터 생성 (데이터 양 확보). 
    - 소수의 고품질 예제로부터 다양한 변형을 생성하여 훈련 데이터셋을 확장하는 방법으로, 데이터 부족 문제를 해결하고 모델의 일반화 능력을 향상시킴.

3. **모델에 적용:** 
    - LoRA(Low-Rank Adaptation) 다단계 튜닝 기술을 사용하여 데이터를 모델에 통합. 
    - 전체 모델 파라미터 대신 적은 수의 어댑터 파라미터만 훈련시켜 컴퓨팅 자원을 효율적으로 사용하면서도 효과적인 특화 성능을 달성함.

### **78.6 InstructLab 사용 예시**
1. **환경 설정:** 
    - `ilab config init` 명령어를 사용하여 작업 디렉토리 설정 및 기본 파라미터 지정. 
    - 이 과정에서 사용할 베이스 모델, 학습률, 배치 크기 등 다양한 훈련 매개변수를 설정할 수 있음.

2. **데이터 구성:** 
    - Taxonomy 저장소를 사용하여 기술(Skills) 및 지식(Knowledge)을 계층적으로 구성 (YAML 형식의 질문-답변 문서 활용). 
    - 이를 통해 모델이 학습해야 할 영역을 체계적으로 정의하고, 필요한 지식과 능력을 명확히 구조화

3. **합성 데이터 생성:** 
    - `ilab data generate` 명령어를 사용하여 로컬 또는 원격 모델을 통해 추가 예제 생성 (다양한 변형 생성). 
    - 초기 예제의 패턴을 학습하여 유사하지만 다양한 새로운 예제를 생성함으로써 데이터셋의 다양성과 규모를 확장함.

4. **모델 훈련:** 
    - `ilab model train` 명령어를 사용하여 새로운 지식과 기술을 모델에 통합 (파라미터 효율적인 미세 조정). 
    - 훈련 과정에서 검증 데이터를 통한 성능 평가와 체크포인트 저장이 자동으로 이루어져 최적의 모델을 선택할 수 있음.

### **78.7 미세 조정 결과**
* 특정 지식이 내재화된 새로운 모델 생성. 이 모델은 해당 분야의 질문에 대해 더 정확하고 상세한 답변을 제공할 수 있음.
* 예시: 2024년 오스카상 최다 노미네이트 작품 질문에 대한 답변 수정. 미세 조정 전에는 부정확하거나 일반적인 답변을 제공했지만, 미세 조정 후에는 정확한 정보와 관련 맥락까지 제공할 수 있게 됨.

### **78.8 활용 방안**
* RAG(Retrieval Augmented Generation)와 함께 사용하여 최신 정보 제공. 미세 조정된 모델이 기본 지식을 담당하고, RAG 시스템이 최신 정보나 추가 세부사항을 제공하는 하이브리드 접근법 구현 가능.
* 정적 리소스 변경 시 자동 또는 정기적인 빌드 수행. 기업 정책, 제품 카탈로그, 내부 지식 베이스 등이 업데이트될 때마다 모델을 자동으로 재훈련하여 항상 최신 정보를 반영할 수 있음.

### **78.9 InstructLab의 미래**
* AI 기여자 커뮤니티 구축 및 도메인 특화 모델 협업. 다양한 분야의 전문가들이 자신의 지식을 AI 모델에 통합하고 이를 공유함으로써 집단 지성을 활용한 AI 발전 생태계 조성.
* 예시: 보험 회사, 법률 회사 등에서 특정 분야에 대한 모델을 구축하여 업무 효율성 향상. 내부 규정, 사례법, 산업 용어 등을 학습한 모델을 통해 직원들의 의사결정 지원 및 고객 응대 품질 개선 가능.

### **78.10 결론**
* InstructLab과 같은 오픈 소스 프로젝트를 통해 AI/ML 전문가가 아니어도 LLM을 특화하여 활용 가능. 이는 AI 민주화를 촉진하고, 다양한 분야에서 AI 활용을 확대하는 데 기여함.
* 개인 노트북에서 로컬 데이터를 사용하여 모델을 훈련하고 온프레미스, 클라우드 또는 다른 사람과 공유 가능. 데이터 프라이버시를 유지하면서도 효과적인 AI 솔루션을 개발할 수 있어, 민감한 정보를 다루는 산업이나 기관에서도 안전하게 활용 가능.


## 79. 2025년 AI 주요 트렌드 (예상)
- 출처: [AI Trends for 2025](https://www.youtube.com/watch?v=5zuF4Ys1eAw)


### **79.1 에이전트 AI (Agentic AI)**
*   **정의:** 추론, 계획, 실행 능력을 갖춘 지능형 시스템. 복잡한 문제를 해결하기 위해 다단계 계획을 수립하고, 도구 및 데이터베이스와 상호 작용.
    *   예시: "고객의 이메일 문의를 분석 → 관련 데이터베이스에서 결제 이력 조회 → 환불 규정에 따라 자동 처리" 같은 다단계 작업 수행.
*   **문제점:** 현재 모델은 일관된 논리적 추론에 어려움을 겪으며, 복잡한 시나리오에서 판단 오류 발생.
    *   구체적 사례: 장기 프로젝트 관리 시 중간 단계에서 목표를 잊고 일관성이 떨어지는 응답 생성.
*   **전망:** 계층적 추론(Hierarchical Reasoning)이나 **"검증-실행" 루프** 같은 기술로 개선될 전망.

### **79.2 추론 시간 연산 (Inference Time Compute)**
*   **정의:** 모델이 실시간 데이터에 대해 추론하는 데 소요되는 시간. 복잡한 요청은 더 많은 추론 시간을 필요
    *   비유: 인간이 쉬운 문제는 즉시 답하지만, 복잡한 수학 문제는 풀이 단계를 거치는 것과 유사.
*   **특징:** 추론 과정은 모델 재학습 없이 개선 가능.
    *   장점: "Chain of Thought(단계적 사고)"를 명시적으로 학습시켜 정확도 향상.
*   **전망:** 추론 시간을 유동적으로 조절하는 **"동적 계산 할당"** 기술이 핵심이 될 것.

### 79.3 **초대형 모델 (Very Large Models):**
*   **정의:** 수많은 매개변수로 구성된 대규모 언어 모델.
    *   매개변수 50조 개 시나리오: 현재 GPT-4급 모델의 25배 이상. 물리적 한계로 인해 효율적 학습 알고리즘(예: Mixture of Experts)이 필수.
*   **규모:** 현재 (2024년) 프론티어 모델은 1~2조 개의 매개변수를 가지며, 차세대 모델은 50조 개 이상 예상.
    *   도전 과제: 연산 비용과 탄소 배출 문제로 인해 "그린 AI" 연구와 병행 필요.

### 79.4 **초소형 모델 (Very Small Models):**
*   **정의:** 몇십억 개의 매개변수만으로 구성된 모델.
    *    사례: 의료 영상 분석용 10억 개 매개변수 모델은 동일 작업에서 GPT-4보다 빠르고 정확할 수 있음.
*   **특징:** 노트북이나 휴대폰에서도 실행 가능하며, 특정 작업 수행에 특화.
    *   장점: 개인정보 보호(데이터가 기기를 떠나지 않음)와 저지연 응답이 가능.
*   **예시:** 20억 개 매개변수의 IBM Granite 모델.
*   **전망:** **"태스크-특화형 미세 조정(Task-Specific Fine-Tuning)"**이 표준화될 것.

### 79.5 **고급 활용 사례 (Advanced Use Cases):**
*   **2024년 주요 기업 AI 활용 사례:** 고객 경험 개선, IT 운영 자동화, 가상 비서, 사이버 보안.
    *   한계: 대부분 규칙 기반 또는 단순 질의 응답 수준.
*   **2025년 예상:** 
    *   고객 서비스: 계약서 협상 시 상대방의 이메일 톤을 분석해 최적의 반응 제안.
    *   IT 최적화: 서버 트래픽 패턴을 실시간으로 학습해 에너지 사용량 20% 절감.
    *   보안: 제로데이 공격을 사전에 예측하는 행위 기반(Behavioral) AI.

### 79.6  **거의 무한한 기억 (Near Infinite Memory):**
*   **배경:** 컨텍스트 창 크기가 2,000 토큰에서 수십만/수백만 토큰으로 확장됨.
    *   의미: 1,000페이지 분량의 문서를 한 번에 처리 가능해짐.
*   **전망:** 
    *   개인화: 사용자의 10년간 대화 기록을 기반으로 취향 예측.
    *   리스크: 프라이버시 침해 논란과 "기억 삭제 권리" 필요성 대두.

### 79.7 **인간-AI 협업 강화 (Human in the Loop Augmentation):**
*   **배경:** 챗봇이 임상 추론에서 의사보다 뛰어난 성능을 보인 연구 결과 존재.
    *   예: 2023년 NEJM 연구에서 AI가 희귀질환 진단 정확도 72% vs 의사 평균 55%.
*   **문제점:** 현재 AI 도구를 활용한 의사 진단 결과가 AI 단독 진단보다 낮은 경우 발생.
    *   원인: AI의 결과를 과신하거나 무시하는 극단적 태도.
*   **전망:** 
    *   "AI as a Co-Pilot" 시스템: 의사가 AI의 추론 과정을 단계별로 검토하며 보조.
    *   자동화된 설명 기능(Explainable AI): AI가 판단 근거를 자연어로 설명.


## 80. 손실 함수
- 출처: [What is a Loss Function? Understanding How AI Models Learn](https://www.youtube.com/watch?v=v_ueBW_5dLg)

### **80.1 손실 함수란?**

*   **정의:** AI 모델의 예측값과 실제값(ground truth) 간의 차이(오차)를 수치화하여 모델의 성능 측정지표  
    *   손실 함수는 머신러닝 및 딥러닝에서 학습 과정의 핵심 요소로, 모델이 얼마나 잘 작동하는지 판단하는 기준을 제공합니다.
*   **정확한 예측:** 손실(loss)이 작음  
    *   손실 값이 작다는 것은 모델의 예측값이 실제값과 가까워졌음을 의미하며, 이는 모델의 신뢰도가 높아졌음을 나타냅니다.
*   **부정확한 예측:** 손실이 큼  
    *   손실 값이 크면 모델의 예측값이 실제값과 크게 벗어나 있다는 것을 의미하며, 이 경우 모델의 성능이 낮다고 평가됩니다.

### **80.2 손실 함수 활용 예시: 유튜브 조회수 예측 모델**

*   **모델 설명:** 동영상 제목을 입력받아 조회수를 예측하는 AI 모델  
    *   예를 들어, "AI로 쉽게 배우는 프로그래밍"이라는 제목의 동영상이 10,000회 조회될 것으로 예측되었다고 가정해봅시다.
*   **예측값과 실제값 비교:** 예측값과 실제 조회수를 비교하여 모델 성능 평가  
    *   만약 실제 조회수가 8,000회였다면, 모델의 오차는 2,000회이며, 이를 손실 함수를 통해 정량적으로 계산할 수 있습니다.
*   **모델 개선:** 손실 함수를 통해 모델의 파라미터 조정 -> 모델 성능 개선  
    *   손실 함수는 모델의 학습 과정에서 파라미터(weight, bias 등)를 조정하는 데 사용되며, 이를 통해 모델이 점점 더 정확한 예측을 할 수 있도록 유도합니다.

### **80.3 손실 함수의 종류**

*   **회귀 손실 함수 (Regression Loss Functions):** 연속적인 값 예측의 정확도 측정 (예: 집값, 온도, 유튜브 조회수)  
    *   회귀 문제에서는 예측값과 실제값 사이의 거리를 측정하는 방식이 주로 사용됩니다.

    *   **MSE (Mean Squared Error, 평균 제곱 오차):** 예측값과 실제값 차이의 제곱 평균  
        *   MSE는 오차를 제곱하기 때문에 큰 오차에 더 큰 가중치를 부여합니다. 이는 이상치(outlier)에 민감한 특성을 가지고 있으며, 이상치가 포함된 데이터셋에서는 신중히 사용해야 합니다.
        *   예시: 예측값 [10, 15, 20], 실제값 [12, 14, 25]일 때, MSE = ((-2)^2 + (1)^2 + (-5)^2)/3 = 10
    *   **MAE (Mean Absolute Error, 평균 절대 오차):** 예측값과 실제값 차이의 절대값 평균  
        *   MAE는 오차의 절대값을 사용하므로 MSE에 비해 이상치에 덜 민감합니다. 따라서 이상치가 많은 데이터셋에서 유리할 수 있습니다.
        *   예시: 위와 같은 데이터셋에서 MAE = (| -2 | + | 1 | + | -5 |)/3 = 2.67
    *   **Huber Loss:** MSE와 MAE의 절충안  
        *   Huber Loss는 작은 오차에는 MSE처럼, 큰 오차에는 MAE처럼 동작합니다. 이는 이상치를 적절히 처리하면서도 안정적인 학습을 가능하게 합니다.
        *   예시: δ=1로 설정하면, 오차가 1 이하일 때는 MSE처럼 작동하고, 1 초과일 때는 MAE처럼 작동

*   **분류 손실 함수 (Classification Loss Functions):** 범주형 예측의 정확도 측정 (예: 스팸 메일 분류, 식물 종 분류)  
    *   분류 문제에서는 예측 확률과 실제 클래스 간의 불일치를 측정하는 방식이 주로 사용됩니다.

    *   **Cross Entropy Loss (교차 엔트로피 손실):** 모델 예측의 불확실성을 실제 결과와 비교하여 측정  
        *   교차 엔트로피 손실은 분류 문제에서 가장 널리 사용되는 손실 함수 중 하나로, 모델의 예측 분포와 실제 분포 사이의 차이를 측정합니다.
        *   예시: 스팸 메일 분류에서 모델이 특정 메일이 스팸일 확률을 0.9로 예측했지만 실제 레이블이 1(스팸)이라면 손실이 작게 계산됩니다.
    *   **Hinge Loss:** SVM(Support Vector Machine)에서 주로 사용, 클래스 간의 margin을 최대화하여 모델이 높은 확신도로 정확한 예측을 하도록 유도  
        *   Hinge Loss는 특히 선형 분류기에서 클래스 간 경계(margin)를 명확히 만드는 데 효과적입니다.
        *   예시: 두 클래스 간의 margin이 1보다 작으면 손실이 발생하며, margin이 클수록 손실이 줄어듦.

### **80.4 손실 함수 활용 방법**

*   **모델 성능 평가:** 손실 함수 값을 통해 모델 성능을 정량적으로 평가  
    *   손실 함수 값이 작을수록 모델이 더 좋은 성능을 발휘한다고 판단할 수 있습니다. 이를 통해 다양한 모델을 비교하거나 하이퍼파라미터 튜닝의 기준으로 사용할 수 있습니다.
*   **모델 파라미터 조정:** 손실 함수 값을 최소화하는 방향으로 모델 파라미터(weight, bias)를 조정 (최적화, Optimization)  
    *   예를 들어, 경사 하강법(Gradient Descent)을 사용하여 손실 함수의 기울기를 계산하고, 이를 바탕으로 파라미터를 조금씩 수정합니다.
*   **Gradient Descent (경사 하강법):** 손실 함수의 기울기(gradient)를 계산하여 손실을 줄이는 방향으로 모델 파라미터를 업데이트  
    *   경사 하강법은 손실 함수의 최솟값을 찾기 위해 반복적으로 파라미터를 업데이트하는 알고리즘입니다. 학습률(learning rate)을 적절히 설정하는 것이 중요합니다.
*   **모델 학습 종료 시점 결정:** 손실이 미리 정의된 임계값 이하로 감소하면 모델 학습 종료  
    *   학습 도중 손실이 더 이상 줄어들지 않거나 일정 수준 이하로 내려가면 학습을 멈출 수 있습니다. 이를 통해 과적합(overfitting)을 방지할 수 있습니다.

### **80.5 유튜브 조회수 예측 모델 개선 결과**

*   **결과 분석:** 모델 파라미터 조정 후, MAE, MSE, Huber Loss 모두 감소  
    *   이는 모델의 예측값이 실제값에 가까워졌음을 의미하며, 특히 MAE가 감소했다는 것은 전체적인 오차가 줄어들었음을 나타냅니다.
*   **특징:** 특히, MSE 감소폭이 큰 것은 모델이 큰 오차를 줄이는 데 효과적이었음을 의미  
    *   MSE는 큰 오차에 더 큰 가중치를 부여하므로, MSE의 감소폭이 크다는 것은 모델이 극단적인 예측 실패를 줄이는 데 성공했다는 것을 암시합니다.

### **80.6 요약**  
손실 함수는 AI 모델의 성능을 평가하고 개선하는 데 중요한 역할을 하는 지표입니다. 적절한 손실 함수를 선택하고 활용함으로써 모델의 예측 정확도를 높일 수 있습니다. 또한, 손실 함수의 종류와 특성을 이해하면 다양한 문제에 맞는 최적의 손실 함수를 선택하여 모델의 성능을 극대화할 수 있습니다.
