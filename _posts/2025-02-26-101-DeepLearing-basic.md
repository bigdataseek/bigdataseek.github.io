---
title: 9차시 1:딥러닝 기초(개념)
layout: single
classes: wide
categories:
  - 딥러닝
tags:
  - ANN
  - CNN
  - RNN
---



## 1. 딥러닝 소개

### **1.1 인공지능, 머신러닝, 딥러닝의 관계**

1. 인공지능 (Artificial Intelligence, AI)
    - **정의**: 인간의 지능적 행동(추론, 학습, 문제 해결 등)을 모방하거나 이를 초월하는 기술.
    - **예시**: 체스 게임에서 승리하는 프로그램, 자율주행 차량, 개인화된 추천 시스템

2. 머신러닝 (Machine Learning, ML)
    - **정의**: 데이터로부터 패턴을 학습하고 이를 바탕으로 새로운 데이터를 예측하는 알고리즘.
    - **특징**:
        - 명시적인 규칙 작성 대신 데이터 기반 학습.
        - 주요 유형: 지도 학습(Supervised Learning), 비지도 학습(Unsupervised Learning), 강화 학습(Reinforcement Learning).
    - **예시**: 스팸 메일 필터링, 주가 예측, 고객 세분화

3. 딥러닝 (Deep Learning, DL)
    - **정의**: 다층 신경망(Deep Neural Network)을 사용하여 복잡한 패턴 학습하는 머신러닝의 하위 분야.
    - **특징**:
        - 입력 데이터를 여러 층(Layer)을 통해 점진적으로 추상화.
        - 인간이 특징을 직접 설계하지 않아도 자동으로 중요한 특징을 추출.
    - **예시**:이미지 분류(고양이 vs 강아지), 음성 인식("오늘 날씨 어때?" -> 텍스트 변환)
  
4. **관계 요약**
    - **인공지능(AI)**은 가장 넓은 개념으로, 머신러닝(ML)과 딥러닝(DL)을 포함.
    - **머신러닝(ML)**은 AI의 한 방법론이며, 데이터 기반 학습을 강조.
    - **딥러닝(DL)**은 머신러닝의 한 분야로, 특히 신경망을 활용해 복잡한 문제 해결.

![ai_deeplearning](/assets/images/ai_deeplearning.png)



### **1.2 딥러닝의 발전 역사**

1. 인공신경망의 역사적 배경
    - **초기 단계 (1940~1950년대)**:
        - 워런 맥컬록(Warren McCulloch)과 월터 피츠(Walter Pitts)가 최초의 인공 뉴런 모델 제안.
        - 퍼셉트론(Perceptron) 개발: 간단한 선형 분류 문제 해결 가능.
    - **겨울 시기 (1970~1980년대)**:
        - 데이터와 컴퓨팅 파워 부족으로 인해 연구가 정체됨.
        - "AI 겨울"로 불리는 시기.
    - **부활 (1990년대 이후)**:
        - 역전파 알고리즘(Backpropagation) 개발로 다층 신경망 학습 가능.
        - CNN(Convolutional Neural Network) 등 특정 문제에 특화된 아키텍처 등장.

2. 딥러닝 혁신의 주요 요인
    - **빅데이터(Big Data)**:
        - 인터넷, IoT, 소셜 미디어 등으로 인해 방대한 양의 데이터 수집 가능.
        - 데이터가 많을수록 딥러닝 모델의 성능 향상.
        - 예시: ImageNet 데이터셋(수백만 장의 이미지와 레이블).
    - **컴퓨팅 파워(Computing Power)**:
        - GPU(Graphics Processing Unit)의 등장으로 병렬 연산 처리 가능.
        - 클라우드 컴퓨팅을 통한 고성능 컴퓨팅 접근성 증가.
        - 예시: NVIDIA의 CUDA 기술로 딥러닝 연산 가속화.
    - 알고리즘 발전(Algorithmic Advancements)**:
        - ReLU(Rectified Linear Unit) 활성화 함수 도입으로 학습 속도 향상.
        - 드롭아웃(Dropout) 등 과적합 방지 기법 개발.
        - Transformer 아키텍처로 자연어 처리(NLP) 분야 혁신.


### **1.3 딥러닝 응용 사례**

1. 컴퓨터 비전 (Computer Vision)
    - **이미지 분류(Image Classification)**:입력 이미지를 특정 카테고리로 분류.
        - 예시: 고양이/강아지 구분, 의료 영상 분석.
    - **객체 탐지(Object Detection)**: 이미지 내 객체 위치와 클래스 동시 예측.        
        - 예시: 자율주행 차량에서 보행자, 신호등 탐지.
    - **세그멘테이션(Segmentation)**: 이미지를 픽셀 단위로 분할.        
        - 예시: 의료 영상에서 종양 영역 추출.

2. 자연어 처리 (Natural Language Processing, NLP)
    - **번역(Machine Translation)**: 언어 간 번역.
        - 예시: Google 번역(Google Translate).
    - **감성 분석(Sentiment Analysis)**: 텍스트의 감정(긍정/부정) 분류.
        - 예시: 영화 리뷰 분석.
    - **텍스트 생성(Text Generation)**: 문맥에 맞는 텍스트 생성.
        - 예시: GPT 모델을 활용한 글쓰기 지원.

3. 음성 인식 및 합성 (Speech Recognition & Synthesis)
    - **음성 인식(Speech Recognition)**: 음성을 텍스트로 변환.
        - 예시: Siri, Alexa.
    - **음성 합성(Speech Synthesis)**: 텍스트를 음성으로 변환.
        - 예시: TTS(Text-to-Speech) 기술.

4. 강화 학습 (Reinforcement Learning)
    - **게임(Game Playing)**: AlphaGo        
    - **로봇 제어(Robot Control)**: 로봇 팔을 이용한 물체 조작.
    - **자율주행(Autonomous Driving)**: 환경을 인식하고 최적의 경로 선택.


## **2. 신경망 기초**

### **2.1 퍼셉트론 모델**

1. 생물학적 뉴런과 인공 뉴런의 유사성
    - **생물학적 뉴런**:  
        - 인간의 뇌는 수십억 개의 뉴런으로 구성되어 있으며, 각 뉴런은 다른 뉴런들과 연결되어 정보를 전달합니다. 뉴런은 입력(시냅스)을 받아들여 이를 처리한 후 출력(액션 포텐셜)을 생성합니다.
    - **인공 뉴런**:  
        - 인공 뉴런은 생물학적 뉴런의 동작을 모방하여 설계되었습니다. 입력값에 가중치를 곱하고, 편향(bias)을 더한 후 활성화 함수를 통해 출력을 결정합니다. 이는 생물학적 뉴런의 시냅스 강도와 발화 여부를 모방한 것입니다.



2. 단일 퍼셉트론의 구조
    - **입력(Input)**:  
        - 퍼셉트론은 여러 개의 입력값 $$ x_1, x_2, \dots, x_n $$ 을 받습니다. 이 값들은 데이터의 특징(feature)을 나타냅니다.
    - **가중치(Weight)**:  
        - 각 입력값에는 가중치 $$ w_1, w_2, \dots, w_n $$이 할당됩니다. 가중치는 입력값의 중요도를 나타내며, 학습 과정에서 조정됩니다.
    - **편향(Bias)**:  
        - 편향은 입력값과 독립적으로 추가되는 상수항으로, 모델의 유연성을 높이는 역할을 합니다.
    - **활성화 함수(Activation Function)**:  
        - 가중합(weighted sum)을 계산한 후, 활성화 함수를 통해 최종 출력을 결정합니다. 활성화 함수는 비선형성을 도입하여 복잡한 문제를 해결할 수 있게 합니다.

        $$ z = \sum_{i=1}^n w_i x_i + b $$ 

3. 선형 분리 가능성과 XOR 문제
    - **선형 분리 가능성**:  
        - 단일 퍼셉트론은 선형 분리 가능한 문제만 해결할 수 있습니다. 즉, 데이터를 직선(또는 초평면)으로 나눌 수 있는 경우에만 작동합니다. 예를 들어, AND, OR 문제는 선형 분리 가능하지만 XOR 문제는 그렇지 않습니다.
    - **XOR 문제**:  
        - XOR 문제는 두 입력값이 같으면 0, 다르면 1을 출력하는 논리 연산입니다. 이를 해결하려면 다층 퍼셉트론(MLP)과 같은 비선형 모델이 필요합니다.

### **2.2 다층 퍼셉트론(MLP)**
1. 은닉층의 개념
    - **은닉층(Hidden Layer)**:  
        - 입력층과 출력층 사이에 위치하며, 데이터의 복잡한 패턴을 학습하는 역할을 합니다. 은닉층의 뉴런들은 비선형 변환을 통해 데이터를 고차원 공간으로 매핑합니다.
    - **다층 구조의 필요성**:  
        - 단일 퍼셉트론은 XOR 문제와 같은 비선형 문제를 해결할 수 없지만, 은닉층을 추가하면 비선형 문제도 해결할 수 있습니다.

2. 순방향 전파(Forward Propagation)
    - **순방향 전파 과정**:  
        1. 입력층에서 데이터를 받아 은닉층으로 전달합니다.
        2. 각 은닉층에서 가중합을 계산하고 활성화 함수를 적용합니다.
        3. 최종 출력층에서 결과를 생성합니다.
    - **수식 표현**:  
        
        $$ z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]} $$
        $$ a^{[l]} = f(z^{[l]}) $$

        여기서 $$ W^{[l]} $$는 가중치 행렬, $$ b^{[l]} $$는 편향 벡터, $$ f $$는 활성화 함수입니다.

3. 비선형 활성화 함수
    - **Sigmoid**:          

        $$ f(x) = \frac{1}{1 + e^{-x}} $$          

        - 출력값이 0과 1 사이로 제한되며, 초기 신경망에서 많이 사용되었으나 기울기 소실(vanishing gradient) 문제로 인해 최근에는 덜 사용됩니다.
    - **Tanh**:  

        $$ f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

        - 출력값이 -1과 1 사이로 제한되며, Sigmoid보다 중심화된 출력을 제공합니다.
    - **ReLU(Rectified Linear Unit)**:  
    
        $$ f(x) = \max(0, x) $$  
    
        - 양수 입력에 대해선 선형이며, 음수 입력에 대해선 0을 출력합니다. 계산이 간단하고 성능이 우수하여 현재 가장 널리 사용됩니다.
    - **Leaky ReLU**:  
        
        $$ f(x) = \max(0.01x, x) $$  

        - 음수 입력에 대해 작은 기울기를 부여하여 ReLU의 "죽은 뉴런" 문제를 완화합니다.


### **2.3 신경망 학습 원리**

1. 손실 함수(Loss Function)
    - **손실 함수의 역할**:  
        - 모델의 예측값과 실제값 사이의 차이를 측정하여 모델의 성능을 평가합니다. 학습 과정에서 손실을 최소화하는 방향으로 가중치를 업데이트합니다.
    - **MSE(Mean Squared Error)**:  

        $$ L = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 $$  

        - 회귀 문제에서 주로 사용되며, 예측값과 실제값의 차이를 제곱한 평균을 계산합니다.
    - **Cross-Entropy Loss**:  

        $$ L = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] $$  

        - 분류 문제에서 주로 사용되며, 확률 분포 간의 차이를 측정합니다.

2. 경사 하강법(Gradient Descent)
    - **경사 하강법의 원리**:  
        - 손실 함수를 최소화하기 위해 가중치를 반복적으로 업데이트합니다.  
        
        $$ w := w - \alpha \frac{\partial L}{\partial w} $$  
        
        - 여기서 $$ \alpha $$는 학습률(learning rate)로, 업데이트 크기를 결정합니다.
    - **변종 알고리즘**:  
        - **SGD(Stochastic Gradient Descent)**: mini-batch를 사용하여 계산 효율성을 향상.
        - **Adam**: 적응형 학습률을 사용하여 더 빠르고 안정적인 학습을 가능하게 합니다.

3. 역전파 알고리즘(Backpropagation)
    - **역전파의 목적**:  
        - 손실 함수를 최소화하기 위해 가중치의 기울기(gradient)를 계산합니다. 이를 위해 체인 룰(chain rule)을 활용하여 출력층부터 입력층까지 기울기를 역방향으로 전파합니다.
    - **과정 요약**:  
        1. 순방향 전파를 통해 출력값과 손실을 계산합니다.
        2. 출력층에서 시작하여 각 레이어의 기울기를 계산합니다.
        3. 계산된 기울기를 사용하여 가중치를 업데이트합니다.



## **3. 딥러닝 주요 개념**

### **3.1 과적합(Overfitting)과 일반화(Generalization)**
1. 훈련 데이터와 테스트 데이터
- **훈련 데이터(Training Data):** 모델이 학습하는 데 사용되는 데이터입니다. 모델은 이 데이터를 통해 패턴을 학습합니다.
- **테스트 데이터(Test Data):** 모델의 성능을 평가하기 위해 사용되는 데이터입니다. 테스트 데이터는 훈련 데이터와 독립적이어야 하며, 이를 통해 모델이 새로운 데이터에 얼마나 잘 일반화되는지 확인할 수 있습니다.

> **중요 포인트:**  
> - 데이터셋을 무작위로 나누어 훈련/테스트 세트로 분리합니다. (예: 80% 훈련, 20% 테스트)
> - 테스트 데이터는 모델 학습 중에 절대 사용되지 않아야 합니다.


### 3.2 과적합의 원인과 징후
1. **과적합(Overfitting):**
  - 모델이 훈련 데이터에 너무 과도하게 적합되어, 새로운 데이터에 대해 제대로 일반화하지 못하는 현상입니다.
2. **원인:**
  - 모델이 지나치게 복잡함 (너무 많은 파라미터).
  - 훈련 데이터가 부족하거나 다양성이 낮음.
  - 학습 횟수가 지나치게 많음.
3. **징후:**
  - 훈련 데이터에 대한 오차(Loss)는 매우 작지만, 테스트 데이터에 대한 오차는 크다.
  - 모델이 훈련 데이터의 노이즈까지 학습하여 예측 성능이 저하됨.


> "만약 여러분이 시험 준비를 할 때 특정 문제만 반복해서 풀면 어떻게 될까요?"  
> → "실제 시험에서 다른 유형의 문제가 나오면 풀기 어려울 것입니다."  



### 3.3 정규화 기법
정규화(Regularization)는 과적합을 방지하기 위한 기법입니다. 

1. **L1/L2 규제(Regularization):**
    - **L1 규제:** 가중치의 절댓값 합을 손실 함수에 추가하여 일부 가중치를 0으로 만듭니다. (희소성 유도)
    - **L2 규제:** 가중치의 제곱 합을 손실 함수에 추가하여 모든 가중치를 균등하게 줄입니다. (Ridge Regularization)
    - **효과:** 모델이 필요 이상으로 복잡해지는 것을 방지하며, 더 간단하고 일반적인 모델을 유도합니다.

2. **드롭아웃(Dropout):**
    - 정의: 학습 과정에서 랜덤하게 일부 뉴런을 비활성화시키는 기법입니다.
    - 효과: 특정 뉴런에 과도하게 의존하는 것을 방지하여 모델의 일반화 능력을 향상시킵니다.
    > "팀 프로젝트에서 한 명이 모든 일을 한다면 다른 팀원들이 실력을 발휘할 기회가 없습니다. 드롭아웃은 모든 팀원이 고르게 일하도록 유도하는 것과 같습니다."

3. **조기 종료(Early Stopping):**
    - 훈련 과정에서 검증 데이터의 오차가 더 이상 감소하지 않을 때 학습을 중단하는 기법입니다.
    - 불필요한 학습을 방지하여 과적합을 줄입니다.


## **3.4 최적화 알고리즘**
딥러닝 모델의 학습은 손실 함수(Loss Function)를 최소화하는 과정

1. 확률적 경사 하강법(SGD)
    - **SGD(Stochastic Gradient Descent):** 매번 하나의 샘플 또는 작은 배치(Batch)를 사용하여 손실 함수의 기울기를 계산하고 가중치를 업데이트합니다.
    - 장점: 계산 비용이 적고, 지역 최적해(Local Minima)를 벗어날 가능성이 큽니다.
    - 단점: 학습 과정이 불안정할 수 있음.

2. 모멘텀(Momentum)
    - **개념:**  
        - 기존의 경사 하강법에 관성을 추가하여 학습 속도를 안정적으로 유지합니다.
        - 이전 업데이트 방향을 기억하고, 그 방향으로 계속 진행하도록 유도합니다.
    - **효과:**  
        - 급격한 변화를 완화하고, 손실 함수의 계곡(Valley)을 따라 더 효율적으로 이동합니다.

3. Adam, RMSprop 등 적응형 최적화
    - **Adam(Adaptive Moment Estimation):**  
        - 모멘텀과 RMSprop의 장점을 결합한 알고리즘입니다.
        - 각 가중치에 대해 개별적인 학습률을 적용하여 학습을 가속화합니다.
    - **RMSprop(Root Mean Square Propagation):**  
        - 학습률을 자동으로 조절하여 학습 과정을 안정화합니다.
    - **장점:**  
        - 복잡한 모델에서도 안정적이고 빠른 학습이 가능합니다.


## **4. 주요 신경망 아키텍처 소개**

### 4.1 합성곱 신경망(CNN, Convolutional Neural Network)
1. **합성곱 레이어(Convolutional Layer):**
- 이미지 데이터에서 공간적 특징(예: 가장자리, 질감)을 추출하는 레이어입니다.
- 필터(Filter)를 사용하여 입력 데이터를 스캔하며 특징 맵(Feature Map)을 생성합니다.

2. **풀링 레이어(Pooling Layer):**
- 데이터의 차원을 줄여 연산량을 감소시키고, 중요한 특징만 남깁니다.
- 대표적인 풀링 방법: Max Pooling(최댓값 선택), Average Pooling(평균값 선택).

3. **이미지 처리에서의 효율성:**
- CNN은 이미지 데이터의 공간적 구조를 보존하며, 국소적인 패턴을 효과적으로 학습합니다.
- 응용: 객체 인식, 이미지 분류, 세그멘테이션 등.

### 4.2 순환 신경망(RNN, Recurrent Neural Network)
1. **시퀀스 데이터 처리:**
- RNN은 시간에 따라 변하는 데이터(예: 텍스트, 음성)를 처리하기 위해 설계되었습니다.
- 이전 상태를 기억하여 현재 입력과 결합하여 출력을 생성합니다.

2. **LSTM(Long Short-Term Memory)과 GRU(Gated Recurrent Unit):**
- **문제점:** 기본 RNN은 긴 시퀀스 데이터에서 기울기 소실(Vanishing Gradient) 문제가 발생할 수 있습니다.
- **LSTM:** 게이트(Gate) 메커니즘을 통해 중요한 정보를 장기적으로 유지합니다.
- **GRU:** LSTM의 간소화된 버전으로, 성능은 유사하지만 계산 비용이 적습니다.

### 4.3 트랜스포머(Transformer)
- **개념:**  
  - RNN의 한계를 극복하기 위해 설계된 아키텍처입니다.
  - Self-Attention 메커니즘을 통해 입력 시퀀스 내 모든 요소 간의 관계를 동시에 고려합니다.
- **응용:**  
  - 자연어 처리(NLP)에서 혁신적인 성과를 거두었으며, GPT, BERT 등의 모델에 활용됩니다.

