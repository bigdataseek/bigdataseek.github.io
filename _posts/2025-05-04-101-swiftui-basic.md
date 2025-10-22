---
title: 7차시 11:SwiftUI Basic
layout: single
classes: wide
categories:
  - SwiftUI
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

# SwiftUI 기초 개념 

## 1. SwiftUI의 선언적 구문과 View 계층

### 1.1 학습 목표
- 명령형 vs 선언형 UI의 차이 이해
- View 계층 구조 이해
- 기본 레이아웃 컴포넌트 활용

### 1.2 명령형 vs 선언형
```swift
// UIKit (명령형): "어떻게" 그릴지 설명
let label = UILabel()
label.text = "Hello"
label.textColor = .blue
view.addSubview(label)

// SwiftUI (선언형): "무엇을" 그릴지 설명
Text("Hello")
    .foregroundStyle(.blue)
```

### 1.3 View 프로토콜
```swift
// 모든 SwiftUI View는 View 프로토콜을 따름
struct ContentView: View {
    var body: some View {
        Text("Hello, World!")
    }
}
```

**핵심 포인트**:
- `body`는 단 하나의 View만 반환
- 여러 View를 담으려면 컨테이너(VStack, HStack 등) 필요
- `some View`는 불투명 반환 타입

### 1.4 기본 View 컴포넌트
```swift
// 텍스트
Text("안녕하세요")
    .font(.title)
    .foregroundStyle(.blue)
    .bold()

// 이미지
Image(systemName: "star.fill")
    .resizable()
    .frame(width: 50, height: 50)
    .foregroundStyle(.yellow)

// 버튼
Button("클릭") {
    print("버튼 클릭됨")
}

// 도형
Circle()
    .fill(.blue)
    .frame(width: 100, height: 100)
```

1\.실습 1: 프로필 카드 만들기
```swift
struct ProfileCard: View {
    var body: some View {
        VStack(spacing: 12) {
            // 프로필 이미지
            Image(systemName: "person.circle.fill")
                .resizable()
                .frame(width: 80, height: 80)
                .foregroundStyle(.blue)
            
            // 이름
            Text("홍길동")
                .font(.title)
                .bold()
            
            // 직함
            Text("iOS Developer")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            
            // 버튼
            Button("팔로우") {
                print("팔로우 클릭")
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
        .background(.gray.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 20))
    }
}
```

2\.과제
1. 자신의 프로필 카드 만들기 (이미지, 이름, 소개, 버튼)
2. 다양한 SF Symbols 아이콘 활용해보기
3. 색상과 폰트 조합 실험하기

### 1.5 실습코드
```swift
import SwiftUI

// MARK: - 1. 기본 View 컴포넌트 연습
struct BasicComponentsView: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 30) {
                // 텍스트 스타일
                Group {
                    Text("기본 텍스트")
                    
                    Text("제목")
                        .font(.title)
                    
                    Text("굵은 텍스트")
                        .bold()
                    
                    Text("파란색 텍스트")
                        .foregroundStyle(.blue)
                    
                    Text("여러 줄 텍스트입니다. 긴 텍스트가 어떻게 표시되는지 확인해보세요.")
                        .multilineTextAlignment(.center)
                        .padding()
                }
                
                Divider()
                
                // 이미지와 아이콘
                Group {
                    Image(systemName: "star.fill")
                        .font(.largeTitle)
                        .foregroundStyle(.yellow)
                    
                    Image(systemName: "heart.fill")
                        .resizable()
                        .frame(width: 50, height: 50)
                        .foregroundStyle(.red)
                    
                    Image(systemName: "person.circle")
                        .font(.system(size: 60))
                        .foregroundStyle(.blue)
                }
                
                Divider()
                
                // 도형
                Group {
                    Circle()
                        .fill(.blue)
                        .frame(width: 80, height: 80)
                    
                    Rectangle()
                        .fill(.green)
                        .frame(width: 100, height: 60)
                    
                    RoundedRectangle(cornerRadius: 20)
                        .fill(.orange)
                        .frame(width: 100, height: 60)
                    
                    Capsule()
                        .fill(.purple)
                        .frame(width: 100, height: 40)
                }
            }
            .padding()
        }
    }
}

// MARK: - 2. 프로필 카드 (완성 예시)
struct ProfileCardView: View {
    var body: some View {
        VStack(spacing: 16) {
            // 프로필 이미지
            ZStack {
                Circle()
                    .fill(.blue.opacity(0.2))
                    .frame(width: 100, height: 100)
                
                Image(systemName: "person.fill")
                    .font(.system(size: 50))
                    .foregroundStyle(.blue)
            }
            
            // 이름과 직함
            VStack(spacing: 4) {
                Text("홍길동")
                    .font(.title2)
                    .bold()
                
                Text("iOS Developer")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            
            // 통계
            HStack(spacing: 30) {
                VStack {
                    Text("128")
                        .font(.headline)
                        .bold()
                    Text("게시물")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                
                VStack {
                    Text("1.2K")
                        .font(.headline)
                        .bold()
                    Text("팔로워")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                
                VStack {
                    Text("456")
                        .font(.headline)
                        .bold()
                    Text("팔로잉")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.vertical, 8)
            
            // 버튼
            HStack(spacing: 12) {
                Button {
                    print("팔로우 클릭")
                } label: {
                    Text("팔로우")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                }
                .buttonStyle(.borderedProminent)
                
                Button {
                    print("메시지 클릭")
                } label: {
                    Text("메시지")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                }
                .buttonStyle(.bordered)
            }
        }
        .padding()
        .background(.gray.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .padding()
    }
}

// MARK: - 3. Modifier 순서 비교
struct ModifierOrderView: View {
    var body: some View {
        VStack(spacing: 40) {
            Text("Modifier 순서 비교")
                .font(.title)
                .bold()
            
            VStack(spacing: 20) {
                // 케이스1: padding → background
                VStack {
                    Text("Hello")
                        .padding()
                        .background(.blue)
                        .foregroundStyle(.white)
                    
                    Text("padding → background")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                
                // 케이스 2: background → padding
                VStack {
                    Text("Hello")
                        .background(.blue)
                        .padding()
                        .foregroundStyle(.white)
                    
                    Text("background → padding")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                
                Divider()
                // 케이스 3: frame → background
                VStack {
                    Text("Hello")
                        .frame(width: 150, height: 50)
                        .background(.green)
                        .foregroundStyle(.white)
                    
                    Text("frame → background")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                
                // 케이스 4: background → frame
                VStack {
                    Text("Hello")
                        .background(.green)
                        .frame(width: 150, height: 50)
                        .foregroundStyle(.white)
                    
                    Text("background → frame")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding()
    }
}

// MARK: - 4. Stack 연습
struct StackPracticeView: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 30) {
                // VStack 예시
                GroupBox("VStack - 세로 배치") {
                    VStack(spacing: 10) {
                        Circle().fill(.red).frame(width: 50, height: 50)
                        Circle().fill(.green).frame(width: 50, height: 50)
                        Circle().fill(.blue).frame(width: 50, height: 50)
                    }
                }
                
                // HStack 예시
                GroupBox("HStack - 가로 배치") {
                    HStack(spacing: 10) {
                        Circle().fill(.red).frame(width: 50, height: 50)
                        Circle().fill(.green).frame(width: 50, height: 50)
                        Circle().fill(.blue).frame(width: 50, height: 50)
                    }
                }
                
                // ZStack 예시
                GroupBox("ZStack - 겹쳐서 배치") {
                    ZStack {
                        Circle().fill(.red).frame(width: 80, height: 80)
                        Circle().fill(.green).frame(width: 60, height: 60)
                        Circle().fill(.blue).frame(width: 40, height: 40)
                    }
                }
                
                // 복합 레이아웃
                GroupBox("복합 레이아웃") {
                    HStack {
                        VStack {
                            Text("왼쪽")
                            Text("영역")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(.blue.opacity(0.2))
                        
                        VStack {
                            Text("오른쪽")
                            Text("영역")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(.green.opacity(0.2))
                    }
                }
            }
            .padding()
        }
    }
}

// MARK: - 5. Spacer와 Alignment 연습
struct SpacerAlignmentView: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 30) {
                // Spacer 예시
                GroupBox("Spacer 사용") {
                    VStack {
                        HStack {
                            Text("왼쪽")
                            Spacer()
                            Text("오른쪽")
                        }
                        .padding()
                        .background(.gray.opacity(0.1))
                        
                        HStack {
                            Text("왼쪽")
                            Text("중앙")
                            Spacer()
                            Text("오른쪽")
                        }
                        .padding()
                        .background(.gray.opacity(0.1))
                    }
                }
                
                // Alignment 예시
                GroupBox("Alignment - leading") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("짧은 텍스트")
                        Text("조금 더 긴 텍스트입니다")
                        Text("가장 긴 텍스트입니다 여러분")
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(.blue.opacity(0.1))
                }
                
                GroupBox("Alignment - center") {
                    VStack(alignment: .center, spacing: 8) {
                        Text("짧은 텍스트")
                        Text("조금 더 긴 텍스트입니다")
                        Text("가장 긴 텍스트입니다 여러분")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(.green.opacity(0.1))
                }
                
                GroupBox("Alignment - trailing") {
                    VStack(alignment: .trailing, spacing: 8) {
                        Text("짧은 텍스트")
                        Text("조금 더 긴 텍스트입니다")
                        Text("가장 긴 텍스트입니다 여러분")
                    }
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    .padding()
                    .background(.orange.opacity(0.1))
                }
            }
            .padding()
        }
    }
}

// MARK: - 6. 실전 연습: 명함 카드
struct BusinessCardView: View {
    var body: some View {
        ZStack {
            // 배경
            LinearGradient(
                colors: [.blue, .purple],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            
            // 내용
            VStack(spacing: 20) {
                Spacer()
                
                // 로고
                Image(systemName: "apple.logo")
                    .font(.system(size: 60))
                    .foregroundStyle(.white)
                
                // 정보
                VStack(spacing: 8) {
                    Text("홍길동")
                        .font(.title)
                        .bold()
                        .foregroundStyle(.white)
                    
                    Text("iOS Developer")
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.8))
                }
                
                Divider()
                    .background(.white)
                    .padding(.horizontal, 40)
                
                // 연락처
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: "envelope.fill")
                        Text("hong@example.com")
                    }
                    
                    HStack {
                        Image(systemName: "phone.fill")
                        Text("010-1234-5678")
                    }
                    
                    HStack {
                        Image(systemName: "link")
                        Text("github.com/hong")
                    }
                }
                .font(.caption)
                .foregroundStyle(.white)
                
                Spacer()
            }
        }
        .frame(width: 320, height: 200)
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .shadow(radius: 10)
    }
}

// MARK: - Preview
#Preview("기본 컴포넌트") {
    BasicComponentsView()
}

#Preview("프로필 카드") {
    ProfileCardView()
}

#Preview("Modifier 순서") {
    ModifierOrderView()
}

#Preview("Stack 연습") {
    StackPracticeView()
}

#Preview("Spacer & Alignment") {
    SpacerAlignmentView()
}

#Preview("명함 카드") {
    BusinessCardView()
        .padding()        
}               
```



## 2: 레이아웃과 Modifier 심화

### 2.1 학습 목표
- VStack, HStack, ZStack의 차이와 활용
- Spacer와 Divider 사용법
- Modifier 체이닝과 순서의 중요성

### 2.2 Stack 레이아웃
```swift
// VStack: 세로로 배치
VStack {
    Text("위")
    Text("아래")
}

// HStack: 가로로 배치
HStack {
    Text("왼쪽")
    Text("오른쪽")
}

// ZStack: 겹쳐서 배치 (z축)
ZStack {
    Circle()
        .fill(.blue)
    Text("앞")
        .foregroundStyle(.white)
}
```

### 2.3 Spacer와 Divider
```swift
HStack {
    Text("왼쪽")
    Spacer() // 공간을 최대한 차지
    Text("오른쪽")
}

VStack {
    Text("위")
    Divider() // 구분선
    Text("아래")
}
```

### 2.4 Alignment와 Spacing
```swift
// alignment: 정렬
VStack(alignment: .leading, spacing: 8) {
    Text("왼쪽 정렬")
    Text("이것도 왼쪽")
}

HStack(alignment: .top) {
    Text("위쪽")
    Text("정렬")
}
```

### 2.5 Modifier 순서의 중요성
```swift
// 순서가 다르면 결과가 다름!

// 예시 1: padding 먼저, 배경색 나중
Text("Hello")
    .padding()        // 1. 패딩 추가
    .background(.blue) // 2. 배경색 (패딩 포함)

// 예시 2: 배경색 먼저, padding 나중
Text("Hello")
    .background(.blue) // 1. 배경색 (텍스트만)
    .padding()        // 2. 패딩 추가
```

**핵심 포인트**: Modifier는 위에서 아래로 순차 적용됨

1\.실습 : SNS 게시물 카드
```swift
struct PostCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // 헤더 (프로필)
            HStack {
                Image(systemName: "person.circle.fill")
                    .resizable()
                    .frame(width: 40, height: 40)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text("홍길동")
                        .font(.headline)
                    Text("2시간 전")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                
                Spacer()
                
                Button {
                    // 더보기 액션
                } label: {
                    Image(systemName: "ellipsis")
                }
            }
            
            // 본문
            Text("오늘 날씨가 정말 좋네요! 🌞")
                .font(.body)
            
            // 이미지
            Rectangle()
                .fill(.blue.opacity(0.3))
                .frame(height: 200)
                .overlay {
                    Image(systemName: "photo")
                        .font(.system(size: 50))
                        .foregroundStyle(.gray)
                }
                .clipShape(RoundedRectangle(cornerRadius: 10))
            
            Divider()
            
            // 액션 버튼
            HStack(spacing: 20) {
                Button {
                    // 좋아요
                } label: {
                    Label("42", systemImage: "heart")
                }
                
                Button {
                    // 댓글
                } label: {
                    Label("5", systemImage: "bubble.right")
                }
                
                Button {
                    // 공유
                } label: {
                    Label("공유", systemImage: "square.and.arrow.up")
                }
                
                Spacer()
            }
            .foregroundStyle(.secondary)
        }
        .padding()
        .background(.white)
        .clipShape(RoundedRectangle(cornerRadius: 15))
        .shadow(radius: 2)
    }
}
```

2\.과제
1. 날씨 정보 카드 만들기 (온도, 날씨 아이콘, 시간별 정보)
2. 음악 플레이어 UI 만들기 (앨범 커버, 제목, 컨트롤 버튼)
3. Modifier 순서를 바꿔가며 결과 차이 관찰하기




