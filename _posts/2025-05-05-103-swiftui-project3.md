---
title: 7차시 12:SwiftUI Project3
layout: single
classes: wide
categories:
  - SwiftUI
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

### **6회차: 마무리 & 배포 준비****학습 목표:**
- TabView로 멀티 화면 구성
- @AppStorage로 사용자 설정 저장
- 에러 처리 및 유효성 검사
- 온보딩 화면 구현
- 런치 스크린 디자인

```swift
//완성된 앱 구조

import SwiftUI

// MARK: - 앱 진입점
@main
struct BookmarkManagerApp: App {
    @State private var store = BookmarkStore()
    
    var body: some Scene {
        WindowGroup {
            MainTabView()
                .environment(store)
        }
    }
}

// MARK: - 메인 탭 뷰
struct MainTabView: View {
    var body: some View {
        TabView {
            BookmarkListAdvancedView()
                .tabItem {
                    Label("북마크", systemImage: "bookmark.fill")
                }
            
            BookmarkStatisticsView()
                .tabItem {
                    Label("통계", systemImage: "chart.bar.fill")
                }
            
            SettingsView()
                .tabItem {
                    Label("설정", systemImage: "gearshape.fill")
                }
        }
    }
}

// MARK: - 설정 화면
struct SettingsView: View {
    @Environment(BookmarkStore.self) private var store
    @AppStorage("sortOrder") private var sortOrder = SortOrder.dateDescending
    @AppStorage("showCategoryIcons") private var showCategoryIcons = true
    @State private var showDeleteAlert = false
    
    enum SortOrder: String, CaseIterable {
        case dateAscending = "오래된 순"
        case dateDescending = "최신 순"
        case titleAscending = "이름 순 (A-Z)"
        case titleDescending = "이름 순 (Z-A)"
    }
    
    var body: some View {
        NavigationStack {
            Form {
                Section("표시 설정") {
                    Toggle("카테고리 아이콘 표시", isOn: $showCategoryIcons)
                    
                    Picker("정렬 순서", selection: $sortOrder) {
                        ForEach(SortOrder.allCases, id: \.self) { order in
                            Text(order.rawValue).tag(order)
                        }
                    }
                }
                
                Section("데이터") {
                    HStack {
                        Text("전체 북마크")
                        Spacer()
                        Text("\(store.bookmarks.count)개")
                            .foregroundStyle(.secondary)
                    }
                    
                    Button("샘플 데이터 추가") {
                        Bookmark.sampleData.forEach { bookmark in
                            store.add(bookmark)
                        }
                    }
                    
                    Button("모든 북마크 삭제", role: .destructive) {
                        showDeleteAlert = true
                    }
                }
                
                Section("정보") {
                    LabeledContent("버전", value: "1.0.0")
                    LabeledContent("개발자", value: "Your Name")
                    
                    Link("GitHub에서 보기", destination: URL(string: "https://github.com")!)
                }
            }
            .navigationTitle("설정")
            .alert("모든 북마크 삭제", isPresented: $showDeleteAlert) {
                Button("취소", role: .cancel) { }
                Button("삭제", role: .destructive) {
                    store.bookmarks.removeAll()
                }
            } message: {
                Text("모든 북마크가 삭제됩니다. 이 작업은 되돌릴 수 없습니다.")
            }
        }
    }
}

// MARK: - 에러 핸들링
enum BookmarkError: Error, LocalizedError {
    case invalidURL
    case saveFailed
    case loadFailed
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "유효하지 않은 URL입니다."
        case .saveFailed:
            return "북마크 저장에 실패했습니다."
        case .loadFailed:
            return "북마크를 불러오는데 실패했습니다."
        }
    }
}

// MARK: - URL 유효성 검사 유틸리티
extension String {
    var isValidURL: Bool {
        if let url = URL(string: self) {
            return UIApplication.shared.canOpenURL(url)
        }
        return false
    }
}

// MARK: - 개선된 편집 뷰 (에러 처리 포함)
struct BookmarkEditViewFinal: View {
    @Environment(\.dismiss) var dismiss
    @Binding var bookmark: Bookmark
    let mode: Mode
    
    @State private var showError = false
    @State private var errorMessage = ""
    
    enum Mode {
        case add, edit
        
        var title: String {
            switch self {
            case .add: return "북마크 추가"
            case .edit: return "북마크 편집"
            }
        }
    }
    
    var body: some View {
        NavigationStack {
            Form {
                Section("기본 정보") {
                    TextField("제목", text: $bookmark.title)
                        .autocorrectionDisabled()
                    
                    TextField("URL", text: $bookmark.url)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.URL)
                        .autocorrectionDisabled()
                    
                    if !bookmark.url.isEmpty && !bookmark.url.isValidURL {
                        Label("올바른 URL 형식이 아닙니다", systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.red)
                            .font(.caption)
                    }
                    
                    Picker("카테고리", selection: $bookmark.category) {
                        ForEach(Bookmark.Category.allCases, id: \.self) { category in
                            Label(category.rawValue, systemImage: category.icon)
                                .tag(category)
                        }
                    }
                }
                
                Section("옵션") {
                    Toggle("즐겨찾기", isOn: $bookmark.isFavorite)
                }
                
                Section {
                    TextEditor(text: $bookmark.notes)
                        .frame(minHeight: 100)
                } header: {
                    Text("메모")
                } footer: {
                    Text("\(bookmark.notes.count)자")
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle(mode.title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("취소") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .confirmationAction) {
                    Button("저장") {
                        if validateAndSave() {
                            dismiss()
                        }
                    }
                    .disabled(!isValid)
                }
            }
            .alert("오류", isPresented: $showError) {
                Button("확인", role: .cancel) { }
            } message: {
                Text(errorMessage)
            }
        }
    }
    
    var isValid: Bool {
        !bookmark.title.isEmpty && 
        !bookmark.url.isEmpty && 
        bookmark.url.isValidURL
    }
    
    func validateAndSave() -> Bool {
        if bookmark.title.trimmingCharacters(in: .whitespaces).isEmpty {
            errorMessage = "제목을 입력해주세요."
            showError = true
            return false
        }
        
        if !bookmark.url.isValidURL {
            errorMessage = "올바른 URL을 입력해주세요."
            showError = true
            return false
        }
        
        return true
    }
}

// MARK: - 런치 스크린용 뷰
struct LaunchScreenView: View {
    @State private var isAnimating = false
    
    var body: some View {
        ZStack {
            Color.blue.ignoresSafeArea()
            
            VStack(spacing: 20) {
                Image(systemName: "bookmark.fill")
                    .font(.system(size: 80))
                    .foregroundStyle(.white)
                    .scaleEffect(isAnimating ? 1.2 : 1.0)
                    .animation(.easeInOut(duration: 1.0).repeatForever(autoreverses: true), value: isAnimating)
                
                Text("북마크 관리")
                    .font(.title)
                    .bold()
                    .foregroundStyle(.white)
            }
        }
        .onAppear {
            isAnimating = true
        }
    }
}

// MARK: - 온보딩 뷰
struct OnboardingView: View {
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false
    @State private var currentPage = 0
    
    let pages = [
        OnboardingPage(
            icon: "bookmark.fill",
            title: "북마크 관리",
            description: "자주 방문하는 웹사이트를 한 곳에서 관리하세요"
        ),
        OnboardingPage(
            icon: "star.fill",
            title: "즐겨찾기",
            description: "중요한 북마크를 즐겨찾기로 표시하세요"
        ),
        OnboardingPage(
            icon: "folder.fill",
            title: "카테고리 정리",
            description: "북마크를 카테고리별로 체계적으로 정리하세요"
        )
    ]
    
    var body: some View {
        ZStack {
            TabView(selection: $currentPage) {
                ForEach(0..<pages.count, id: \.self) { index in
                    VStack(spacing: 30) {
                        Spacer()
                        
                        Image(systemName: pages[index].icon)
                            .font(.system(size: 100))
                            .foregroundStyle(.blue)
                        
                        Text(pages[index].title)
                            .font(.title)
                            .bold()
                        
                        Text(pages[index].description)
                            .font(.body)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 40)
                        
                        Spacer()
                        
                        if index == pages.count - 1 {
                            Button {
                                hasCompletedOnboarding = true
                            } label: {
                                Text("시작하기")
                                    .font(.headline)
                                    .foregroundStyle(.white)
                                    .frame(maxWidth: .infinity)
                                    .padding()
                                    .background(.blue)
                                    .clipShape(RoundedRectangle(cornerRadius: 12))
                            }
                            .padding(.horizontal, 40)
                        }
                        
                        Spacer()
                    }
                    .tag(index)
                }
            }
            .tabViewStyle(.page)
            .indexViewStyle(.page(backgroundDisplayMode: .always))
        }
    }
}

struct OnboardingPage {
    let icon: String
    let title: String
    let description: String
}

// MARK: - 앱 래퍼 (온보딩 포함)
struct AppWrapperView: View {
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false
    
    var body: some View {
        if hasCompletedOnboarding {
            MainTabView()
        } else {
            OnboardingView()
        }
    }
}

// MARK: - Preview
#Preview("Main Tab") {
    MainTabView()
        .environment(BookmarkStore())
}

#Preview("Settings") {
    NavigationStack {
        SettingsView()
            .environment(BookmarkStore())
    }
}

#Preview("Onboarding") {
    OnboardingView()
}

#Preview("Launch Screen") {
    LaunchScreenView()
}
```

## **2. 배포 체크리스트**

## 3. 회차별 과제

각 회차마다 학생들이 직접 구현해볼 수 있는 과제를 추가로 제시합니다:

### **1회차 과제**
- 자신만의 데이터 모델 추가 속성 설계 (예: 방문 횟수, 마지막 방문일)
- 새로운 카테고리 3개 추가하기

### **2회차 과제**
- 리스트 항목에 스와이프 제스처 추가
- 상세 화면에 공유 기능 추가

### **3회차 과제**
- URL 자동 완성 기능 구현
- 북마크 복사 기능 추가

### **4회차 과제**
- iCloud 동기화 구현 (선택)
- 즐겨찾기 순서 변경 기능

### **5회차 과제**
- 태그 기능 추가
- 고급 검색 (태그, 날짜 범위)

### **6회차 과제**
- 위젯 구현
- 앱 아이콘 디자인 및 적용


## 4. 심화 학습 주제
기본 커리큘럼을 완료한 후 도전해볼 수 있는 추가 주제들:

1. **WidgetKit**: 홈 화면 위젯 구현
2. **App Intents**: Siri 단축어 통합
3. **CloudKit**: 클라우드 동기화
4. **Core Data**: 더 복잡한 데이터 관리
5. **StoreKit**: 인앱 결제 (프리미엄 기능)
6. **ShareSheet**: 다른 앱과 공유
7. **Safari Extension**: 사파리에서 바로 북마크 추가
8. **SwiftData**: iOS 17+ 새로운 데이터 프레임워크
