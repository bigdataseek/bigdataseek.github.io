---
title: 7차시 12:SwiftUI Project
layout: single
classes: wide
categories:
  - SwiftUI
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

# SwiftUI 앱 제작 

Apple Landmarks 튜토리얼을 완료한 학생들을 위한 체계적인 SwiftUI 앱 제작 커리큘럼입니다. 실전 프로젝트로 "북마크 관리 앱"을 만들어보겠습니다.


## 1.상세 커리큘럼 및 소스 코드

### **1회차: 프로젝트 설계 & 데이터 모델링****학습 목표:**
- Identifiable, Codable 프로토콜 이해
- 중첩 열거형(nested enum) 활용
- 샘플 데이터로 테스트 환경 구축

```swift
import Foundation

// MARK: - Bookmark 모델
struct Bookmark: Identifiable, Codable, Hashable {
    var id = UUID()
    var title: String
    var url: String
    var category: Category
    var isFavorite: Bool = false
    var notes: String = ""
    var createdAt: Date = Date()
    
    // 카테고리 열거형
    enum Category: String, Codable, CaseIterable {
        case work = "업무"
        case study = "공부"
        case entertainment = "엔터테인먼트"
        case shopping = "쇼핑"
        case etc = "기타"
        
        var icon: String {
            switch self {
            case .work: return "briefcase.fill"
            case .study: return "book.fill"
            case .entertainment: return "film.fill"
            case .shopping: return "cart.fill"
            case .etc: return "folder.fill"
            }
        }
    }
}

// MARK: - 샘플 데이터
extension Bookmark {
    static let sampleData: [Bookmark] = [
        Bookmark(
            title: "Apple Developer",
            url: "https://developer.apple.com",
            category: .study,
            isFavorite: true,
            notes: "SwiftUI 공식 문서"
        ),
        Bookmark(
            title: "GitHub",
            url: "https://github.com",
            category: .work,
            isFavorite: true,
            notes: "코드 저장소"
        ),
        Bookmark(
            title: "YouTube",
            url: "https://youtube.com",
            category: .entertainment,
            notes: "동영상 플랫폼"
        ),
        Bookmark(
            title: "쿠팡",
            url: "https://coupang.com",
            category: .shopping
        ),
        Bookmark(
            title: "Hacking with Swift",
            url: "https://hackingwithswift.com",
            category: .study,
            isFavorite: true,
            notes: "SwiftUI 튜토리얼"
        )
    ]
}
```

### **2회차: 리스트 화면 & 상세 화면 구현****학습 목표:**
- NavigationStack과 NavigationLink 활용
- 컴포넌트 분리 및 재사용성
- List의 onDelete 제스처
- Link로 외부 URL 연결

```swift
import SwiftUI

// MARK: - 북마크 리스트 화면
struct BookmarkListView: View {
    @State private var bookmarks = Bookmark.sampleData
    @State private var showingAddSheet = false
    
    var body: some View {
        NavigationStack {
            List {
                ForEach(bookmarks) { bookmark in
                    NavigationLink {
                        BookmarkDetailView(bookmark: bookmark)
                    } label: {
                        BookmarkRow(bookmark: bookmark)
                    }
                }
                .onDelete(perform: deleteBookmarks)
            }
            .navigationTitle("북마크")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showingAddSheet = true
                    } label: {
                        Image(systemName: "plus")
                    }
                }
                ToolbarItem(placement: .topBarLeading) {
                    EditButton()
                }
            }
        }
    }
    
    func deleteBookmarks(at offsets: IndexSet) {
        bookmarks.remove(atOffsets: offsets)
    }
}

// MARK: - 북마크 Row 컴포넌트
struct BookmarkRow: View {
    let bookmark: Bookmark
    
    var body: some View {
        HStack {
            Image(systemName: bookmark.category.icon)
                .foregroundStyle(.blue)
                .frame(width: 30)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(bookmark.title)
                    .font(.headline)
                
                Text(bookmark.url)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            
            Spacer()
            
            if bookmark.isFavorite {
                Image(systemName: "star.fill")
                    .foregroundStyle(.yellow)
            }
        }
        .padding(.vertical, 4)
    }
}

// MARK: - 북마크 상세 화면
struct BookmarkDetailView: View {
    let bookmark: Bookmark
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // 헤더
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: bookmark.category.icon)
                            .font(.system(size: 40))
                            .foregroundStyle(.blue)
                        
                        Spacer()
                        
                        if bookmark.isFavorite {
                            Image(systemName: "star.fill")
                                .font(.title2)
                                .foregroundStyle(.yellow)
                        }
                    }
                    
                    Text(bookmark.title)
                        .font(.title)
                        .bold()
                }
                
                Divider()
                
                // 정보 섹션
                VStack(alignment: .leading, spacing: 12) {
                    InfoRow(label: "URL", value: bookmark.url)
                    InfoRow(label: "카테고리", value: bookmark.category.rawValue)
                    InfoRow(label: "생성일", value: bookmark.createdAt.formatted(date: .long, time: .omitted))
                }
                
                if !bookmark.notes.isEmpty {
                    Divider()
                    
                    VStack(alignment: .leading, spacing: 8) {
                        Text("메모")
                            .font(.headline)
                        
                        Text(bookmark.notes)
                            .foregroundStyle(.secondary)
                    }
                }
                
                Spacer()
                
                // 웹사이트 열기 버튼
                Link(destination: URL(string: bookmark.url)!) {
                    Label("웹사이트 열기", systemImage: "safari")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(.blue)
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                }
            }
            .padding()
        }
        .navigationTitle("상세 정보")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - 정보 Row 컴포넌트
struct InfoRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .bold()
        }
    }
}

// MARK: - Preview
#Preview {
    BookmarkListView()
}

#Preview("Detail") {
    NavigationStack {
        BookmarkDetailView(bookmark: Bookmark.sampleData[0])
    }
}
```

### **3회차: 데이터 추가/편집 기능****학습 목표:**
- Form과 다양한 입력 컴포넌트 활용
- @Binding을 통한 양방향 데이터 바인딩
- Sheet 모달 화면 구현
- swipeActions로 제스처 기반 액션

```swift
import SwiftUI

// MARK: - 북마크 추가/편집 화면
struct BookmarkEditView: View {
    @Environment(\.dismiss) var dismiss
    @Binding var bookmark: Bookmark
    
    var body: some View {
        NavigationStack {
            Form {
                Section("기본 정보") {
                    TextField("제목", text: $bookmark.title)
                    
                    TextField("URL", text: $bookmark.url)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.URL)
                    
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
                
                Section("메모") {
                    TextEditor(text: $bookmark.notes)
                        .frame(minHeight: 100)
                }
            }
            .navigationTitle("북마크 편집")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("취소") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .confirmationAction) {
                    Button("저장") {
                        dismiss()
                    }
                    .disabled(bookmark.title.isEmpty || bookmark.url.isEmpty)
                }
            }
        }
    }
}

// MARK: - 업데이트된 리스트 화면 (추가/편집 기능 포함)
struct BookmarkListViewV2: View {
    @State private var bookmarks = Bookmark.sampleData
    @State private var showingAddSheet = false
    @State private var editingBookmark: Bookmark?
    @State private var newBookmark = Bookmark(title: "", url: "", category: .etc)
    
    var body: some View {
        NavigationStack {
            List {
                ForEach(bookmarks) { bookmark in
                    NavigationLink {
                        BookmarkDetailViewV2(
                            bookmark: binding(for: bookmark)
                        )
                    } label: {
                        BookmarkRow(bookmark: bookmark)
                    }
                    .swipeActions(edge: .leading) {
                        Button {
                            toggleFavorite(bookmark)
                        } label: {
                            Label("즐겨찾기", systemImage: bookmark.isFavorite ? "star.slash" : "star.fill")
                        }
                        .tint(.yellow)
                    }
                }
                .onDelete(perform: deleteBookmarks)
            }
            .navigationTitle("북마크")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        newBookmark = Bookmark(title: "", url: "", category: .etc)
                        showingAddSheet = true
                    } label: {
                        Image(systemName: "plus")
                    }
                }
                ToolbarItem(placement: .topBarLeading) {
                    EditButton()
                }
            }
            .sheet(isPresented: $showingAddSheet) {
                BookmarkEditView(bookmark: $newBookmark)
                    .onDisappear {
                        if !newBookmark.title.isEmpty && !newBookmark.url.isEmpty {
                            bookmarks.append(newBookmark)
                        }
                    }
            }
        }
    }
    
    // 특정 북마크의 바인딩 생성
    private func binding(for bookmark: Bookmark) -> Binding<Bookmark> {
        guard let index = bookmarks.firstIndex(where: { $0.id == bookmark.id }) else {
            fatalError("북마크를 찾을 수 없습니다")
        }
        return $bookmarks[index]
    }
    
    func deleteBookmarks(at offsets: IndexSet) {
        bookmarks.remove(atOffsets: offsets)
    }
    
    func toggleFavorite(_ bookmark: Bookmark) {
        if let index = bookmarks.firstIndex(where: { $0.id == bookmark.id }) {
            bookmarks[index].isFavorite.toggle()
        }
    }
}

// MARK: - 업데이트된 상세 화면 (편집 기능 포함)
struct BookmarkDetailViewV2: View {
    @Binding var bookmark: Bookmark
    @State private var showingEditSheet = false
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // 헤더
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: bookmark.category.icon)
                            .font(.system(size: 40))
                            .foregroundStyle(.blue)
                        
                        Spacer()
                        
                        if bookmark.isFavorite {
                            Image(systemName: "star.fill")
                                .font(.title2)
                                .foregroundStyle(.yellow)
                        }
                    }
                    
                    Text(bookmark.title)
                        .font(.title)
                        .bold()
                }
                
                Divider()
                
                // 정보 섹션
                VStack(alignment: .leading, spacing: 12) {
                    InfoRow(label: "URL", value: bookmark.url)
                    InfoRow(label: "카테고리", value: bookmark.category.rawValue)
                    InfoRow(label: "생성일", value: bookmark.createdAt.formatted(date: .long, time: .omitted))
                }
                
                if !bookmark.notes.isEmpty {
                    Divider()
                    
                    VStack(alignment: .leading, spacing: 8) {
                        Text("메모")
                            .font(.headline)
                        
                        Text(bookmark.notes)
                            .foregroundStyle(.secondary)
                    }
                }
                
                Spacer()
                
                // 웹사이트 열기 버튼
                if let url = URL(string: bookmark.url) {
                    Link(destination: url) {
                        Label("웹사이트 열기", systemImage: "safari")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(.blue)
                            .foregroundStyle(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                }
            }
            .padding()
        }
        .navigationTitle("상세 정보")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            Button("편집") {
                showingEditSheet = true
            }
        }
        .sheet(isPresented: $showingEditSheet) {
            BookmarkEditView(bookmark: $bookmark)
        }
    }
}

// MARK: - Preview
#Preview("Edit View") {
    BookmarkEditView(bookmark: .constant(Bookmark.sampleData[0]))
}

#Preview("List with Edit") {
    BookmarkListViewV2()
}
```

### **4회차: 데이터 영속화 & 상태 관리****학습 목표:**
- @Observable 매크로를 통한 상태 관리
- Environment를 통한 전역 상태 공유
- JSON 파일로 데이터 영속화
- didSet 프로퍼티 옵저버 활용

```swift
import SwiftUI

// MARK: - 북마크 저장소 (@Observable 사용)
@Observable
class BookmarkStore {
    var bookmarks: [Bookmark] = [] {
        didSet {
            save()
        }
    }
    
    private let savePath = URL.documentsDirectory.appending(path: "bookmarks.json")
    
    init() {
        load()
    }
    
    // MARK: - CRUD 메서드
    
    func add(_ bookmark: Bookmark) {
        bookmarks.append(bookmark)
    }
    
    func update(_ bookmark: Bookmark) {
        if let index = bookmarks.firstIndex(where: { $0.id == bookmark.id }) {
            bookmarks[index] = bookmark
        }
    }
    
    func delete(_ bookmark: Bookmark) {
        bookmarks.removeAll { $0.id == bookmark.id }
    }
    
    func delete(at offsets: IndexSet) {
        bookmarks.remove(atOffsets: offsets)
    }
    
    func toggleFavorite(_ bookmark: Bookmark) {
        if let index = bookmarks.firstIndex(where: { $0.id == bookmark.id }) {
            bookmarks[index].isFavorite.toggle()
        }
    }
    
    // MARK: - 필터링 메서드
    
    func favoriteBookmarks() -> [Bookmark] {
        bookmarks.filter { $0.isFavorite }
    }
    
    func bookmarks(for category: Bookmark.Category) -> [Bookmark] {
        bookmarks.filter { $0.category == category }
    }
    
    func search(_ query: String) -> [Bookmark] {
        if query.isEmpty {
            return bookmarks
        }
        return bookmarks.filter { bookmark in
            bookmark.title.localizedCaseInsensitiveContains(query) ||
            bookmark.url.localizedCaseInsensitiveContains(query) ||
            bookmark.notes.localizedCaseInsensitiveContains(query)
        }
    }
    
    // MARK: - 데이터 저장/로드
    
    private func save() {
        do {
            let data = try JSONEncoder().encode(bookmarks)
            try data.write(to: savePath, options: [.atomic, .completeFileProtection])
        } catch {
            print("북마크 저장 실패: \(error.localizedDescription)")
        }
    }
    
    private func load() {
        do {
            let data = try Data(contentsOf: savePath)
            bookmarks = try JSONDecoder().decode([Bookmark].self, from: data)
        } catch {
            // 저장된 데이터가 없으면 샘플 데이터 사용
            bookmarks = Bookmark.sampleData
        }
    }
}

// MARK: - 앱 진입점 (App 파일)
@main
struct BookmarkApp: App {
    @State private var store = BookmarkStore()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(store)
        }
    }
}

// MARK: - 메인 컨텐츠 뷰
struct ContentView: View {
    @Environment(BookmarkStore.self) private var store
    @State private var showingAddSheet = false
    @State private var newBookmark = Bookmark(title: "", url: "", category: .etc)
    
    var body: some View {
        NavigationStack {
            List {
                if !store.favoriteBookmarks().isEmpty {
                    Section("즐겨찾기") {
                        ForEach(store.favoriteBookmarks()) { bookmark in
                            NavigationLink {
                                BookmarkDetailViewV3(bookmark: bookmark)
                            } label: {
                                BookmarkRow(bookmark: bookmark)
                            }
                        }
                    }
                }
                
                ForEach(Bookmark.Category.allCases, id: \.self) { category in
                    let categoryBookmarks = store.bookmarks(for: category)
                    if !categoryBookmarks.isEmpty {
                        Section {
                            ForEach(categoryBookmarks) { bookmark in
                                NavigationLink {
                                    BookmarkDetailViewV3(bookmark: bookmark)
                                } label: {
                                    BookmarkRow(bookmark: bookmark)
                                }
                                .swipeActions(edge: .leading) {
                                    Button {
                                        store.toggleFavorite(bookmark)
                                    } label: {
                                        Label("즐겨찾기", systemImage: bookmark.isFavorite ? "star.slash" : "star.fill")
                                    }
                                    .tint(.yellow)
                                }
                            }
                            .onDelete { offsets in
                                let bookmarksToDelete = offsets.map { categoryBookmarks[$0] }
                                bookmarksToDelete.forEach { store.delete($0) }
                            }
                        } header: {
                            Label(category.rawValue, systemImage: category.icon)
                        }
                    }
                }
            }
            .navigationTitle("북마크")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        newBookmark = Bookmark(title: "", url: "", category: .etc)
                        showingAddSheet = true
                    } label: {
                        Image(systemName: "plus")
                    }
                }
                ToolbarItem(placement: .topBarLeading) {
                    EditButton()
                }
            }
            .sheet(isPresented: $showingAddSheet) {
                BookmarkEditViewV3(bookmark: $newBookmark, mode: .add)
                    .onDisappear {
                        if !newBookmark.title.isEmpty && !newBookmark.url.isEmpty {
                            store.add(newBookmark)
                        }
                    }
            }
        }
    }
}

// MARK: - 상세 화면 V3 (Store 사용)
struct BookmarkDetailViewV3: View {
    @Environment(BookmarkStore.self) private var store
    let bookmark: Bookmark
    @State private var showingEditSheet = false
    @State private var editingBookmark: Bookmark?
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // 헤더
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: bookmark.category.icon)
                            .font(.system(size: 40))
                            .foregroundStyle(.blue)
                        
                        Spacer()
                        
                        if bookmark.isFavorite {
                            Image(systemName: "star.fill")
                                .font(.title2)
                                .foregroundStyle(.yellow)
                        }
                    }
                    
                    Text(bookmark.title)
                        .font(.title)
                        .bold()
                }
                
                Divider()
                
                // 정보 섹션
                VStack(alignment: .leading, spacing: 12) {
                    InfoRow(label: "URL", value: bookmark.url)
                    InfoRow(label: "카테고리", value: bookmark.category.rawValue)
                    InfoRow(label: "생성일", value: bookmark.createdAt.formatted(date: .long, time: .omitted))
                }
                
                if !bookmark.notes.isEmpty {
                    Divider()
                    
                    VStack(alignment: .leading, spacing: 8) {
                        Text("메모")
                            .font(.headline)
                        
                        Text(bookmark.notes)
                            .foregroundStyle(.secondary)
                    }
                }
                
                Spacer()
                
                // 웹사이트 열기 버튼
                if let url = URL(string: bookmark.url) {
                    Link(destination: url) {
                        Label("웹사이트 열기", systemImage: "safari")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(.blue)
                            .foregroundStyle(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                }
            }
            .padding()
        }
        .navigationTitle("상세 정보")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            Button("편집") {
                editingBookmark = bookmark
                showingEditSheet = true
            }
        }
        .sheet(isPresented: $showingEditSheet) {
            if let editingBookmark = editingBookmark {
                BookmarkEditViewV3(bookmark: .constant(editingBookmark), mode: .edit)
                    .onDisappear {
                        if let updated = editingBookmark {
                            store.update(updated)
                        }
                    }
            }
        }
    }
}

// MARK: - 편집 뷰 V3 (Add/Edit 모드)
struct BookmarkEditViewV3: View {
    @Environment(\.dismiss) var dismiss
    @Binding var bookmark: Bookmark
    let mode: Mode
    
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
                    
                    TextField("URL", text: $bookmark.url)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.URL)
                    
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
                
                Section("메모") {
                    TextEditor(text: $bookmark.notes)
                        .frame(minHeight: 100)
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
                        dismiss()
                    }
                    .disabled(bookmark.title.isEmpty || bookmark.url.isEmpty)
                }
            }
        }
    }
}

// MARK: - Preview
#Preview {
    ContentView()
        .environment(BookmarkStore())
}
```
