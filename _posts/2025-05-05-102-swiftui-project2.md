---
title: 7차시 12:SwiftUI Project2
layout: single
classes: wide
categories:
  - SwiftUI
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

### **5회차: 고급 UI & 애니메이션**
- searchable modifier로 검색 기능
- 복합 필터링 로직 구현
- withAnimation으로 부드러운 전환
- symbolEffect로 SF Symbols 애니메이션
- ContentUnavailableView 활용
- Alert와 confirmation dialog

```swift
import SwiftUI

// MARK: - 고급 리스트 뷰 (검색 & 필터링)
struct BookmarkListAdvancedView: View {
    @Environment(BookmarkStore.self) private var store
    @State private var searchText = ""
    @State private var selectedCategory: Bookmark.Category?
    @State private var showFavoritesOnly = false
    @State private var showingAddSheet = false
    @State private var newBookmark = Bookmark(title: "", url: "", category: .etc)
    
    var filteredBookmarks: [Bookmark] {
        var result = store.bookmarks
        
        // 검색 필터
        if !searchText.isEmpty {
            result = result.filter { bookmark in
                bookmark.title.localizedCaseInsensitiveContains(searchText) ||
                bookmark.url.localizedCaseInsensitiveContains(searchText)
            }
        }
        
        // 카테고리 필터
        if let category = selectedCategory {
            result = result.filter { $0.category == category }
        }
        
        // 즐겨찾기 필터
        if showFavoritesOnly {
            result = result.filter { $0.isFavorite }
        }
        
        return result
    }
    
    var body: some View {
        NavigationStack {
            List {
                ForEach(filteredBookmarks) { bookmark in
                    NavigationLink {
                        BookmarkDetailAnimatedView(bookmark: bookmark)
                    } label: {
                        BookmarkRowAnimated(bookmark: bookmark)
                    }
                    .swipeActions(edge: .leading) {
                        Button {
                            withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                                store.toggleFavorite(bookmark)
                            }
                        } label: {
                            Label("즐겨찾기", systemImage: bookmark.isFavorite ? "star.slash" : "star.fill")
                        }
                        .tint(.yellow)
                    }
                    .swipeActions(edge: .trailing) {
                        Button(role: .destructive) {
                            withAnimation {
                                store.delete(bookmark)
                            }
                        } label: {
                            Label("삭제", systemImage: "trash")
                        }
                    }
                }
            }
            .searchable(text: $searchText, prompt: "북마크 검색")
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
                    Menu {
                        // 즐겨찾기 필터
                        Button {
                            withAnimation {
                                showFavoritesOnly.toggle()
                            }
                        } label: {
                            Label(
                                showFavoritesOnly ? "전체 보기" : "즐겨찾기만",
                                systemImage: showFavoritesOnly ? "star.slash" : "star.fill"
                            )
                        }
                        
                        Divider()
                        
                        // 카테고리 필터
                        Menu("카테고리") {
                            Button("전체") {
                                withAnimation {
                                    selectedCategory = nil
                                }
                            }
                            
                            ForEach(Bookmark.Category.allCases, id: \.self) { category in
                                Button {
                                    withAnimation {
                                        selectedCategory = category
                                    }
                                } label: {
                                    Label(category.rawValue, systemImage: category.icon)
                                }
                            }
                        }
                    } label: {
                        Image(systemName: "line.3.horizontal.decrease.circle")
                    }
                }
            }
            .overlay {
                if filteredBookmarks.isEmpty {
                    ContentUnavailableView(
                        "북마크가 없습니다",
                        systemImage: "bookmark.slash",
                        description: Text(searchText.isEmpty ? "새 북마크를 추가해보세요" : "검색 결과가 없습니다")
                    )
                }
            }
            .sheet(isPresented: $showingAddSheet) {
                BookmarkEditViewV3(bookmark: $newBookmark, mode: .add)
                    .onDisappear {
                        if !newBookmark.title.isEmpty && !newBookmark.url.isEmpty {
                            withAnimation {
                                store.add(newBookmark)
                            }
                        }
                    }
            }
        }
    }
}

// MARK: - 애니메이션 적용된 Row
struct BookmarkRowAnimated: View {
    let bookmark: Bookmark
    @State private var isPressed = false
    
    var body: some View {
        HStack {
            Image(systemName: bookmark.category.icon)
                .foregroundStyle(.blue)
                .font(.title3)
                .frame(width: 40, height: 40)
                .background(.blue.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .scaleEffect(isPressed ? 0.9 : 1.0)
            
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
                    .symbolEffect(.bounce, value: bookmark.isFavorite)
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onLongPressGesture(minimumDuration: 0.1) {
            // 길게 누르기 완료
        } onPressingChanged: { pressing in
            withAnimation(.easeInOut(duration: 0.1)) {
                isPressed = pressing
            }
        }
    }
}

// MARK: - 애니메이션 적용된 상세 화면
struct BookmarkDetailAnimatedView: View {
    @Environment(BookmarkStore.self) private var store
    let bookmark: Bookmark
    @State private var showingEditSheet = false
    @State private var editingBookmark: Bookmark?
    @State private var isAnimating = false
    @State private var showDeleteAlert = false
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // 애니메이션 헤더
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: bookmark.category.icon)
                            .font(.system(size: 40))
                            .foregroundStyle(.blue)
                            .rotationEffect(.degrees(isAnimating ? 360 : 0))
                            .animation(.spring(response: 0.6, dampingFraction: 0.6), value: isAnimating)
                        
                        Spacer()
                        
                        if bookmark.isFavorite {
                            Image(systemName: "star.fill")
                                .font(.title2)
                                .foregroundStyle(.yellow)
                                .symbolEffect(.pulse, value: bookmark.isFavorite)
                        }
                    }
                    
                    Text(bookmark.title)
                        .font(.title)
                        .bold()
                }
                .padding(.horizontal)
                .opacity(isAnimating ? 1 : 0)
                .offset(y: isAnimating ? 0 : -20)
                
                Divider()
                    .padding(.horizontal)
                
                // 정보 섹션
                VStack(alignment: .leading, spacing: 12) {
                    InfoRowAnimated(label: "URL", value: bookmark.url, delay: 0.1)
                    InfoRowAnimated(label: "카테고리", value: bookmark.category.rawValue, delay: 0.2)
                    InfoRowAnimated(label: "생성일", value: bookmark.createdAt.formatted(date: .long, time: .omitted), delay: 0.3)
                }
                .padding(.horizontal)
                
                if !bookmark.notes.isEmpty {
                    Divider()
                        .padding(.horizontal)
                    
                    VStack(alignment: .leading, spacing: 8) {
                        Text("메모")
                            .font(.headline)
                        
                        Text(bookmark.notes)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.horizontal)
                    .opacity(isAnimating ? 1 : 0)
                    .offset(y: isAnimating ? 0 : -20)
                    .animation(.easeOut(duration: 0.5).delay(0.4), value: isAnimating)
                }
                
                Spacer(minLength: 20)
                
                // 버튼 그룹
                VStack(spacing: 12) {
                    // 웹사이트 열기 버튼
                    if let url = URL(string: bookmark.url) {
                        Link(destination: url) {
                            HStack {
                                Image(systemName: "safari")
                                Text("웹사이트 열기")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(.blue)
                            .foregroundStyle(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                        }
                        .buttonStyle(.plain)
                    }
                    
                    // 즐겨찾기 토글 버튼
                    Button {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                            store.toggleFavorite(bookmark)
                        }
                    } label: {
                        HStack {
                            Image(systemName: bookmark.isFavorite ? "star.slash" : "star.fill")
                            Text(bookmark.isFavorite ? "즐겨찾기 해제" : "즐겨찾기 추가")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(bookmark.isFavorite ? .gray.opacity(0.2) : .yellow.opacity(0.2))
                        .foregroundStyle(bookmark.isFavorite ? .gray : .yellow)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                    
                    // 삭제 버튼
                    Button(role: .destructive) {
                        showDeleteAlert = true
                    } label: {
                        HStack {
                            Image(systemName: "trash")
                            Text("삭제")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(.red.opacity(0.1))
                        .foregroundStyle(.red)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                }
                .padding(.horizontal)
                .opacity(isAnimating ? 1 : 0)
                .offset(y: isAnimating ? 0 : 20)
                .animation(.easeOut(duration: 0.5).delay(0.5), value: isAnimating)
            }
            .padding(.vertical)
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
                        store.update(editingBookmark)
                    }
            }
        }
        .alert("북마크 삭제", isPresented: $showDeleteAlert) {
            Button("취소", role: .cancel) { }
            Button("삭제", role: .destructive) {
                withAnimation {
                    store.delete(bookmark)
                }
                dismiss()
            }
        } message: {
            Text("이 북마크를 삭제하시겠습니까?")
        }
        .onAppear {
            withAnimation(.easeOut(duration: 0.5)) {
                isAnimating = true
            }
        }
    }
}

// MARK: - 애니메이션 정보 Row
struct InfoRowAnimated: View {
    let label: String
    let value: String
    let delay: Double
    @State private var isVisible = false
    
    var body: some View {
        HStack {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .bold()
        }
        .opacity(isVisible ? 1 : 0)
        .offset(x: isVisible ? 0 : -20)
        .onAppear {
            withAnimation(.easeOut(duration: 0.5).delay(delay)) {
                isVisible = true
            }
        }
    }
}

// MARK: - 통계 대시보드 뷰
struct BookmarkStatisticsView: View {
    @Environment(BookmarkStore.self) private var store
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // 전체 통계
                VStack(alignment: .leading, spacing: 12) {
                    Text("전체 통계")
                        .font(.headline)
                    
                    HStack(spacing: 16) {
                        StatCard(
                            icon: "bookmark.fill",
                            title: "전체",
                            value: "\(store.bookmarks.count)",
                            color: .blue
                        )
                        
                        StatCard(
                            icon: "star.fill",
                            title: "즐겨찾기",
                            value: "\(store.favoriteBookmarks().count)",
                            color: .yellow
                        )
                    }
                }
                
                // 카테고리별 통계
                VStack(alignment: .leading, spacing: 12) {
                    Text("카테고리별")
                        .font(.headline)
                    
                    ForEach(Bookmark.Category.allCases, id: \.self) { category in
                        let count = store.bookmarks(for: category).count
                        if count > 0 {
                            CategoryStatRow(category: category, count: count)
                        }
                    }
                }
            }
            .padding()
        }
        .navigationTitle("통계")
    }
}

// MARK: - 카테고리 통계 Row
struct CategoryStatRow: View {
    let category: Bookmark.Category
    let count: Int
    @State private var animateBar = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(category.rawValue, systemImage: category.icon)
                    .foregroundStyle(.primary)
                
                Spacer()
                
                Text("\(count)")
                    .font(.title3)
                    .bold()
                    .foregroundStyle(.blue)
            }
            
            // 프로그레스 바
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(.gray.opacity(0.2))
                        .frame(height: 6)
                    
                    Rectangle()
                        .fill(.blue)
                        .frame(
                            width: animateBar ? geometry.size.width * CGFloat(count) / 10 : 0,
                            height: 6
                        )
                }
                .clipShape(Capsule())
            }
            .frame(height: 6)
        }
        .padding()
        .background(.blue.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .onAppear {
            withAnimation(.easeOut(duration: 0.8)) {
                animateBar = true
            }
        }
    }
}

// MARK: - 통계 카드
struct StatCard: View {
    let icon: String
    let title: String
    let value: String
    let color: Color
    @State private var scale: CGFloat = 0.8
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title)
                .foregroundStyle(color)
            
            Text(value)
                .font(.title)
                .bold()
            
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(color.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .scaleEffect(scale)
        .onAppear {
            withAnimation(.spring(response: 0.6, dampingFraction: 0.7)) {
                scale = 1.0
            }
        }
    }
}

// MARK: - 제스처 연습 뷰
struct GestureExampleView: View {
    @State private var offset = CGSize.zero
    @State private var isDragging = false
    @State private var rotation: Double = 0
    @State private var scale: CGFloat = 1.0
    
    var body: some View {
        VStack(spacing: 40) {
            Text("제스처 연습")
                .font(.title)
                .bold()
            
            // 드래그 제스처
            RoundedRectangle(cornerRadius: 20)
                .fill(.blue)
                .frame(width: 150, height: 150)
                .overlay {
                    Text("드래그")
                        .foregroundStyle(.white)
                        .bold()
                }
                .offset(offset)
                .scaleEffect(isDragging ? 1.2 : 1.0)
                .gesture(
                    DragGesture()
                        .onChanged { value in
                            offset = value.translation
                            isDragging = true
                        }
                        .onEnded { _ in
                            withAnimation(.spring()) {
                                offset = .zero
                                isDragging = false
                            }
                        }
                )
            
            // 회전 & 확대 제스처
            RoundedRectangle(cornerRadius: 20)
                .fill(.green)
                .frame(width: 150, height: 150)
                .overlay {
                    Text("회전 & 확대")
                        .foregroundStyle(.white)
                        .bold()
                }
                .rotationEffect(.degrees(rotation))
                .scaleEffect(scale)
                .gesture(
                    RotationGesture()
                        .onChanged { value in
                            rotation = value.degrees
                        }
                        .onEnded { _ in
                            withAnimation(.spring()) {
                                rotation = 0
                            }
                        }
                        .simultaneously(with: MagnificationGesture()
                            .onChanged { value in
                                scale = value
                            }
                            .onEnded { _ in
                                withAnimation(.spring()) {
                                    scale = 1.0
                                }
                            }
                        )
                )
        }
        .padding()
    }
}

// MARK: - Preview
#Preview("Advanced List") {
    BookmarkListAdvancedView()
        .environment(BookmarkStore())
}

#Preview("Statistics") {
    NavigationStack {
        BookmarkStatisticsView()
            .environment(BookmarkStore())
    }
}

#Preview("Gestures") {
    GestureExampleView()
}
```
