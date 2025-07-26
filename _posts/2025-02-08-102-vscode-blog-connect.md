---
title: 2차시 5:VScode에서 Blog 수정하기
layout: single
classes: wide
permalink: /vscode-blog-connect/
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---
Minimal Mistakes 테마를 사용하여 블로그를 운영시 VS Code와 연동해 편리하게 관리하려면 다음과 같은 설정 절차를 따라야 합니다. 

## 1. Git 계정 생성 및 관련 소스 Fork하기
1. **Git 계정 생성하기**
    <https://github.com> 에서 SingUp하기 (개발자 전용 Gmail사용하기)
    

2. **관련 블로그 테마 Fork하기**
    - minimal mistakes 테마
    - 수정된 샘플 테마인 <https://github.com/bigdataseek/bigdataseek.github.io> 로 이동
    - 우상단의 Fork 클릭
    - repostory명 입력란에 본인 git의 username 입력하기(username이 marbledata면 marbledata로 입력)
    - username.github.io의 Settings > Pages > branch를 none에서 main/master 변경 저장.

3. **Git 설치**
   - [Git for Windows](https://git-scm.com/download/win)를 설치합니다.
   - 설치 후 Git Bash를 사용해 기본 명령어를 실행할 수 있습니다.

4. **Git Bash 실행하여 본인 repository clone**
    - 시작 메뉴에서 "Git Bash"를 검색하여 실행합니다.
    - 본인 repository clone
    ```bash
     git clone https://github.com/본인username/본인username.github.io
     ```
    


## 2. **환경 설정 준비**
1. **Ruby 및 Bundler 설치**
   Minimal Mistakes는 Jekyll 기반으로 동작하므로 Ruby가 필요합니다.
   - [RubyInstaller](https://rubyinstaller.org/)를 통해 Windows용 Ruby를 설치합니다.
     - 설치 시 "Add Ruby executables to your PATH" 옵션을 체크하세요.
     - 설치 완료 후, `ridk install`을 실행해 필수 구성 요소를 설치합니다.
   - 설치 확인:
     ```bash
     ruby -v
     gem -v
     ```

2. **Jekyll 및 Bundler 설치**
   - RubyGems로 Jekyll과 Bundler를 설치합니다.
     ```bash
     gem install jekyll bundler
     ```

4. **VS Code 설치**
   - [VS Code](https://code.visualstudio.com/)를 설치하기



## 3. **VS Code와 연동**
1. **프로젝트 열기**
   - VS Code에서 `File > Open Folder`를 선택하고 블로그 프로젝트 폴더를 엽니다.

2. **Live Server 설정**

    - Gemfile 업데이트:
    ```bash
     bundle install
     ```

    - Jekyll 개발 서버를 실행합니다:
     ```bash
     bundle exec jekyll serve
     ```
   - 브라우저에서 `http://localhost:4000`로 블로그를 확인할 수 있습니다.
   - 소스에 변경사항 있으면 개발 서버는 자동으로 refresh되므로 단순히 브라우저만 refresh
   - _config.yml 수정시 Ctl + c로 강제종료 후, 재실행하고 브라우저 refresh

3. **VS Code 터미널 설정**
   - VS Code 내장 터미널에서 Jekyll 명령어를 실행하려면 `bash` 쉘을 사용하는 것이 좋습니다.
     - `Ctrl + Shift + P` > `Select Default Shell` > `Git Bash` 선택.

4. **자동화된 편집 경험**
   - VS Code에서 Ruby 관련 확장 프로그램과 Markdown 관련 플러그인을 설치해 개발 경험을 향상시킵니다.
   - 주요 확장 추천:
        - `Markdown All in One`, 
        - `Prettier - Code formatter`



## 4. **테마 커스터마이징**
Minimal Mistakes 테마는 `_config.yml`,`navigation.yml`와 `_posts`, `_pages` 디렉토리를 수정하여 커스터마이징할 수 있습니다.

1. **_config.yml**
  - 로고 및 이미지는 `assets/images/`에 이미지를 추가한 뒤 `_config.yml`에서 참조
  - 개인정보 관련 문구 수정

2. **블로그 작성**
  - _posts 폴더에 생성
  - 파일 생성시 파일명은 년월일 작성 후 간단한 타이틀명 기재.이때 -(대쉬)사용
    (2025-02-07-standard-blog.md)
  - md 파일 최상단에 Front matter 기재(일종의 파일정보), ---(대쉬 세개로 시작, 종료)
    
    ```
    ---    
    title: 블로그 작성하기  
    layout: single
    classes: wide
    categories:
    - Data Analysis
    tags:
    - GenAI
    ---
    ```

    - 블로그 작성에 필요한 2가지
        - 글작성
        ```
            #
            **
            -
            *
            (```)백틱 셋으로 열고, 닫기는 코드소스 표시
            >
            (---)대쉬 셋은 문단을 나누는 경계선,위아래는 빈칸 유지해야
            <> 자동링크
            [VS Code](링크주소)
        ```

        - 이미지 및 동영상 첨부

            ```
            <!-- 이미지를 삽입하려면 assets/images에 이미지를 넣고 이미지명을 이곳에 기재 -->
            
            ![Unsplash image 10]({%raw%}{{ site.url }}{{ site.baseurl }}/assets/images/unsplash-image-10.jpg{%endraw%})
            ```

            ```
            <!-- 이미지를 삽입하고 Size 조정시-->
            <img src="{%raw%}{{ site.url }}{{ site.baseurl }}/assets/images/dawithgenai2.jpg{%endraw%}" alt="dawithgenai2" width="300">
            ```

            ```
            <!-- Youtube 동영상을 삽입하려면 id번호만 기재 -->
            {% raw %}{% include video id="T6z-0dpXPvU" provider="youtube" %}{%endraw%}
            ```

            ```
            <!-- 자체 동영상을 삽입하려면 -->
            <video controls width="640" height="360">
              <source src="{%raw%}{{ site.url }}{{ site.baseurl }}/assets/videos/sample_video.mp4" type="video/mp4{%endraw%}">
              Your browser does not support the video tag.
            </video>
            ```

3. **정적 페이지 작성**
  - _pages폴더에 파일 생성
  - 파일생성시 영문으로 단순하게 작성(about.md). 
  - 파일명을  permalink로
    (home 메인화면에서 클릭시 호출, _data폴더의 navigation.yml에서 정의)

        ```
        - title: "About Me"
            url: /about/
        ```

    - md 파일 최상단에 Front matter 기재(일종의 파일정보),---(대쉬 세개로 시작, 종료)
    ```
    ---    
    title: 
    layout: single
    permalink: /about/
    ---
    ```

## 5. **수정사항 Commit**
  - git의 세가지 명령어(add, commit, push), vscode의 git bash에서
  ```
    git add .
    git commit -m "변경사항 기재"
    git push
  ```
  
  - push 전에 공용 컴퓨터인 경우 (프로젝트 폴더내 local로 환경설정), vscode의 git bash에서 
  ```
    git config --local user.name "본인 github의 username" 
    git config --local user.email "본인 github에 등록시 이메일"
  ```
  - git push시 다음과 같은 에러 발생하는 경우
    ```
    error: RPC failed; HTTP 400 curl 22 The requested URL returned error: 400
    ```
    다음과 같이 처리
    ```
    git config http.postBuffer 10485760
    ```

  - push 완료 후 github에서 변경사항을 자동으로 실제 웹 블로그에 적용(40초~ 1분정도 소요)

## 6.  **공용 컴퓨터에서 GitHub CLI(`gh`)로 GitHub 로그인**
- Windows에서 설치하려면 공식 GitHub CLI 페이지 에서 다운로드
  - 사용 환경: Git Bash, PowerShell, CMD 등 다양한 터미널에서 실행 가능
- VS Code 내장 터미널에서: 다음 명령어로 새 사용자가 로그인하면 새 사용자 활성화 됨.
  ```bash 
  gh auth login
  ```

- 현재 활성화된 계정을 확인하려면 다음 명령어를 사용하세요:
  ```bash
  gh auth status
  ```

- **이전 사용자로 계정 전환**
  ```bash
  gh auth switch
  ```

- **로그아웃**
  ```bash
  gh auth logout
  ```

- **모든 계정을 로그아웃**
  ```bash
  gh auth logout --hostname github.com
  ```

## 7. **test 폴더 엿보기**
  - 다양한 post 예시 소스가 있음


## 8. 요약
1. Ruby, Git, VS Code 설치.
2. Minimal Mistakes 테마 설치
3. VS Code와 연동해 개발.
4. GitHub Pages로 배포.

