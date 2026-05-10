# 함수 그래프 개형 검증 Streamlit 앱

2022 개정 교육과정 미적분Ⅱ 수업에서 함수의 증감표를 이용해 손으로 그린 그래프와 디지털 그래프를 비교·검증하기 위한 Streamlit 웹 애플리케이션입니다.

## 주요 기능

- 함수식 입력 및 그래프 시각화
- 1차 도함수와 2차 도함수 자동 계산
- x절편, y절편, 극값 후보, 변곡점 후보, 수직점근선 후보 표시
- 도함수 부호 기반 증가·감소 표 제공
- 이계도함수 부호 기반 오목·볼록 표 제공
- 손그래프와 앱 결과 비교 성찰 작성
- 검증 보고서 Markdown 다운로드
- 수업 평가 기준 탭 제공

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속하면 됩니다.

## GitHub + Streamlit Community Cloud 배포

1. GitHub에서 새 저장소를 만듭니다.
2. 이 프로젝트 폴더의 파일을 저장소에 올립니다.

```bash
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Build Streamlit graph verification app"
git branch -M main
git remote add origin https://github.com/사용자명/저장소명.git
git push -u origin main
```

3. [Streamlit Community Cloud](https://share.streamlit.io/)에 접속합니다.
4. GitHub 계정으로 로그인합니다.
5. `New app`을 누릅니다.
6. Repository에 방금 만든 GitHub 저장소를 선택합니다.
7. Branch는 `main`, Main file path는 `app.py`로 설정합니다.
8. `Deploy`를 누르면 배포가 시작됩니다.

배포가 끝나면 `https://저장소명.streamlit.app` 형태의 URL이 생성됩니다. 이 URL을 수업 지도안의 Streamlit URL 항목에 넣으면 됩니다.

## 수업 활용 흐름

1. 학생들이 활동지에 주어진 함수의 증감표를 먼저 손으로 작성합니다.
2. 손으로 그래프 개형을 그립니다.
3. 앱에 같은 함수식을 입력해 그래프와 주요 지점을 확인합니다.
4. 손그래프와 앱 결과가 다르면 도함수 부호, 이계도함수 부호, 점근선 판단을 다시 점검합니다.
5. 비교·성찰 탭에 오류 원인과 수정 내용을 작성하고 보고서를 내려받습니다.
