# 🏞️ Streamlit 이미지 분류기 (Vision Transformer)

사용자가 웹에서 이미지를 업로드 또는 카메라로 촬영하면, Hugging Face 사전학습 모델로   
이미지가 무엇인지 분류해주는 Streamlit 입니다.

- 모델: `google/vit-base-patch16-224` (ImageNet 1000 클래스 분류)
- 흐름: 이미지 입력 → 모델 추론 → Top-1/Top-K 결과 + 확률 시각화

---

## 1.📌 설명
Streamlit에서 이미지 업로드/카메라 촬영으로 입력하면,
Hugging Face ViT 모델이 Top-K 분류 결과를 확률/차트로 시각화해주는 웹 앱입니다.

---

## 2. 주요 기능

* 필수 기능
- 이미지 업로드 기능 (jpg/png/webp)
- 업로드 이미지 미리보기
- "분류하기" 버튼 클릭 시 추론 실행
- Top-1 결과 강조 + Top-K 결과 표시
- 확률(신뢰도) 퍼센트 + progress bar로 시각화
- 모델 캐싱 적용(`st.cache_resource`)으로 재실행 시 속도 개선

* 추가 기능
- Top-5(Top-K) 결과 막대 차트 시각화 (Altair)
- 카메라로 직접 촬영해서 분류 (`st.camera_input`)
- 여러 이미지 한 번에 업로드/분류 (멀티 업로드)
- 분류 결과에 이모지 표시 + 라벨을 보기 좋게 정리(대표 단어만 표시)

---

## 3. 사용 라이브러리

- Streamlit: 웹 UI 구성
- Hugging Face Transformers: 이미지 분류 파이프라인(`pipeline - image-classification`)
- PyTorch(torch): 모델 실행 백엔드
- Pillow(PIL): 업로드된 이미지 처리
- Pandas + Altair: Top-K 결과 차트 시각화

---

## 4. 폴더 구조

streamlit_vit/
- .gitignore
- app.py
- requirements.txt
- README.md

---

## 5. 설치 및 실행 방법 - 로컬

### 5-1. 가상환경 생성/활성화

macOS 기준

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 5-2. 패키지 설치 및 실행
```bash
 pip install -r requirements.txt
 streamlit run app.py
```
---

## 6. requirements.txt
- numpy 2.x 환경에서 일부 라이브러리가 충돌해서 고정

```text
numpy<2reamlit
transformers
torch
pillow
pandas
altair

--
```
---

7. ## 🟨 특이 사항
1) 모델 캐싱(st.cache_resource)
- Streamlit은 입력이 바뀔 때마다 스크립트를 위에서 아래로 다시 실행합니다. 모델을 매번 다시 로드하면 너무 느려져서 캐싱으로 해결했습니다.

2) 이미지 모드 통일(convert - RGB)
- PNG 투명 배경 등으로 RGBA가 들어올 수 있어, 모델 입력 안정성을 위해 RGB로 통일했습니다.

3) 라벨정리 
- 돼지 사진을 업로드 하였더니 돼지와 비슷한 동의어가 같이 여러개 나열되어서 UX를 위해 대표 단어만 표시하도록 했습니다. 
- 원본라벨도 확인가능하도록 캡션으로 남겼습니다. 




