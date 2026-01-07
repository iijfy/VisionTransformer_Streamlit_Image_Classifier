import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd
import altair as alt



# 1. 페이지 기본설정
st.set_page_config(
    page_title="이미지 분류(Vision Transformer)",
    page_icon="🎭",
    layout="centered",
)

st.title("🎭 이미지 분류(Vision Transformer)")
st.write(
    "이미지를 직접 업로드 또는 카메라로 찍으면 구글AI가 무엇인지 분류해줄꺼예요^^\n"
    "- 처음 실행은 모델 다운로드 때문에 조금 느릴 수 있어요. 조금만 기다려 주세요!"
)
#google/vit-base-patch16-224


# 2. 모델 로딩 (캐싱적용)
@st.cache_resource
def load_classifier():
    """
    Streamlit은 입력이 바뀔 때마다 스크립트를 위->아래로 다시 실행
    그래서 모델을 매번 새로 만들면 너무 느려짐.
    st.cache_resource로 모델을 캐싱하면 한번 로드후에 재사용이 가능
    """
    return pipeline(
        task="image-classification",
        model="google/vit-base-patch16-224",
        device=-1, # CPU
    )



# 3. 유틸 함수들 (라벨 정리 / 이모지 / 차트)
def prettify_label(raw_label: str) -> str:
    """
    ImageNet 라벨은 동의어가 쉼표로 붙어오는 경우가 있어서,
    사용자는 보통 핵심 단어 1개만 보고 싶어하기 때문에 첫 단어만 잘라서 보여준다.
    """
    return raw_label.split(",")[0].strip()


def emoji_for_label(label: str) -> str:

    x = label.lower()

    # 동물
    if "dog" in x or "retriever" in x or "poodle" in x:
        return "🐶"
    if "cat" in x or "kitten" in x:
        return "🐱"
    if "pig" in x or "hog" in x or "boar" in x:
        return "🐷"
    if "cow" in x or "ox" in x or "cattle" in x:
        return "🐮"
    if "horse" in x:
        return "🐴"
    if "bird" in x or "eagle" in x or "sparrow" in x:
        return "🐦"

    # 탈것/사물
    if "car" in x or "truck" in x or "bus" in x:
        return "🚗"
    if "plane" in x or "aircraft" in x:
        return "✈️"
    if "ship" in x or "boat" in x:
        return "🚢"

    return "🔎"


def results_to_df(results):
    """
    pipeline 결과(list[dict]) -> DataFrame
    결과를 표/차트로 다루기 쉬워짐
    """
    # results: [{"label": "...", "score": 0.98}, ...]
    df = pd.DataFrame(results).copy()
    df["percent"] = df["score"] * 100
    return df


def draw_topk_bar_chart(df: pd.DataFrame):
    """
    Altair로 Top-K 막대 차트
    """
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("percent:Q", title="확률(%)"),
            y=alt.Y("label:N", sort="-x", title="예측 클래스"),
            tooltip=["label", alt.Tooltip("percent:Q", format=".1f")],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)



# 4. 사이드바 옵션
st.sidebar.header("⚙️ 옵션 설정")
top_k = st.sidebar.slider("Top-K 결과 개수", 1, 10, 5, 1)

use_pretty_label = st.sidebar.checkbox("정돈된 라벨 표시", value=True)
show_emoji = st.sidebar.checkbox("이모지 표시", value=True)
show_chart = st.sidebar.checkbox("Top-K 막대차트 표시", value=True)

st.sidebar.caption("한번에 여러 장 업로드하시면, 한 번에 분류할 수 있어요!")



# 5. 입력 탭(업로드 / 카메라)
tab_upload, tab_camera = st.tabs(["📁 업로드", "📷 카메라"])


# 5-1. 업로드 탭: 여러 이미지 업로드 + 한 번에 분류
with tab_upload:
    st.subheader("📁 이미지 업로드")

    uploaded_files = st.file_uploader(
        "이미지 파일을 업로드하세요 (여러 장 가능)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("👆 이미지를 업로드하면 미리보기와 분류 버튼이 나타나요.")
    else:
        # 업로드된 이미지를 PIL로 변환해서 리스트로 보관
        images = []
        for f in uploaded_files:
            img = Image.open(f).convert("RGB")  # RGBA/그레이 방어
            images.append((f.name, img))

        st.write(f"업로드된 이미지: **{len(images)}장**")

        # 미리보기 (여러 장이면 expander로 접어두면 화면이 덜 길어짐)
        with st.expander("✅ 업로드 미리보기 열기", expanded=True):
            for name, img in images:
                st.markdown(f"**- {name}**")
                st.image(img, use_container_width=True)

        # 폼 적용으로 깔끔하게 분류 버튼 배치
        with st.form("upload_form"):
            submit = st.form_submit_button("🔍 업로드 이미지 분류하기")

        if submit:
            classifier = load_classifier()

            with st.spinner("모델이 이미지를 분석 중이에요..."):
                # 이미지 여러 장을 순회하며 결과 생성
                for name, img in images:
                    st.divider()
                    st.markdown(f"### 🖼️ {name}")

                    results = classifier(img, top_k=top_k)
                    top1 = results[0]

                    raw_label = top1["label"]
                    score = float(top1["score"])

                    label = prettify_label(raw_label) if use_pretty_label else raw_label
                    emoji = emoji_for_label(label) if show_emoji else ""

                    st.success("분류 완료!")
                    st.metric("Top-1 예측", f"{emoji} {label}".strip())
                    st.write(f"신뢰도: **{score * 100:.1f}%**")
                    st.progress(min(max(score, 0.0), 1.0))

                    # Top-K 표/차트
                    df = results_to_df(results)

                    st.markdown(f"#### 📌 Top-{top_k} 결과")
                    # 리스트+progress 형태
                    for r in results:
                        r_label_raw = r["label"]
                        r_label = prettify_label(r_label_raw) if use_pretty_label else r_label_raw
                        r_score = float(r["score"])
                        r_emoji = emoji_for_label(r_label) if show_emoji else ""
                        st.write(f"**{r_emoji} {r_label}** — {r_score * 100:.1f}%")
                        st.progress(min(max(r_score, 0.0), 1.0))

                    # 막대차트
                    if show_chart:
                        st.markdown("#### 📊 Top-K 막대 차트")
                      
                        chart_df = df.copy()
                        if use_pretty_label:
                            chart_df["label"] = chart_df["label"].apply(prettify_label)
                        draw_topk_bar_chart(chart_df)

                    # 원본 라벨 체크용
                    if use_pretty_label:
                        st.caption(f"원본 라벨(Top-1): {raw_label}")



# 5-2. 카메라 탭: 촬영 후 분류

with tab_camera:
    st.subheader("📷 카메라로 찍어서 분류")

    camera_image = st.camera_input("카메라로 사진을 찍어보세요")

    if camera_image is None:
        st.info("👆 카메라로 찍으면 아래에 결과가 나와요.")
    else:
        img = Image.open(camera_image).convert("RGB")
        st.image(img, use_container_width=True)

        if st.button("🔍 카메라 이미지 분류하기", type="primary"):
            classifier = load_classifier()

            with st.spinner("모델이 이미지를 분석 중이에요..."):
                results = classifier(img, top_k=top_k)

            top1 = results[0]
            raw_label = top1["label"]
            score = float(top1["score"])

            label = prettify_label(raw_label) if use_pretty_label else raw_label
            emoji = emoji_for_label(label) if show_emoji else ""

            st.success("분류 완료!")
            st.metric("Top-1 예측", f"{emoji} {label}".strip())
            st.write(f"신뢰도: **{score * 100:.1f}%**")
            st.progress(min(max(score, 0.0), 1.0))

            df = results_to_df(results)

            st.markdown(f"#### 📌 Top-{top_k} 결과")
            if show_chart:
                chart_df = df.copy()
                if use_pretty_label:
                    chart_df["label"] = chart_df["label"].apply(prettify_label)
                draw_topk_bar_chart(chart_df)

            # 표로 보기
            with st.expander("📄 결과 표로 보기"):
                if use_pretty_label:
                    df2 = df.copy()
                    df2["label"] = df2["label"].apply(prettify_label)
                    st.dataframe(df2[["label", "percent"]])
                else:
                    st.dataframe(df[["label", "percent"]])

            if use_pretty_label:
                st.caption(f"원본 라벨(Top-1): {raw_label}")
