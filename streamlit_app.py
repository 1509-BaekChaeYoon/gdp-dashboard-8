# streamlit_app.py
"""
Streamlit 앱: 해수면온도(SST) vs 폭염일수 대시보드 (한국어 UI)

- 공개 데이터 URL 접근 부분 제거, 예시 데이터 기반으로 시각화
- 사용자 입력 텍스트 기반 합성 데이터 포함
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import pytz
import plotly.express as px
import matplotlib.pyplot as plt

# ---------- 설정 ----------
st.set_page_config(page_title="폭염·해수온 대시보드", layout="wide", initial_sidebar_state="expanded")

# Pretendard 폰트 적용 시도
PRETENDARD_PATH = "/fonts/Pretendard-Bold.ttf"
_custom_css = f"""
<style>
@font-face {{
  font-family: 'PretendardCustom';
  src: url('{PRETENDARD_PATH}');
}}
html, body, [class*="css"] {{
  font-family: PretendardCustom, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}}
</style>
"""
st.markdown(_custom_css, unsafe_allow_html=True)
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont(PRETENDARD_PATH)
    plt.rcParams['font.family'] = 'PretendardCustom'
except Exception:
    pass

# ---------- 유틸리티 ----------
@st.cache_data
def today_local_date():
    tz = pytz.timezone("Asia/Seoul")
    return dt.datetime.now(tz).date()

TODAY = today_local_date()

def remove_future_dates(df, date_col="date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col].dt.date <= TODAY]
    return df

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# ---------- 공개 데이터 예시 ----------
st.header("공개 데이터 기반 대시보드 (예시 데이터 사용)")

public_data_notice = st.empty()
public_warning = True

years = np.arange(2005, 2025)
dates = pd.to_datetime([f"{y}-07-01" for y in years])
sst_values = 20.0 + (years - 2005) * 0.02 + np.random.normal(0, 0.08, len(years))
public_sst_df = pd.DataFrame({'date': dates, 'sst': sst_values})

col1, col2 = st.columns([3,1])
with col1:
    fig = px.line(public_sst_df, x='date', y='sst', title='한반도 주변 해수면 온도(예시 데이터)',
                  labels={'date':'연도', 'sst':'해수면 온도 (℃)'})
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.markdown("**데이터 출처(예시)**")
    st.write("- NOAA ERSST v5 (예시 URL).")
    st.write("- NOAA OISST (대체 가능).")
    st.write("- 기상청 폭염일수 포털 예시 URL")
    st.write("**알림:** 공개 데이터 접근이 불완전하여 예시 데이터 사용 중입니다.")
    st.download_button("해수면온도 CSV 다운로드", data=df_to_csv_bytes(public_sst_df), file_name="public_sst_preprocessed.csv", mime="text/csv")

# ---------- 사용자 입력 기반 대시보드 ----------
st.header("사용자 입력 기반 대시보드 — 입력 텍스트 분석 결과")

@st.cache_data
def synthesize_from_text():
    years = np.arange(2005, 2025)
    sst = 20.0 + (years - 2005) * 0.02 + np.random.normal(0, 0.05, len(years))
    heatdays = 5 + (years - 2005) * (13 / 20) + np.random.normal(0, 1.5, len(years))
    heatdays = np.clip(heatdays.round(1), 0, None)
    sleep_hours = 8.5 - (years - 2005) * (0.3 / 20) + np.random.normal(0, 0.05, len(years))
    df = pd.DataFrame({
        'year': years,
        'date': pd.to_datetime([f"{y}-07-01" for y in years]),
        'sst': sst,
        'heatwave_days': heatdays,
        'avg_sleep_hours': sleep_hours
    })
    df = remove_future_dates(df, 'date')
    return df

user_df = synthesize_from_text()

st.sidebar.header("필터 · 옵션")
yr_min = int(user_df['year'].min())
yr_max = int(user_df['year'].max())
year_range = st.sidebar.slider("연도 범위 선택", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max), step=1)
smoothing_window = st.sidebar.selectbox("시계열 스무딩(이동평균) 기간(년)", options=[1,3,5], index=1)
show_trend = st.sidebar.checkbox("추세선 표시", value=True)

df_vis = user_df[(user_df['year'] >= year_range[0]) & (user_df['year'] <= year_range[1])].copy()
df_vis['sst_smooth'] = df_vis['sst'].rolling(smoothing_window, center=True, min_periods=1).mean()
df_vis['heat_smooth'] = df_vis['heatwave_days'].rolling(smoothing_window, center=True, min_periods=1).mean()

st.subheader("요약 시각화")
p1, p2 = st.columns([2,1])
with p1:
    fig1 = px.line(df_vis, x='year', y='sst_smooth', labels={'year':'연도', 'sst_smooth':'해수면 온도(℃)'},
                   title="해수면 온도(여름 대표값) vs 폭염일수")
    fig1.add_bar(x=df_vis['year'], y=df_vis['heat_smooth'], name='폭염일수(이동평균)', yaxis='y2', opacity=0.5)
    fig1.update_layout(
        yaxis=dict(title="해수면 온도 (℃)"),
        yaxis2=dict(title="폭염일수 (일)", overlaying='y', side='right'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig1, use_container_width=True)
with p2:
    corr_val = df_vis['sst'].corr(df_vis['heatwave_days'])
    st.metric("SST vs 폭염일수 상관계수 (피어슨)", f"{corr_val:.2f}")
    st.write("설명: 양(+)의 상관관계가 있음을 입력 텍스트 기반 합성 데이터로 재현함.")

st.subheader("해수면 온도와 폭염일수 산점도")
fig2 = px.scatter(df_vis, x='sst', y='heatwave_days', trendline=None,
                  labels={'sst':'해수면 온도 (℃)', 'heatwave_days':'연간 폭염일수 (일)'},
                  hover_data=['year'])
st.plotly_chart(fig2, use_container_width=True)

st.subheader("청소년 평균 수면시간(추정)")
fig3 = px.line(df_vis, x='year', y='avg_sleep_hours', labels={'avg_sleep_hours':'평균 수면시간 (시간)','year':'연도'},
               title="평균 수면시간(가정: 폭염/열대야 영향으로 감소)")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("전처리된 표 (다운로드 가능)")
st.dataframe(df_vis[['year','date','sst','heatwave_days','avg_sleep_hours']].reset_index(drop=True), use_container_width=True)
st.download_button("전처리된 표 CSV 다운로드", data=df_to_csv_bytes(df_vis[['year','date','sst','heatwave_days','avg_sleep_hours']]), file_name="user_input_synthesized.csv", mime="text/csv")

st.header("간단 결론 및 권고 (프롬프트 기반)")
st.markdown(
"""
- 포인트: 한반도 주변 해수면 온도 상승(최근 20년, 약 +0.3~0.5℃ 수준 추정)은 내륙 폭염일수 증가와 양(+)의 상관관계를 보입니다.
- 영향: 폭염·열대야는 청소년의 수면 시간과 질을 저하시켜 학업성취 및 정신·신체 건강에 부정적 영향을 미칠 가능성이 큽니다.
- 권고:
  1. 야간 냉방 접근성 확대 및 학교 내 휴식환경 개선
  2. 폭염·열대야에 따른 학생 수면 모니터링 및 교육
  3. 장기적으로 해양·기후 모니터링 강화 및 온실가스 감축 정책 병행
"""
)
import plotly.express as px
import matplotlib.pyplot as plt

# -------------------------------
# 보고서 본문 (접었다 펼 수 있는 구조)
# -------------------------------

with st.expander("📑 보고서 내용 (접었다/폈다 가능)"):
    st.subheader("1. 연구 배경")
    st.write("""
    최근 수년간 여름철 평균 기온이 꾸준히 상승하면서 청소년들의 수면 부족 및 학업 성취도에 영향을 미친다는 연구가 늘어나고 있다.
    """)

    # 예시 그래프 1
    import matplotlib.pyplot as plt
    import numpy as np

    years = np.arange(2015, 2024, 2)
    temp = [24.1, 24.3, 24.8, 25.2, 25.5]  # 예시 온도 데이터
    score = [82, 80, 78, 76, 74]           # 예시 성취도 데이터

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(years, temp, marker="o", label="평균 기온(°C)")
    ax2.plot(years, score, marker="s", color="orange", label="학업 성취도(점수)")

    ax1.set_xlabel("연도")
    ax1.set_ylabel("평균 기온(°C)")
    ax2.set_ylabel("성취도(점수)")
    ax1.set_title("여름철 평균 기온과 청소년 학업 성취도 추이")

    st.pyplot(fig)

    st.subheader("2. 분석 결과")
    st.write("""
    - 여름철 평균 기온이 꾸준히 상승하고 있음  
    - 기온 상승과 동시에 성취도 점수가 하락하는 경향 확인  
    """)

    st.subheader("3. 결론 및 제언")
    st.write("""
    기후 변화가 학업 성취도에 간접적 영향을 미칠 수 있으므로, 교육기관 차원에서 여름철 학습 환경 개선(냉방, 수면 관리 교육 등)이 필요하다.
    """)
