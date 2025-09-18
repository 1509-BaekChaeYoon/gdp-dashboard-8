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
# ---------- 폭염 세대 보고서 (접었다 폈다 가능) ----------
st.header("폭염 세대: 잠을 빼앗긴 10대, 벼랑 끝에 서다")

with st.expander("▶ 보고서 열기 / 접기"):
    st.markdown(
    """
    ### 서론 (문제 제기)
    최근 들어 지구의 해수 온도는 꾸준히 상승하고 있다. 이는 단순한 해양 환경의 변화가 아니라, 대기와 기후 전반에 영향을 미치는 심각한 현상이다. 특히 바닷물의 온도가 높아질수록 대기 중 열에너지가 증가하여 여름철 폭염이 더욱 강렬해지고, 그 결과 ‘열대야’라는 새로운 일상이 우리 앞에 등장하게 되었다.
    낮 동안 달궈진 열기는 밤에도 식지 않고 남아, 청소년들의 수면 환경을 위협한다. 성장기 청소년에게 필수적인 깊은 수면이 방해받으면, 집중력과 학습 능력은 저하되고, 교실에서는 졸음과 피로에 시달리는 학생들이 늘어난다. 단순한 불편함을 넘어, 불안과 스트레스, 정서적 불균형까지 초래한다. 결국 폭염은 청소년의 건강·학습·미래를 뒤흔드는 거대한 그림자로 자리 잡고 있다.

    ### 본론 1 (데이터 분석)
    #### 1-1. 폭염과 수면 패턴 변화 상관관계
    야간 기온 상승과 청소년 수면 시간의 변화를 분석한 68개국 다국적 연구(미국수면의학회) 결과에 따르면, 기온이 높을수록 수면 시간이 단축되고 수면의 질 역시 저하되는 경향이 확인되었다. 특히 열대야가 이어지는 시기에는 △뒤척임 증가 △깊은 수면 도달 지연 △전체 수면 효율 저하가 동시에 나타났다.
    이를 시각화한 선 그래프에서는 기온이 일정 수준 이상 오를 때 평균 수면 시간이 가파르게 감소하는 추세가 뚜렷하게 나타난다. 이는 기후 변화가 단순한 환경적 불편이 아니라, 청소년 수면 건강에 직접적이고 구조적인 위협임을 과학적으로 보여준다.

    #### 1-2. 청소년 수면 부족 실태와 통계
    대한민국 청소년건강행태온라인조사 결과, 청소년의 70~90%가 권장 수면 시간(8시간 이상)을 확보하지 못하고 있음이 드러났다. 학업 부담, 스마트폰 사용과 같은 생활 습관 요인에 더해, 여름철 폭염·열대야는 수면 부족을 한층 악화시키는 주요 기후 요인으로 작용한다.
    도넛 차트 분석에서는 ‘충분한 수면 확보 집단’이 극소수에 불과하고, ‘수면 부족 집단’이 압도적 다수를 차지하는 구조적 문제가 명확하게 드러난다. 이는 곧 생활습관 요인과 기후 변화 요인이 결합해 청소년 수면 건강에 심각한 부담을 가중시키고 있음을 직관적으로 보여주는 증거다.

    ### 본론 2 (원인 및 영향 탐구)
    #### 2-1. 학업 성취도 저하
    수면 부족은 단순한 피로감에 그치지 않고, 인지적·학습적 수행능력 전반에 걸친 체계적 저하를 초래한다. 충분한 수면을 취하지 못한 청소년은 집중력과 단기 기억력이 떨어지고, 이는 수업 참여도 감소와 학습 효율 저하로 직결된다. 시험 성적 하락, 과제 수행 능력 저하와 같은 구체적 결과로 이어지며, 장기적으로는 학업 성취도의 지속적 하락을 유발한다.
    특히 기후 변화로 인한 폭염·열대야가 야간 수면을 방해할 경우, 수면 시간의 절대적 감소 + 깊은 수면 부족이 동시에 발생하여 학습 능력 저하는 더욱 심각해진다. 결국 이는 학습 계획 수립의 어려움, 학업 성취 불균형, 장래 진로 선택의 제약 등 청소년의 교육적 미래 전반에 부정적 영향을 준다.

    #### 2-2. 건강 악화
    수면 부족은 청소년의 정신적·신체적 건강 모두에 복합적 악영향을 끼친다.
    - 정신 건강 측면: 불안·우울·스트레스가 증가하여 정서적 안정성이 붕괴된다.
    - 신체 건강 측면: 면역력 저하, 성장 호르몬 분비 감소로 정상 발달이 방해된다.
    장기간의 수면 부족은 대사 기능 불균형을 야기해 비만 위험을 높이고, 장기적으로는 심혈관 질환, 당뇨 등 만성 질환 발병률까지 증가시킬 수 있다.
    여기에 폭염과 열대야로 인한 수면 질 저하는 이러한 부정적 영향을 가속화한다.

    ### 결론 (제언)
    그래서, 우리는 무엇을 해야 할까?
    이 보고서를 통해 우리는 기후 변화와 해수면 상승이 단순히 환경 문제에 그치지 않고, 우리의 일상과 직결된 문제임을 확인했다. 특히 여름철 폭염과 열대야는 청소년들의 수면 패턴을 심각하게 위협하고 있으며, 이는 곧 집중력 저하와 학업 수행력, 나아가 정신 건강까지 영향을 미치고 있다. 이제는 단순히 문제를 아는 것에서 멈추지 않고, 우리가 직접 실천할 수 있는 변화를 만들어가야 한다.

    #### 제언
    1. **‘수면 지킴이 데이터랩’ – 과학적으로 문제 이해하기**  
       - 학급별 ‘수면·기온 관찰 프로젝트’ 운영  
       - 데이터 시각화 및 카드뉴스/영상 제작 공유
    2. **‘시원한 교실 프로젝트’ – 지금 당장 실천하기**  
       - 블라인드, 칼환기, 불·전자기기 관리 등 교실 온도 조절  
    3. **‘데이터로 말하기’ – 우리의 목소리 전달하기**  
       - 기록·자료 기반으로 정책 제안 및 학생회 활동 연계

    ### 참고 자료
    - https://www.kihasa.re.kr/hswr/assets/pdf/91/journal-39-1-230.pdf?utm_source=chatgpt.com
    - https://medigatenews.com/news/1841034635?utm_source=chatgpt.com
    - https://www.ncei.noaa.gov/products/extended-reconstructed-sst
    - https://data.kma.go.kr/resources/html/en/aowdp.html
    - https://www.sciencedirect.com/science/article/pii/S235286741930132X
    - https://koreascience.kr/article/JAKO201609633506261.pdf
    - https://www.healthlifeherald.com/news/articleView.html?idxno=2542
    - https://www.goodnews1.com/news/articleView.html?idxno=449373
    """
    )