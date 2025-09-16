# streamlit_app.py
"""
Streamlit 앱: 해수면온도(SST) vs 폭염일수 대시보드 (한국어 UI)

구현 원칙 요약:
- 먼저 공식 공개 데이터(예: NOAA OISST / NOAA ERSST, 기상청 폭염일수)를 시도해서 불러옵니다.
- API/파일 호출 실패 시 예시(설명 기반) 데이터로 자동 대체하고 화면에 안내합니다.
- 사용자 입력 대시보드는 이 프롬프트의 '입력(Input)' 텍스트 설명만 사용(파일 업로드나 추가 입력 요구 없음).
- 모든 레이블·버튼·툴팁은 한국어로 작성.
- 출처(URL)를 코드 주석으로 명시.

주요 공개 데이터 출처(예시):
- NOAA OISST (Daily, 0.25°) / Optimum Interpolation SST:
  https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html
  (PSL/NOAA THREDDS/NetCDF 접근 가능)
- NOAA ERSST v5 (Monthly, 2°):
  https://www.ncei.noaa.gov/products/extended-reconstructed-sea-surface-temperature
  https://www1.ncdc.noaa.gov/pub/data/cmb/ersst/v5/netcdf/
- 기상청(대한민국) 폭염일수 통계 (다운로드/조회):
  https://data.kma.go.kr/climate/heatWave/selectHeatWaveChart.do
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import pytz
from dateutil import parser
import io
import xarray as xr
import requests
import plotly.express as px
import matplotlib.pyplot as plt

# ---------- 설정 ----------
st.set_page_config(page_title="폭염·해수온 대시보드", layout="wide", initial_sidebar_state="expanded")

# Attempt to apply Pretendard font if available
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

# Matplotlib font attempt (fallback to default if font not available)
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont(PRETENDARD_PATH)
    plt.rcParams['font.family'] = 'PretendardCustom'
except Exception:
    pass

# ---------- 유틸리티 ----------
@st.cache_data
def today_local_date():
    # User timezone: Asia/Seoul (developer message)
    tz = pytz.timezone("Asia/Seoul")
    return dt.datetime.now(tz).date()

TODAY = today_local_date()

def remove_future_dates(df, date_col="date"):
    # date_col may be datetime or string
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col].dt.date <= TODAY]
    return df

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# ---------- 공개 데이터 불러오기 (시도) ----------
st.header("공식 공개 데이터 기반 대시보드 (NOAA + 기상청 등)")

public_data_notice = st.empty()
public_warning = None

# We'll try ERSST v5 monthly (NOAA) as a robust, easy-to-download monthly SST record.
ERSST_NETCDF_URL = "https://www1.ncdc.noaa.gov/pub/data/cmb/ersst/v5/netcdf/ersst.v5.1854-2023.nc"
# Note: 파일명 / URL 갱신 가능 — 코드에서는 예시 URL을 사용하며 실패 시 대체 데이터를 사용합니다.
# 출처: NOAA ERSST v5 (Extended Reconstructed Sea Surface Temperature)
# https://www.ncei.noaa.gov/products/extended-reconstructed-sea-surface-temperature

@st.cache_data
def load_noaa_ersst(url=ERSST_NETCDF_URL, region_bbox=None):
    """
    시도: xarray로 ERSST netCDF 열기.
    region_bbox = (lonmin, lonmax, latmin, latmax) — 한반도 주변 영역을 잘라냄 (예: 120E-135E, 33N-44N)
    실패 시 None 반환.
    """
    try:
        ds = xr.open_dataset(url)
        # ERSST variable often 'sst' with dims (time, lat, lon)
        varname = None
        for candidate in ['sst', 'SST', 'sst_anom']:
            if candidate in ds:
                varname = candidate
                break
        if varname is None:
            varname = list(ds.data_vars)[0]
        sst = ds[varname]
        # select region if bbox provided
        if region_bbox is not None:
            lonmin, lonmax, latmin, latmax = region_bbox
            # ensure lon coordinate range compatible
            if ds.lon.max() > 180:
                # convert to -180..180
                ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
            sst = sst.sel(lon=slice(lonmin, lonmax), lat=slice(latmin, latmax))
        # monthly mean over region (area simple mean here)
        sst_mean = sst.mean(dim=['lon', 'lat'], skipna=True).to_dataframe().reset_index()
        sst_mean.columns = ['date', 'sst']
        # ERSST is monthly — convert to datetime (already time)
        sst_mean['date'] = pd.to_datetime(sst_mean['time'] if 'time' in sst_mean.columns else sst_mean['date'])
        # keep only up to today
        sst_mean = remove_future_dates(sst_mean, 'date')
        return sst_mean[['date', 'sst']]
    except Exception as e:
        return None

# define Korean peninsula bbox approx (lonmin, lonmax, latmin, latmax) in degrees
KOREA_BBOX = (120, 135, 33, 44)

public_sst_df = load_noaa_ersst(region_bbox=KOREA_BBOX)

# 기상청 폭염일수 데이터 시도 (기상자료개방포털 — CSV 다운로드 경로는 웹 인터페이스 의존)
# We'll attempt to call the KMA open portal page to indicate source; direct API endpoints may require parameters.
KMA_HEAT_URL = "https://data.kma.go.kr/climate/heatWave/selectHeatWaveChart.do"
# Note: For robust reproducibility, we will attempt to fetch a CSV via data.go.kr metadata if available.
@st.cache_data
def load_kma_heatyear():
    # The portal typically provides CSV via web UI; here we attempt minimal fetch to validate availability.
    try:
        r = requests.get(KMA_HEAT_URL, timeout=10)
        if r.status_code == 200:
            # We cannot parse the interactive page to CSV reliably here. Return a placeholder indicator (not failure).
            # Real deployment: use KMA Open API endpoints or download CSV from data.go.kr.
            return True
        else:
            return False
    except Exception:
        return False

kma_available = load_kma_heatyear()

# If either public dataset load failed, prepare synthetic/example dataset based on prompt narrative
if public_sst_df is None or not kma_available:
    public_warning = True
    public_data_notice.warning("공개 데이터(또는 KMA 접근)에 일부 실패했습니다. 예시(설명 기반) 데이터로 자동 대체하여 시각화합니다. 실제 분석 시 위 출처에서 원본 데이터를 연결하세요.")
    # create example SST (연도별, 2005-2024 monthly mean approximated)
    years = np.arange(2005, 2025)
    dates = pd.to_datetime([f"{y}-07-01" for y in years])  # 집중 여름값(7월) 대표 사용
    # trend: +0.015 ~ +0.025 ℃/year for summertime per narrative (20년 약 0.3~0.5C)
    sst_values = 20.0 + (years - 2005) * 0.02 + np.random.normal(0, 0.08, len(years))
    public_sst_df = pd.DataFrame({'date': dates, 'sst': sst_values})
else:
    public_data_notice.info("공개 데이터(ERSST) 로드 완료. (지역: 한반도 주변, 월별)")

# ---------- 공개 데이터 대시보드 표시 ----------
st.subheader("(공개 데이터) 한반도 주변 해수면 온도(월별) — NOAA ERSST (또는 예시 데이터)")
col1, col2 = st.columns([3,1])

with col1:
    fig = px.line(public_sst_df, x='date', y='sst', title='한반도 주변 해수면 온도(월별 평균 — 대표 여름값 포함)',
                  labels={'date':'연도', 'sst':'해수면 온도 (℃)'})
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**데이터 출처(시도한 원본)**")
    st.write("- NOAA ERSST v5 (예시 URL).")
    st.write("  - https://www.ncei.noaa.gov/products/extended-reconstructed-sea-surface-temperature")
    st.write("- NOAA OISST (대체 가능): https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html")
    st.write("- 기상청 폭염일수 포털: https://data.kma.go.kr/climate/heatWave/selectHeatWaveChart.do")
    if public_warning:
        st.write("**알림:** 현재 공개 데이터 접근이 완전하지 않아 예시 데이터(프롬프트 설명 기반)를 사용했습니다.")
    st.download_button("해수면온도(전처리된) CSV 다운로드", data=df_to_csv_bytes(public_sst_df), file_name="public_sst_preprocessed.csv", mime="text/csv")

# ---------- 사용자 입력 대시보드 (프롬프트 텍스트 기반) ----------
st.header("사용자 입력 기반 대시보드 — 입력 텍스트 분석 결과")
st.markdown("입력: '폭염 세대: 잠을 뺏긴 10대, 벼랑 끝에 서다' 텍스트 설명을 기반으로 시각화(파일 업로드 요구 없음).")

# From prompt narrative we derive a synthetic dataset that matches claims:
# - 지난 20년간 한반도 주변 해수면 온도 연평균 약 +0.3~0.5℃ (2005-2024)
# - 폭염일수(연간)도 증가 경향
@st.cache_data
def synthesize_from_text():
    years = np.arange(2005, 2025)
    # Create annual mean SST (summer-focused), slope ~0.02 per year => ~0.4 over 20 years
    sst = 20.0 + (years - 2005) * 0.02 + np.random.normal(0, 0.05, len(years))
    # Create annual heatwave days, increase from ~5 -> ~18 over period (for Korea-like behavior)
    heatdays = 5 + (years - 2005) * (13 / 20) + np.random.normal(0, 1.5, len(years))
    heatdays = np.clip(heatdays.round(1), 0, None)
    # Create a proxy for '청소년 수면 시간 평균' decline (hours per night) based on narrative
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

# Side controls automatically configured based on data characteristics
st.sidebar.header("필터 · 옵션")
yr_min = int(user_df['year'].min())
yr_max = int(user_df['year'].max())
year_range = st.sidebar.slider("연도 범위 선택", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max), step=1)

smoothing_window = st.sidebar.selectbox("시계열 스무딩(이동평균) 기간(년)", options=[1,3,5], index=1)
show_trend = st.sidebar.checkbox("추세선 표시", value=True)

# apply filters
df_vis = user_df[(user_df['year'] >= year_range[0]) & (user_df['year'] <= year_range[1])].copy()
df_vis['sst_smooth'] = df_vis['sst'].rolling(smoothing_window, center=True, min_periods=1).mean()
df_vis['heat_smooth'] = df_vis['heatwave_days'].rolling(smoothing_window, center=True, min_periods=1).mean()

# Plots: (1) 연도별 SST & 폭염일수 동시 표시 (이중축), (2) 상관관계 산점도, (3) 수면시간 변화
st.subheader("요약 시각화")

p1, p2 = st.columns([2,1])

with p1:
    fig1 = px.line(df_vis, x='year', y='sst_smooth', labels={'year':'연도', 'sst_smooth':'해수면 온도(℃)'}, title="해수면 온도(여름 대표값) vs 폭염일수")
    fig1.add_bar(x=df_vis['year'], y=df_vis['heat_smooth'], name='폭염일수(이동평균)', yaxis='y2', opacity=0.5)
    fig1.update_layout(
        yaxis=dict(title="해수면 온도 (℃)"),
        yaxis2=dict(title="폭염일수 (일)", overlaying='y', side='right'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig1, use_container_width=True)

with p2:
    # correlation
    corr_val = df_vis['sst'].corr(df_vis['heatwave_days'])
    st.metric("SST vs 폭염일수 상관계수 (피어슨)", f"{corr_val:.2f}")
    st.write("설명: 양(+)의 상관관계가 있음을 입력 텍스트(요약) 기반 합성 데이터로 재현함.")

# Scatter with regression line
st.subheader("해수면 온도와 폭염일수 상관관계 (산점도)")
fig2 = px.scatter(df_vis, x='sst', y='heatwave_days', trendline="ols",
                  labels={'sst':'해수면 온도 (℃)', 'heatwave_days':'연간 폭염일수 (일)'},
                  hover_data=['year'])
st.plotly_chart(fig2, use_container_width=True)

# Sleep hours trend
st.subheader("청소년 평균 수면시간(추정) — 텍스트 설명 기반 가정")
fig3 = px.line(df_vis, x='year', y='avg_sleep_hours', labels={'avg_sleep_hours':'평균 수면시간 (시간)','year':'연도'}, title="평균 수면시간(가정: 폭염/열대야 영향으로 감소)")
st.plotly_chart(fig3, use_container_width=True)

# Show processed table and download
st.subheader("전처리된 표 (다운로드 가능)")
st.dataframe(df_vis[['year','date','sst','heatwave_days','avg_sleep_hours']].reset_index(drop=True), use_container_width=True)
st.download_button("전처리된 표 CSV 다운로드", data=df_to_csv_bytes(df_vis[['year','date','sst','heatwave_days','avg_sleep_hours']]), file_name="user_input_synthesized.csv", mime="text/csv")

# ---------- 간단 분석 리포트 (텍스트) ----------
st.header("간단 결론 및 권고 (프롬프트 기반)")
st.markdown(
"""
- 포인트: 한반도 주변 해수면 온도 상승(최근 20년, 약 +0.3~0.5℃ 수준 추정)은 내륙 폭염일수 증가와 양(+)의 상관관계를 보입니다.
- 영향: 폭염·열대야는 청소년의 수면 시간과 질을 저하시켜 학업성취 및 정신·신체 건강에 부정적 영향을 미칠 가능성이 큽니다.
- 권고(정책·실천):
  1. 야간 냉방 접근성 확대 및 학교 내 휴식환경 개선(폭염기 실내 냉방·쿨링센터 운영).
  2. 폭염·열대야에 따른 학생 수면 모니터링 및 교육(스마트폰 사용 감축, 수면 위생 교육).
  3. 장기적으로는 해양·기후 모니터링 강화 및 온실가스 감축 정책 병행.
"""
)

# ---------- 추가 안내(운영자용) ----------
st.sidebar.markdown("### 운영자 안내")
st.sidebar.write("실제 운영 시:")
st.sidebar.write("- NOAA OISST(일별 0.25°) 또는 ERSST(월별 2°) 원본 NetCDF를 xarray로 직접 불러와 지역 평균을 계산하세요.")
st.sidebar.write("- 기상청 폭염일수는 기상자료개방포털의 CSV/API를 사용해 연도·지역별 표본을 확보하세요.")
st.sidebar.write("- 현재 예시 URL 및 접근 방식은 데모 목적이며, 배포 환경에서는 파일 접근 권한/대역폭 고려가 필요합니다.")

# End
