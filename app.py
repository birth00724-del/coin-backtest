import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="업비트 전략 백테스터", layout="wide")
st.title("📈 업비트 전략 백테스터")
st.caption("기본 CSV를 자동으로 불러오며, 필요하면 업로드로 교체할 수 있습니다. (필수 열: Date, Close, Volume)")

# --------------------- Sidebar: Controls ---------------------
st.sidebar.header("⚙️ 설정")
initial_capital = st.sidebar.number_input("초기자금", min_value=1.0, value=100.0, step=1.0)

strategy_options = ["이동평균", "거래량돌파", "OBV", "VWAP"]
chosen_strategies = st.sidebar.multiselect("전략 선택", strategy_options, default=["이동평균", "VWAP"])

slippage_pct = st.sidebar.select_slider("슬리피지(%)", options=[0, 1, 2, 3, 4, 5], value=3)
slippage = slippage_pct / 100.0

years = st.sidebar.slider("분석 기간(년)", min_value=1, max_value=5, value=5, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("데이터 소스")
uploaded = st.sidebar.file_uploader("CSV 업로드(선택)", type=["csv"])
use_default = st.sidebar.checkbox("기본 CSV 사용(프로젝트 폴더)", value=True)

# --------------------- Load Data: default first, then replace if uploaded ---------------------
def load_default_csv(path="upbit_fake_daily_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df.sort_index()

data = None
status_msg = ""

try:
    if use_default:
        data = load_default_csv()
        status_msg = "✅ 기본 데이터(upbit_fake_daily_data.csv)를 불러왔습니다."
    else:
        status_msg = "ℹ️ 기본 CSV 사용 안 함으로 설정되었습니다."
except FileNotFoundError:
    status_msg = "⚠️ 기본 CSV가 프로젝트 폴더에 없습니다. 업로드를 이용하세요."

# 업로드가 있으면 업로드 파일로 교체
if uploaded is not None:
    try:
        data = pd.read_csv(uploaded, parse_dates=["Date"], index_col="Date").sort_index()
        status_msg = "✅ 업로드한 CSV로 데이터가 교체되었습니다."
    except Exception as e:
        st.error(f"업로드 파일을 읽는 중 오류: {e}")

if data is None:
    st.error("데이터가 없습니다. 기본 CSV를 폴더에 두거나 파일을 업로드해주세요.")
    st.stop()

# 필수 컬럼 체크
required_cols = {"Close", "Volume"}
if not required_cols.issubset(set(data.columns)):
    st.error(f"CSV에 필수 열이 없습니다. 필요 열: {required_cols}. 현재 열: {list(data.columns)}")
    st.stop()

st.success(status_msg)

# --------------------- 기간 필터(최근 N년) ---------------------
end = data.index.max()
start = end - pd.DateOffset(years=years)
data = data.loc[start:end].copy()

# --------------------- Strategy Implementations ---------------------
def compute_ma_returns(df: pd.DataFrame, slippage: float) -> pd.Series:
    d = df.copy()
    d["MA_Short"] = d["Close"].rolling(20).mean()
    d["MA_Long"]  = d["Close"].rolling(60).mean()
    d["Signal"] = np.where(d["MA_Short"] > d["MA_Long"], 1, -1)
    ret = d["Signal"].shift(1) * d["Close"].pct_change()
    ret -= slippage * abs(d["Signal"].diff().fillna(0))
    return ret.dropna()

def compute_vol_breakout_returns(df: pd.DataFrame, slippage: float) -> pd.Series:
    d = df.copy()
    d["Vol_Avg"] = d["Volume"].rolling(20).mean()
    signal = np.where(
        (d["Volume"] > 1.5 * d["Vol_Avg"]) & (d["Close"].pct_change() > 0.01), 1,
        np.where((d["Volume"] > 1.5 * d["Vol_Avg"]) & (d["Close"].pct_change() < -0.01), -1, 0)
    )
    d["Signal"] = signal
    ret = d["Signal"].shift(1) * d["Close"].pct_change()
    ret -= slippage * abs(d["Signal"].diff().fillna(0))
    return ret.dropna()

def compute_obv_returns(df: pd.DataFrame, slippage: float) -> pd.Series:
    d = df.copy()
    obv = [0]
    for i in range(1, len(d)):
        if d["Close"].iloc[i] > d["Close"].iloc[i - 1]:
            obv.append(obv[-1] + d["Volume"].iloc[i])
        elif d["Close"].iloc[i] < d["Close"].iloc[i - 1]:
            obv.append(obv[-1] - d["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    d["OBV"] = obv
    d["OBV_Short"] = d["OBV"].rolling(20).mean()
    d["OBV_Long"]  = d["OBV"].rolling(60).mean()
    d["Signal"] = np.where(d["OBV_Short"] > d["OBV_Long"], 1, -1)
    ret = d["Signal"].shift(1) * d["Close"].pct_change()
    ret -= slippage * abs(d["Signal"].diff().fillna(0))
    return ret.dropna()

def compute_vwap_returns(df: pd.DataFrame, slippage: float) -> pd.Series:
    d = df.copy()
    d["Cum_Vol"] = d["Volume"].cumsum()
    d["Cum_PV"]  = (d["Close"] * d["Volume"]).cumsum()
    d["VWAP"]    = d["Cum_PV"] / d["Cum_Vol"]
    d["Signal"] = np.where(d["Close"] > d["VWAP"], 1, -1)
    ret = d["Signal"].shift(1) * d["Close"].pct_change()
    ret -= slippage * abs(d["Signal"].diff().fillna(0))
    return ret.dropna()

strategy_funcs = {
    "이동평균": compute_ma_returns,
    "거래량돌파": compute_vol_breakout_returns,
    "OBV": compute_obv_returns,
    "VWAP": compute_vwap_returns,
}

# --------------------- Metrics ---------------------
def curve_from_returns(returns: pd.Series, init_cap: float) -> pd.Series:
    return (1 + returns).cumprod() * init_cap

def metrics_from_curve(curve: pd.Series, rf_daily: float = 0.0001):
    final_cap = curve.iloc[-1]
    n_years = max(len(curve) / 252, 1e-9)
    cagr = (final_cap / curve.iloc[0]) ** (1 / n_years) - 1
    daily_ret = curve.pct_change().dropna()
    mdd = (curve / curve.cummax() - 1).min()
    sharpe = ((daily_ret.mean() - rf_daily) / (daily_ret.std() + 1e-12)) * np.sqrt(252)
    win_rate = (daily_ret > 0).mean()
    return final_cap, cagr, mdd, sharpe, win_rate

# --------------------- Compute ---------------------
if not chosen_strategies:
    st.warning("전략을 최소 1개 이상 선택하세요.")
    st.stop()

curves = {}
rows = []
for name in chosen_strategies:
    ret = strategy_funcs[name](data, slippage)
    curve = curve_from_returns(ret, initial_capital)
    final_cap, cagr, mdd, sharpe, win_rate = metrics_from_curve(curve)
    curves[name] = curve
    rows.append([
        name, f"{initial_capital:,.2f}", f"{final_cap:,.2f}",
        f"{cagr*100:.2f}%", f"{win_rate*100:.1f}%", f"{mdd*100:.2f}%", f"{sharpe:.2f}"
    ])

# 동일 비중 포트폴리오 (선택 전략 2개 이상)
if len(chosen_strategies) >= 2:
    aligned = pd.concat([curve.pct_change().fillna(0) for curve in curves.values()], axis=1).mean(axis=1)
    port_curve = curve_from_returns(aligned, initial_capital)
    pf_final, pf_cagr, pf_mdd, pf_sharpe, pf_win = metrics_from_curve(port_curve)
    curves["포트폴리오(동일비중)"] = port_curve
    rows.append([
        "포트폴리오(동일비중)",
        f"{initial_capital:,.2f}",
        f"{pf_final:,.2f}",
        f"{pf_cagr*100:.2f}%",
        f"{pf_win*100:.1f}%",
        f"{pf_mdd*100:.2f}%",
        f"{pf_sharpe:.2f}",
    ])

# --------------------- Plot ---------------------
fig = go.Figure()
palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]
for i, (label, curve) in enumerate(curves.items()):
    fig.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines",
                             name=label, line=dict(width=2, color=palette[i % len(palette)])))
fig.update_layout(
    title=f"누적 자산곡선 (초기자금 {initial_capital:,.0f}, 슬리피지 {int(slippage_pct)}%, 최근 {years}년)",
    xaxis_title="날짜",
    yaxis_title="자산가치",
    template="plotly_white",
    legend_title="전략"
)

col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    out = pd.DataFrame(rows, columns=["전략", "초기자금", "최종자금", "CAGR", "승률", "MDD", "샤프"])
    st.dataframe(out, use_container_width=True)

st.success("완료! 기본 CSV 자동 로드, 업로드 시 교체가 가능합니다.")
