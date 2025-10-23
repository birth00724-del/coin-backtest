import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="업비트 전략 백테스터", layout="wide")
st.title("📈 업비트 전략 백테스터")
st.caption("기본 CSV 자동 로드 + 업로드 교체 가능 / 전략 선택 시 해당 파라미터만 표시 / 매매내역 CSV 다운로드 지원")

# --------------------- Sidebar: Global Controls ---------------------
st.sidebar.header("⚙️ 공통 설정")
initial_capital = st.sidebar.number_input("초기자금", min_value=1.0, value=100.0, step=1.0)
slippage_pct = st.sidebar.select_slider("슬리피지(%)", options=[0, 1, 2, 3, 4, 5], value=3)
slippage = slippage_pct / 100.0
years = st.sidebar.slider("분석 기간(년)", min_value=1, max_value=5, value=5, step=1)

# 데이터 소스
st.sidebar.markdown("---")
st.sidebar.subheader("데이터 소스")
uploaded = st.sidebar.file_uploader("CSV 업로드(선택)", type=["csv"])
use_default = st.sidebar.checkbox("기본 CSV 사용(프로젝트 폴더)", value=True)

# --------------------- Load Data ---------------------
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

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded, parse_dates=["Date"], index_col="Date").sort_index()
        status_msg = "✅ 업로드한 CSV로 데이터가 교체되었습니다."
    except Exception as e:
        st.error(f"업로드 파일을 읽는 중 오류: {e}")

if data is None:
    st.error("데이터가 없습니다. 기본 CSV를 폴더에 두거나 파일을 업로드해주세요.")
    st.stop()

required_cols = {"Close", "Volume"}
if not required_cols.issubset(data.columns):
    st.error(f"CSV에 필수 열이 없습니다. 필요 열: {required_cols}. 현재 열: {list(data.columns)}")
    st.stop()

st.success(status_msg)

# 기간 필터: 최근 N년
end = data.index.max()
start = end - pd.DateOffset(years=years)
data = data.loc[start:end].copy()

# --------------------- Strategy selection ---------------------
st.sidebar.markdown("---")
st.sidebar.header("📌 전략 선택")
strategy_options = ["이동평균", "거래량돌파", "OBV", "VWAP"]
chosen_strategies = st.sidebar.multiselect("전략(복수 선택 가능)", strategy_options, default=["이동평균", "VWAP"])

# --------------------- Strategy Parameters (only for selected) ---------------------
st.sidebar.markdown("---")
st.sidebar.header("🧪 전략 파라미터")
params = {}

if "이동평균" in chosen_strategies:
    with st.sidebar.expander("이동평균 (MA Cross)", expanded=True):
        short_n = st.number_input("단기 MA 기간", min_value=2, value=20, step=1, key="ma_short")
        long_n  = st.number_input("장기 MA 기간", min_value=3, value=60, step=1, key="ma_long")
        if short_n >= long_n:
            st.warning("이동평균: 단기 기간은 장기 기간보다 작아야 합니다.")
        params["이동평균"] = {"short": int(short_n), "long": int(long_n)}

if "거래량돌파" in chosen_strategies:
    with st.sidebar.expander("거래량 돌파 (Volume Breakout)", expanded=True):
        vol_window = st.number_input("거래량 평균 기간", min_value=2, value=20, step=1, key="vol_win")
        vol_mult   = st.number_input("거래량 배수 (예: 1.5)", min_value=0.1, value=1.5, step=0.1, format="%.2f", key="vol_mult")
        up_thr_pct   = st.number_input("상승 임계 수익률(%)", value=1.0, step=0.1, format="%.2f", key="up_thr")
        down_thr_pct = st.number_input("하락 임계 수익률(%)", value=-1.0, step=0.1, format="%.2f", key="dn_thr")
        params["거래량돌파"] = {"win": int(vol_window), "mult": float(vol_mult), "up": float(up_thr_pct), "dn": float(down_thr_pct)}

if "OBV" in chosen_strategies:
    with st.sidebar.expander("OBV 추세 (OBV Trend)", expanded=True):
        obv_short = st.number_input("OBV 단기 기간", min_value=2, value=20, step=1, key="obv_short")
        obv_long  = st.number_input("OBV 장기 기간", min_value=3, value=60, step=1, key="obv_long")
        if obv_short >= obv_long:
            st.warning("OBV: 단기 기간은 장기 기간보다 작아야 합니다.")
        params["OBV"] = {"short": int(obv_short), "long": int(obv_long)}

if "VWAP" in chosen_strategies:
    with st.sidebar.expander("VWAP", expanded=True):
        vwap_window = st.number_input("VWAP 기간 (0=누적)", min_value=0, value=0, step=1, key="vwap_win")
        vwap_alpha  = st.number_input("VWAP 필터 α(%) (0=미사용)", min_value=0.0, value=0.0, step=0.1, format="%.1f", key="vwap_alpha")
        params["VWAP"] = {"window": int(vwap_window), "alpha": float(vwap_alpha)}

# --------------------- Strategy Implementations → (returns, signal) ---------------------
def compute_ma_returns(df: pd.DataFrame, short_n: int, long_n: int, slippage: float):
    d = df.copy()
    d["MA_Short"] = d["Close"].rolling(short_n).mean()
    d["MA_Long"]  = d["Close"].rolling(long_n).mean()
    signal = np.where(d["MA_Short"] > d["MA_Long"], 1, -1)
    sig = pd.Series(signal, index=d.index)
    ret = sig.shift(1) * d["Close"].pct_change()
    ret -= slippage * np.abs(sig.diff().fillna(0))
    return ret.dropna(), sig

def compute_vol_breakout_returns(df: pd.DataFrame, win: int, mult: float, up_thr: float, dn_thr: float, slippage: float):
    d = df.copy()
    d["Vol_Avg"] = d["Volume"].rolling(win).mean()
    pct = d["Close"].pct_change()
    up = (d["Volume"] > mult * d["Vol_Avg"]) & (pct >  up_thr/100.0)
    dn = (d["Volume"] > mult * d["Vol_Avg"]) & (pct <  dn_thr/100.0)
    signal = np.where(up, 1, np.where(dn, -1, 0))
    sig = pd.Series(signal, index=d.index)
    ret = sig.shift(1) * pct
    ret -= slippage * np.abs(sig.diff().fillna(0))
    return ret.dropna(), sig

def compute_obv_returns(df: pd.DataFrame, short_n: int, long_n: int, slippage: float):
    d = df.copy()
    obv = [0]
    for i in range(1, len(d)):
        if d["Close"].iloc[i] > d["Close"].iloc[i-1]:
            obv.append(obv[-1] + d["Volume"].iloc[i])
        elif d["Close"].iloc[i] < d["Close"].iloc[i-1]:
            obv.append(obv[-1] - d["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    d["OBV"] = obv
    d["OBV_Short"] = d["OBV"].rolling(short_n).mean()
    d["OBV_Long"]  = d["OBV"].rolling(long_n).mean()
    signal = np.where(d["OBV_Short"] > d["OBV_Long"], 1, -1)
    sig = pd.Series(signal, index=d.index)
    ret = sig.shift(1) * d["Close"].pct_change()
    ret -= slippage * np.abs(sig.diff().fillna(0))
    return ret.dropna(), sig

def compute_vwap_returns(df: pd.DataFrame, window: int, alpha_pct: float, slippage: float):
    d = df.copy()
    if window > 0:
        num = (d["Close"] * d["Volume"]).rolling(window).sum()
        den = d["Volume"].rolling(window).sum()
        d["VWAP"] = num / den.replace(0, np.nan)
    else:
        d["Cum_Vol"] = d["Volume"].cumsum()
        d["Cum_PV"]  = (d["Close"] * d["Volume"]).cumsum()
        d["VWAP"]    = d["Cum_PV"] / d["Cum_Vol"].replace(0, np.nan)

    if alpha_pct > 0:
        up = d["Close"] > d["VWAP"] * (1 + alpha_pct/100.0)
        dn = d["Close"] < d["VWAP"] * (1 - alpha_pct/100.0)
        signal = np.where(up, 1, np.where(dn, -1, 0))
    else:
        signal = np.where(d["Close"] > d["VWAP"], 1, -1)

    sig = pd.Series(signal, index=d.index)
    ret = sig.shift(1) * d["Close"].pct_change()
    ret -= slippage * np.abs(sig.diff().fillna(0))
    return ret.dropna(), sig

def run_strategy(name: str, df: pd.DataFrame):
    if name == "이동평균":
        p = params["이동평균"];  return compute_ma_returns(df, p["short"], p["long"], slippage)
    if name == "거래량돌파":
        p = params["거래량돌파"]; return compute_vol_breakout_returns(df, p["win"], p["mult"], p["up"], p["dn"], slippage)
    if name == "OBV":
        p = params["OBV"];      return compute_obv_returns(df, p["short"], p["long"], slippage)
    if name == "VWAP":
        p = params["VWAP"];     return compute_vwap_returns(df, p["window"], p["alpha"], slippage)
    raise ValueError("알 수 없는 전략 이름")

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

# --------------------- Trade Log Generator ---------------------
def generate_trade_log(df: pd.DataFrame, signal: pd.Series, slippage: float) -> pd.DataFrame:
    """
    체결가는 '신호가 변한 날의 종가'로 기록.
    직전 체결가 대비 구간 수익률(raw_return)을 계산하고,
    adj_return = raw_return - slippage * abs(new_pos - old_pos) 로 조정.
    """
    sig = signal.dropna().astype(float)
    if sig.empty:
        return pd.DataFrame()

    # 신호가 바뀌는 시점(포지션 변경 시점)
    sig_change = sig.diff().fillna(0)
    change_idx = list(sig_change[sig_change != 0].index)

    if not change_idx:
        return pd.DataFrame()  # 트레이드 없음

    logs = []
    # 초기 엔트리: 첫 변화 이전의 포지션을 알기 위해 이전 값 가져오기
    prev_idx = change_idx[0]
    # 첫 체결은 prev_idx 시점에서 old_pos -> new_pos로 변경
    old_pos = sig.loc[:prev_idx].iloc[:-1].iloc[-1] if len(sig.loc[:prev_idx]) > 1 else 0.0
    entry_time = prev_idx
    entry_price = df.loc[entry_time, "Close"]
    # 첫 체결 로그(진입 자체에 대한 return은 계산 대상 아님, 다음 체결 때 계산)
    logs.append({
        "date": entry_time, "action": "ENTER",
        "old_pos": float(old_pos), "new_pos": float(sig.loc[entry_time]),
        "price": float(entry_price),
        "raw_return_%": np.nan, "adj_return_%": np.nan,
        "slippage_applied_%": slippage * abs(sig.loc[entry_time] - old_pos) * 100
    })
    last_pos = sig.loc[entry_time]
    last_price = entry_price
    last_time = entry_time

    # 이후 변경들: 각 변경 시 이전 구간의 수익률을 계산
    for t in change_idx[1:]:
        px = df.loc[t, "Close"]
        new_pos = sig.loc[t]

        # 직전 포지션(last_pos)으로 last_time→t 구간 보유했다고 가정
        if last_pos > 0:  # long
            raw_ret = (px / last_price) - 1.0
        elif last_pos < 0:  # short
            raw_ret = (last_price / px) - 1.0
        else:  # flat
            raw_ret = 0.0

        slip = slippage * abs(new_pos - last_pos)
        adj_ret = raw_ret - slip

        logs.append({
            "date": t,
            "action": "SWITCH",
            "old_pos": float(last_pos),
            "new_pos": float(new_pos),
            "price": float(px),
            "raw_return_%": raw_ret * 100.0,
            "adj_return_%": adj_ret * 100.0,
            "slippage_applied_%": slip * 100.0
        })

        last_pos = new_pos
        last_price = px
        last_time = t

    # 마지막 구간을 종가로 정산(마지막 날짜)
    final_t = df.index[-1]
    if final_t > last_time:
        px = df.loc[final_t, "Close"]
        # 마지막 시점에 포지션을 '유지'한 채 평가손익을 계산하고 'EXIT'로 표기(평가상 정리)
        if last_pos > 0:
            raw_ret = (px / last_price) - 1.0
        elif last_pos < 0:
            raw_ret = (last_price / px) - 1.0
        else:
            raw_ret = 0.0

        # 마지막은 포지션 유지 → 0으로 청산한다고 가정하면 슬리피지 × |0 - last_pos|
        slip = slippage * abs(0 - last_pos)
        adj_ret = raw_ret - slip

        logs.append({
            "date": final_t,
            "action": "EXIT",
            "old_pos": float(last_pos),
            "new_pos": 0.0,
            "price": float(px),
            "raw_return_%": raw_ret * 100.0,
            "adj_return_%": adj_ret * 100.0,
            "slippage_applied_%": slip * 100.0
        })

    trade_log = pd.DataFrame(logs).sort_values("date").reset_index(drop=True)
    return trade_log

# --------------------- Compute & Output ---------------------
if not chosen_strategies:
    st.warning("전략을 최소 1개 이상 선택하세요.")
    st.stop()

curves = {}
rows = []
trade_logs = {}

for name in chosen_strategies:
    rets, sig = run_strategy(name, data)
    curve = curve_from_returns(rets, initial_capital)
    final_cap, cagr, mdd, sharpe, win_rate = metrics_from_curve(curve)
    curves[name] = curve
    rows.append([
        name, f"{initial_capital:,.2f}", f"{final_cap:,.2f}",
        f"{cagr*100:.2f}%", f"{win_rate*100:.1f}%", f"{mdd*100:.2f}%", f"{sharpe:.2f}"
    ])

    # 매매내역 생성 (체결가는 종가, 수익률은 슬리피지 차감 방식)
    log_df = generate_trade_log(data, sig, slippage)
    trade_logs[name] = log_df

# 동일 비중 포트폴리오(차트/요약만, 로그는 개별 전략 위주)
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

    st.markdown("### ⬇️ 전략별 매매내역 CSV 다운로드")
    for name in chosen_strategies:
        log_df = trade_logs.get(name)
        if log_df is None or log_df.empty:
            st.button(f"{name} 로그 (거래 없음)", disabled=True, key=f"btn_{name}")
        else:
            csv = log_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label=f"{name} 매매내역 CSV 다운로드",
                data=csv,
                file_name=f"tradelog_{name}.csv",
                mime="text/csv",
                key=f"dl_{name}"
            )

st.success("완료! 매매내역 CSV에 raw_return과 슬리피지 차감 adj_return이 포함됩니다.")
