import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="업비트 전략 백테스터", layout="wide")
st.title("📈 업비트 전략 백테스터")
st.caption("업비트 일봉 CSV (전체 기간)를 기반으로 4가지 전략 백테스트 및 매매내역 CSV 생성")

# ============================================================
# 1️⃣ 데이터 로드
# ============================================================
@st.cache_data
def load_upbit_csv(path: str):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["candle_date_time_kst"])
    df = df.rename(columns={
        "trade_price": "Close",
        "candle_acc_trade_volume": "Volume"
    })
    df = df[["Date", "Close", "Volume"]].sort_values("Date").set_index("Date")
    return df

try:
    data = load_upbit_csv("upbit_KRW-BTC_daily_all.csv")
    st.success(f"✅ 업비트 일봉 데이터 불러오기 완료! ({len(data):,}일치)")
except FileNotFoundError:
    st.error("❌ upbit_KRW-BTC_daily_all.csv 파일이 프로젝트 폴더에 없습니다.")
    st.stop()

# ============================================================
# 2️⃣ 설정
# ============================================================
st.sidebar.header("⚙️ 설정")
initial_capital = st.sidebar.number_input("초기자금", min_value=1.0, value=100.0, step=1.0)
slippage_pct = st.sidebar.select_slider("슬리피지(%)", options=[0, 1, 2, 3, 4, 5], value=3)
slippage = slippage_pct / 100.0

strategy_options = ["이동평균", "거래량돌파", "OBV", "VWAP"]
chosen_strategies = st.sidebar.multiselect("전략 선택", strategy_options, default=["이동평균", "VWAP"])

params = {}

# ============================================================
# 3️⃣ 전략별 파라미터 (선택된 전략만 표시)
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("🧪 전략 파라미터")

if "이동평균" in chosen_strategies:
    with st.sidebar.expander("이동평균 (MA Cross)", expanded=True):
        short_n = st.number_input("단기 MA 기간", min_value=2, value=20, step=1)
        long_n = st.number_input("장기 MA 기간", min_value=3, value=60, step=1)
        params["이동평균"] = {"short": int(short_n), "long": int(long_n)}

if "거래량돌파" in chosen_strategies:
    with st.sidebar.expander("거래량 돌파 (Volume Breakout)", expanded=True):
        vol_window = st.number_input("거래량 평균 기간", min_value=2, value=20, step=1)
        vol_mult = st.number_input("거래량 배수", min_value=0.1, value=1.5, step=0.1)
        up_thr_pct = st.number_input("상승 임계 수익률(%)", value=1.0, step=0.1)
        down_thr_pct = st.number_input("하락 임계 수익률(%)", value=-1.0, step=0.1)
        params["거래량돌파"] = {"win": int(vol_window), "mult": float(vol_mult),
                            "up": float(up_thr_pct), "dn": float(down_thr_pct)}

if "OBV" in chosen_strategies:
    with st.sidebar.expander("OBV 추세", expanded=True):
        obv_short = st.number_input("OBV 단기 기간", min_value=2, value=20, step=1)
        obv_long = st.number_input("OBV 장기 기간", min_value=3, value=60, step=1)
        params["OBV"] = {"short": int(obv_short), "long": int(obv_long)}

if "VWAP" in chosen_strategies:
    with st.sidebar.expander("VWAP", expanded=True):
        vwap_window = st.number_input("VWAP 기간 (0=누적)", min_value=0, value=0, step=1)
        params["VWAP"] = {"window": int(vwap_window)}

# ============================================================
# 4️⃣ 전략 정의
# ============================================================
def compute_ma(df, short_n, long_n, slippage):
    d = df.copy()
    d["MA_Short"] = d["Close"].rolling(short_n).mean()
    d["MA_Long"] = d["Close"].rolling(long_n).mean()
    sig = np.where(d["MA_Short"] > d["MA_Long"], 1, -1)
    ret = pd.Series(sig, index=d.index).shift(1) * d["Close"].pct_change()
    ret -= slippage * abs(pd.Series(sig, index=d.index).diff().fillna(0))
    return ret.dropna(), pd.Series(sig, index=d.index)

def compute_vol(df, win, mult, up_thr, dn_thr, slippage):
    d = df.copy()
    d["Vol_Avg"] = d["Volume"].rolling(win).mean()
    pct = d["Close"].pct_change()
    up = (d["Volume"] > mult * d["Vol_Avg"]) & (pct > up_thr/100)
    dn = (d["Volume"] > mult * d["Vol_Avg"]) & (pct < dn_thr/100)
    sig = np.where(up, 1, np.where(dn, -1, 0))
    ret = pd.Series(sig, index=d.index).shift(1) * pct
    ret -= slippage * abs(pd.Series(sig, index=d.index).diff().fillna(0))
    return ret.dropna(), pd.Series(sig, index=d.index)

def compute_obv(df, short_n, long_n, slippage):
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
    d["OBV_Long"] = d["OBV"].rolling(long_n).mean()
    sig = np.where(d["OBV_Short"] > d["OBV_Long"], 1, -1)
    ret = pd.Series(sig, index=d.index).shift(1) * d["Close"].pct_change()
    ret -= slippage * abs(pd.Series(sig, index=d.index).diff().fillna(0))
    return ret.dropna(), pd.Series(sig, index=d.index)

def compute_vwap(df, window, slippage):
    d = df.copy()
    if window > 0:
        num = (d["Close"] * d["Volume"]).rolling(window).sum()
        den = d["Volume"].rolling(window).sum()
        d["VWAP"] = num / den
    else:
        d["VWAP"] = (d["Close"] * d["Volume"]).cumsum() / d["Volume"].cumsum()
    sig = np.where(d["Close"] > d["VWAP"], 1, -1)
    ret = pd.Series(sig, index=d.index).shift(1) * d["Close"].pct_change()
    ret -= slippage * abs(pd.Series(sig, index=d.index).diff().fillna(0))
    return ret.dropna(), pd.Series(sig, index=d.index)

def run_strategy(name, df):
    if name == "이동평균":
        p = params["이동평균"]; return compute_ma(df, p["short"], p["long"], slippage)
    if name == "거래량돌파":
        p = params["거래량돌파"]; return compute_vol(df, p["win"], p["mult"], p["up"], p["dn"], slippage)
    if name == "OBV":
        p = params["OBV"]; return compute_obv(df, p["short"], p["long"], slippage)
    if name == "VWAP":
        p = params["VWAP"]; return compute_vwap(df, p["window"], slippage)
    raise ValueError("Unknown strategy")

# ============================================================
# 5️⃣ 매매내역 생성
# ============================================================
def extract_trades(df, signal, slippage):
    sig = signal.fillna(0).astype(float)
    entry_idx = sig[(sig.shift(1) <= 0) & (sig > 0)].index
    exit_idx = sig[(sig.shift(1) > 0) & (sig <= 0)].index

    trades = []
    ei = xi = 0
    while ei < len(entry_idx):
        buy_date = entry_idx[ei]
        sell_date = None
        while xi < len(exit_idx):
            if exit_idx[xi] > buy_date:
                sell_date = exit_idx[xi]
                xi += 1
                break
            xi += 1
        if sell_date is None:
            sell_date = df.index[-1]
        buy_price = df.loc[buy_date, "Close"]
        sell_price = df.loc[sell_date, "Close"]
        raw = (sell_price / buy_price) - 1
        adj = raw - 2 * slippage
        trades.append({
            "매수일": buy_date.date(),
            "매수가": buy_price,
            "매도일": sell_date.date(),
            "매도가": sell_price,
            "수익률(%)": raw * 100,
            "슬리피지반영 수익률(%)": adj * 100
        })
        ei += 1
    return pd.DataFrame(trades)

# ============================================================
# 6️⃣ 성과 계산 및 시각화
# ============================================================
def curve_from_returns(returns, init_cap):
    return (1 + returns).cumprod() * init_cap

def metrics(curve):
    final_cap = curve.iloc[-1]
    n_years = len(curve) / 252
    cagr = (final_cap / curve.iloc[0]) ** (1/n_years) - 1
    mdd = (curve / curve.cummax() - 1).min()
    daily_ret = curve.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    win_rate = (daily_ret > 0).mean()
    return final_cap, cagr, mdd, sharpe, win_rate

curves, results, trades_all = {}, [], {}
for s in chosen_strategies:
    rets, sig = run_strategy(s, data)
    curve = curve_from_returns(rets, initial_capital)
    final, cagr, mdd, sharpe, win = metrics(curve)
    curves[s] = curve
    results.append([s, f"{initial_capital:.2f}", f"{final:.2f}", f"{cagr*100:.2f}%", f"{mdd*100:.2f}%", f"{sharpe:.2f}", f"{win*100:.1f}%"])
    trades_all[s] = extract_trades(data, sig, slippage)

# ============================================================
# 7️⃣ 출력
# ============================================================
fig = go.Figure()
for i, (name, curve) in enumerate(curves.items()):
    fig.add_trace(go.Scatter(x=curve.index, y=curve, mode="lines", name=name))
fig.update_layout(title="누적 자산곡선", xaxis_title="날짜", yaxis_title="자산가치", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

summary = pd.DataFrame(results, columns=["전략", "초기자금", "최종자금", "CAGR", "MDD", "샤프비율", "승률"])
st.dataframe(summary, use_container_width=True)

st.markdown("### ⬇️ 전략별 매매내역 CSV 다운로드")
for s in chosen_strategies:
    t = trades_all[s]
    if t.empty:
        st.write(f"📄 {s}: 거래 없음")
    else:
        csv = t.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label=f"{s} 매매내역 다운로드",
            data=csv,
            file_name=f"trades_{s}.csv",
            mime="text/csv"
        )

st.success("완료! 전체 기간 데이터 기반 전략 테스트 완료 ✅")
