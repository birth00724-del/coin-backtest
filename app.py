import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="업비트 전략 백테스터", layout="wide")
st.title("📊 업비트 전략 백테스터 (CSV 전체 기간 분석)")

# ============================================================
# 1️⃣ 데이터 로드 (업비트 형식 대응)
# ============================================================
@st.cache_data
def load_upbit_csv(path: str):
    df = pd.read_csv(path)
    # 컬럼 자동 인식
    if "date_kst" in df.columns:
        df["Date"] = pd.to_datetime(df["date_kst"], errors="coerce")
    elif "date_utc" in df.columns:
        df["Date"] = pd.to_datetime(df["date_utc"], errors="coerce")
    elif "timestamp" in df.columns:
        df["Date"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    else:
        st.error("❌ 날짜 컬럼을 찾지 못했습니다.")
        st.stop()

    df = df.rename(columns={"close": "Close", "volume": "Volume"})
    df = df[["Date", "Close", "Volume"]].dropna().sort_values("Date").set_index("Date")
    return df

try:
    data = load_upbit_csv("upbit_KRW-BTC_daily_all.csv")
    st.success(f"✅ CSV 파일 불러오기 완료 ({len(data):,}일치 데이터)")
except Exception as e:
    st.error(f"❌ CSV 파일을 읽는 중 오류 발생: {e}")
    st.stop()

# ============================================================
# 2️⃣ 기본 설정
# ============================================================
st.sidebar.header("⚙️ 설정")
initial_capital = st.sidebar.number_input("초기자금", min_value=1.0, value=100.0, step=1.0)
slippage_pct = st.sidebar.select_slider("슬리피지(%)", options=[0, 1, 2, 3, 4, 5], value=3)
slippage = slippage_pct / 100.0

strategy_options = ["이동평균", "거래량돌파", "OBV", "VWAP"]
chosen_strategies = st.sidebar.multiselect("전략 선택", strategy_options, default=["이동평균", "VWAP"])
params = {}

# ============================================================
# 3️⃣ 전략별 파라미터 UI
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("🧪 전략별 파라미터")

if "이동평균" in chosen_strategies:
    with st.sidebar.expander("이동평균", expanded=True):
        short_n = st.number_input("단기 이동평균", 2, 200, 20)
        long_n = st.number_input("장기 이동평균", 3, 400, 60)
        params["이동평균"] = {"short": short_n, "long": long_n}

if "거래량돌파" in chosen_strategies:
    with st.sidebar.expander("거래량 돌파", expanded=True):
        vol_window = st.number_input("평균 거래량 기간", 2, 100, 20)
        vol_mult = st.number_input("거래량 배수", 0.5, 5.0, 1.5, step=0.1)
        up_thr = st.number_input("상승 기준(%)", 0.1, 10.0, 1.0)
        dn_thr = st.number_input("하락 기준(%)", -10.0, -0.1, -1.0)
        params["거래량돌파"] = {"win": vol_window, "mult": vol_mult, "up": up_thr, "dn": dn_thr}

if "OBV" in chosen_strategies:
    with st.sidebar.expander("OBV 추세", expanded=True):
        obv_s = st.number_input("OBV 단기", 2, 100, 20)
        obv_l = st.number_input("OBV 장기", 3, 400, 60)
        params["OBV"] = {"short": obv_s, "long": obv_l}

if "VWAP" in chosen_strategies:
    with st.sidebar.expander("VWAP", expanded=True):
        vwap_window = st.number_input("VWAP 기간 (0=누적)", 0, 300, 0)
        params["VWAP"] = {"window": vwap_window}

# ============================================================
# 4️⃣ 전략 함수 정의
# ============================================================
def compute_ma(df, short, long, slp):
    d = df.copy()
    d["S"] = d["Close"].rolling(short).mean()
    d["L"] = d["Close"].rolling(long).mean()
    sig = np.where(d["S"] > d["L"], 1, -1)
    ret = pd.Series(sig, index=d.index).shift(1) * d["Close"].pct_change()
    ret -= slp * abs(pd.Series(sig).diff().fillna(0))
    return ret.dropna(), pd.Series(sig, index=d.index)

def compute_vol(df, win, mult, up, dn, slp):
    d = df.copy()
    d["avgV"] = d["Volume"].rolling(win).mean()
    pct = d["Close"].pct_change()
    up_cond = (d["Volume"] > mult * d["avgV"]) & (pct > up / 100)
    dn_cond = (d["Volume"] > mult * d["avgV"]) & (pct < dn / 100)
    sig = np.where(up_cond, 1, np.where(dn_cond, -1, 0))
    ret = pd.Series(sig, index=d.index).shift(1) * pct
    ret -= slp * abs(pd.Series(sig).diff().fillna(0))
    return ret.dropna(), pd.Series(sig, index=d.index)

def compute_obv(df, short, long, slp):
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
    d["S"] = d["OBV"].rolling(short).mean()
    d["L"] = d["OBV"].rolling(long).mean()
    sig = np.where(d["S"] > d["L"], 1, -1)
    ret = pd.Series(sig, index=d.index).shift(1) * d["Close"].pct_change()
    ret -= slp * abs(pd.Series(sig).diff().fillna(0))
    return ret.dropna(), pd.Series(sig, index=d.index)

def compute_vwap(df, win, slp):
    d = df.copy()
    if win == 0:
        d["VWAP"] = (d["Close"] * d["Volume"]).cumsum() / d["Volume"].cumsum()
    else:
        d["VWAP"] = (d["Close"] * d["Volume"]).rolling(win).sum() / d["Volume"].rolling(win).sum()
    sig = np.where(d["Close"] > d["VWAP"], 1, -1)
    ret = pd.Series(sig, index=d.index).shift(1) * d["Close"].pct_change()
    ret -= slp * abs(pd.Series(sig).diff().fillna(0))
    return ret.dropna(), pd.Series(sig, index=d.index)

# ============================================================
# 5️⃣ 매매내역 생성
# ============================================================
def extract_trades(df, sig, slp):
    sig = sig.fillna(0).astype(float)
    buys = sig[(sig.shift(1) <= 0) & (sig > 0)].index
    sells = sig[(sig.shift(1) > 0) & (sig <= 0)].index
    trades = []
    for b in buys:
        s = sells[sells > b]
        if len(s) == 0:
            s_date = df.index[-1]
        else:
            s_date = s[0]
        bp = df.loc[b, "Close"]
        sp = df.loc[s_date, "Close"]
        raw = (sp / bp - 1)
        adj = raw - 2 * slp
        trades.append({
            "매수일": b.date(),
            "매수가": bp,
            "매도일": s_date.date(),
            "매도가": sp,
            "수익률(%)": raw * 100,
            "슬리피지반영 수익률(%)": adj * 100
        })
    return pd.DataFrame(trades)

# ============================================================
# 6️⃣ 실행 및 결과 표시
# ============================================================
def curve_from_returns(ret): return (1 + ret).cumprod() * initial_capital
def metrics(curve):
    final = curve.iloc[-1]
    years = len(curve) / 252
    cagr = (final / curve.iloc[0]) ** (1 / years) - 1
    mdd = (curve / curve.cummax() - 1).min()
    daily = curve.pct_change().dropna()
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252)
    winrate = (daily > 0).mean()
    return final, cagr, mdd, sharpe, winrate

curves, summary, trades_all = {}, [], {}
for s in chosen_strategies:
    if s == "이동평균": r, sig = compute_ma(data, **params["이동평균"], slp=slippage)
    elif s == "거래량돌파": r, sig = compute_vol(data, **params["거래량돌파"], slp=slippage)
    elif s == "OBV": r, sig = compute_obv(data, **params["OBV"], slp=slippage)
    elif s == "VWAP": r, sig = compute_vwap(data, **params["VWAP"], slp=slippage)
    curve = curve_from_returns(r)
    final, cagr, mdd, sharpe, win = metrics(curve)
    curves[s] = curve
    summary.append([s, f"{initial_capital:.0f}", f"{final:.2f}", f"{cagr*100:.2f}%", f"{mdd*100:.2f}%", f"{sharpe:.2f}", f"{win*100:.1f}%"])
    trades_all[s] = extract_trades(data, sig, slippage)

# ============================================================
# 7️⃣ 시각화 및 다운로드
# ============================================================
fig = go.Figure()
for name, c in curves.items():
    fig.add_trace(go.Scatter(x=c.index, y=c, mode="lines", name=name))
fig.update_layout(title="누적자산곡선", xaxis_title="날짜", yaxis_title="자산가치", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.dataframe(pd.DataFrame(summary, columns=["전략", "초기자금", "최종자금", "CAGR", "MDD", "샤프", "승률"]))

st.markdown("### ⬇️ 매매내역 다운로드")
for s, t in trades_all.items():
    if not t.empty:
        st.download_button(f"{s} 매매내역 CSV 다운로드", t.to_csv(index=False).encode("utf-8-sig"), f"trades_{s}.csv", "text/csv")
    else:
        st.write(f"📄 {s}: 거래 없음")

st.success("✅ 분석 완료 — 전체 CSV 기간 기준 백테스트")
