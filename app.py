import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="업비트 전략 백테스터", layout="wide")
st.title("📊 업비트 전략 백테스터 (CSV 전체 기간 분석)")

# ============================================================
# 1️⃣ 데이터 로드 (업비트 CSV 호환)
# ============================================================
@st.cache_data
def load_upbit_csv(path: str):
    df = pd.read_csv(path)
    if "date_kst" in df.columns:
        df["Date"] = pd.to_datetime(df["date_kst"], errors="coerce")
    elif "date_utc" in df.columns:
        df["Date"] = pd.to_datetime(df["date_utc"], errors="coerce")
    elif "timestamp" in df.columns:
        df["Date"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    else:
        st.error("❌ CSV에서 날짜 컬럼을 찾을 수 없습니다.")
        st.stop()

    df = df.rename(columns={"close": "Close", "volume": "Volume"})
    df = df[["Date", "Close", "Volume"]].dropna().sort_values("Date").set_index("Date")
    return df

try:
    data = load_upbit_csv("upbit_KRW-BTC_daily_all.csv")
    st.success(f"✅ CSV 로드 성공 ({len(data):,}행)")
except Exception as e:
    st.error(f"❌ CSV 로드 실패: {e}")
    st.stop()

# ============================================================
# 2️⃣ 기본 설정
# ============================================================
st.sidebar.header("⚙️ 설정")
initial_capital = st.sidebar.number_input("초기자금", min_value=1.0, value=100.0, step=1.0)
slippage_pct = st.sidebar.select_slider("슬리피지(%)", options=[0, 1, 2, 3, 4, 5], value=3)
slippage = slippage_pct / 100.0

strategies = ["이동평균", "거래량돌파", "OBV", "VWAP"]
chosen = st.sidebar.multiselect("전략 선택", strategies, default=["이동평균", "VWAP"])
params = {}

# ============================================================
# 3️⃣ 전략별 파라미터
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("🧪 전략별 파라미터")

if "이동평균" in chosen:
    with st.sidebar.expander("이동평균"):
        params["이동평균"] = {
            "short": st.number_input("단기 MA", 2, 200, 20),
            "long": st.number_input("장기 MA", 3, 400, 60)
        }

if "거래량돌파" in chosen:
    with st.sidebar.expander("거래량 돌파"):
        params["거래량돌파"] = {
            "win": st.number_input("거래량 평균기간", 2, 100, 20),
            "mult": st.number_input("거래량 배수", 0.5, 5.0, 1.5, 0.1),
            "up": st.number_input("상승기준(%)", 0.1, 10.0, 1.0),
            "dn": st.number_input("하락기준(%)", -10.0, -0.1, -1.0)
        }

if "OBV" in chosen:
    with st.sidebar.expander("OBV"):
        params["OBV"] = {
            "short": st.number_input("OBV 단기", 2, 100, 20),
            "long": st.number_input("OBV 장기", 3, 400, 60)
        }

if "VWAP" in chosen:
    with st.sidebar.expander("VWAP"):
        params["VWAP"] = {
            "window": st.number_input("VWAP 기간 (0=누적)", 0, 300, 0)
        }

# ============================================================
# 4️⃣ 전략 함수 정의
# ============================================================
def compute_ma(df, short, long, slp):
    d = df.copy()
    d["S"], d["L"] = d["Close"].rolling(short).mean(), d["Close"].rolling(long).mean()
    sig = np.where(d["S"] > d["L"], 1, -1)
    ret = pd.Series(sig, index=d.index).shift(1) * d["Close"].pct_change()
    ret -= slp * abs(pd.Series(sig).diff().fillna(0))
    return ret.dropna(), pd.Series(sig, index=d.index)

def compute_vol(df, win, mult, up, dn, slp):
    d = df.copy()
    d["avgV"] = d["Volume"].rolling(win).mean()
    pct = d["Close"].pct_change()
    upc = (d["Volume"] > mult * d["avgV"]) & (pct > up / 100)
    dnc = (d["Volume"] > mult * d["avgV"]) & (pct < dn / 100)
    sig = np.where(upc, 1, np.where(dnc, -1, 0))
    ret = pd.Series(sig, index=d.index).shift(1) * pct
    ret -= slp * abs(pd.Series(sig).diff().fillna(0))
    return ret.dropna(), pd.Series(sig, index=d.index)

def compute_obv(df, short, long, slp):
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
    d["S"], d["L"] = d["OBV"].rolling(short).mean(), d["OBV"].rolling(long).mean()
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
# 5️⃣ 매매내역
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
        bp, sp = df.loc[b, "Close"], df.loc[s_date, "Close"]
        raw, adj = (sp / bp - 1), (sp / bp - 1) - 2 * slp
        trades.append({
            "매수일": b.date(), "매수가": bp,
            "매도일": s_date.date(), "매도가": sp,
            "수익률(%)": raw * 100,
            "슬리피지반영 수익률(%)": adj * 100
        })
    return pd.DataFrame(trades)

# ============================================================
# 6️⃣ 성과 계산
# ============================================================
def curve_from_returns(ret): return (1 + ret).cumprod() * initial_capital

def metrics(curve):
    if curve.empty:
        return 0, 0, 0, 0, 0  # 빈 데이터 방지
    final = curve.iloc[-1]
    years = max(len(curve) / 252, 1)
    cagr = (final / curve.iloc[0]) ** (1 / years) - 1
    mdd = (curve / curve.cummax() - 1).min()
    daily = curve.pct_change().dropna()
    if daily.empty:
        sharpe, win = 0, 0
    else:
        sharpe = (daily.mean() / daily.std()) * np.sqrt(252)
        win = (daily > 0).mean()
    return final, cagr, mdd, sharpe, win

# ============================================================
# 7️⃣ 실행
# ============================================================
curves, results, logs = {}, [], {}
for s in chosen:
    if s == "이동평균": r, sig = compute_ma(data, **params["이동평균"], slp=slippage)
    elif s == "거래량돌파": r, sig = compute_vol(data, **params["거래량돌파"], slp=slippage)
    elif s == "OBV": r, sig = compute_obv(data, **params["OBV"], slp=slippage)
    elif s == "VWAP": r, sig = compute_vwap(data, **params["VWAP"], slp=slippage)
    else: continue

    curve = curve_from_returns(r)
    final, cagr, mdd, sharpe, win = metrics(curve)
    curves[s] = curve
    results.append([s, f"{initial_capital:.0f}", f"{final:.2f}", f"{cagr*100:.2f}%", f"{mdd*100:.2f}%", f"{sharpe:.2f}", f"{win*100:.1f}%"])
    logs[s] = extract_trades(data, sig, slippage)

# ============================================================
# 8️⃣ 시각화 및 결과 표시
# ============================================================
fig = go.Figure()
for n, c in curves.items():
    if not c.empty:
        fig.add_trace(go.Scatter(x=c.index, y=c, mode="lines", name=n))
fig.update_layout(title="누적 자산곡선", xaxis_title="날짜", yaxis_title="자산가치", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.dataframe(pd.DataFrame(results, columns=["전략", "초기자금", "최종자금", "CAGR", "MDD", "샤프", "승률"]))

st.markdown("### ⬇️ 매매내역 다운로드")
for n, t in logs.items():
    if not t.empty:
        st.download_button(f"{n} 매매내역 CSV 다운로드", t.to_csv(index=False).encode("utf-8-sig"), f"trades_{n}.csv", "text/csv")
    else:
        st.write(f"📄 {n}: 거래 없음")

st.success("✅ 분석 완료 — 전체 CSV 기간 기준 백테스트")
