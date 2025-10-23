import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="업비트 전략 백테스터", layout="wide")
st.title("📊 업비트 전략 백테스터 (업비트 CSV 전체기간)")

# ============================================================
# 1) 데이터 로드 (업비트 CSV 호환)
# ============================================================
@st.cache_data
def load_upbit_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 날짜 컬럼 자동 인식 (date_kst → date_utc → timestamp(ms))
    if "date_kst" in df.columns:
        df["Date"] = pd.to_datetime(df["date_kst"], errors="coerce")
    elif "date_utc" in df.columns:
        df["Date"] = pd.to_datetime(df["date_utc"], errors="coerce")
    elif "timestamp" in df.columns:
        df["Date"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    else:
        raise KeyError("CSV에서 날짜 컬럼(date_kst/date_utc/timestamp)을 찾지 못했습니다.")

    # 종가/거래량 컬럼 정규화 (소문자 close/volume 가정, 다른 케이스도 커버)
    price_col = "close" if "close" in df.columns else ("trade_price" if "trade_price" in df.columns else None)
    vol_col   = "volume" if "volume" in df.columns else ("candle_acc_trade_volume" if "candle_acc_trade_volume" in df.columns else None)
    if price_col is None or vol_col is None:
        raise KeyError("CSV에서 종가(close/trade_price) 또는 거래량(volume/candle_acc_trade_volume) 컬럼을 찾지 못했습니다.")

    df = df.rename(columns={price_col: "Close", vol_col: "Volume"})
    df = df[["Date", "Close", "Volume"]].dropna().sort_values("Date").set_index("Date")
    return df

try:
    data = load_upbit_csv("upbit_KRW-BTC_daily_all.csv")
    st.success(f"✅ CSV 로드 성공: {len(data):,}행 (기간: {data.index.min().date()} ~ {data.index.max().date()})")
except Exception as e:
    st.error(f"❌ CSV 로드 실패: {e}")
    st.stop()

# ============================================================
# 2) 공통 설정
# ============================================================
st.sidebar.header("⚙️ 공통 설정")
initial_capital = st.sidebar.number_input("초기자금", min_value=1.0, value=100.0, step=1.0)
slippage_pct = st.sidebar.select_slider("슬리피지(%)", options=[0, 1, 2, 3, 4, 5], value=3)
slippage = slippage_pct / 100.0

strategy_list = ["이동평균", "거래량돌파", "OBV", "VWAP"]
chosen = st.sidebar.multiselect("전략 선택 (복수 가능)", strategy_list, default=["이동평균", "VWAP"])

# ============================================================
# 3) 전략별 파라미터 (선택된 전략만 표시)
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("🧪 전략 파라미터")
params = {}

if "이동평균" in chosen:
    with st.sidebar.expander("이동평균 (MA Cross)", expanded=True):
        short_n = st.number_input("단기 MA", min_value=2, max_value=400, value=20, step=1, key="ma_s")
        long_n  = st.number_input("장기 MA", min_value=3, max_value=600, value=60, step=1, key="ma_l")
        if short_n >= long_n:
            st.warning("단기 MA는 장기 MA보다 작아야 합니다.")
        params["이동평균"] = {"short": int(short_n), "long": int(long_n)}

if "거래량돌파" in chosen:
    with st.sidebar.expander("거래량 돌파 (Volume Breakout)", expanded=True):
        vol_window = st.number_input("거래량 평균기간", min_value=2, max_value=200, value=20, step=1, key="vol_win")
        vol_mult   = st.number_input("거래량 배수", min_value=0.1, max_value=10.0, value=1.5, step=0.1, format="%.1f", key="vol_mult")
        up_thr_pct   = st.number_input("상승 임계(%)", min_value=0.0, max_value=20.0, value=1.0, step=0.1, format="%.1f", key="up_thr")
        down_thr_pct = st.number_input("하락 임계(%)", min_value=-20.0, max_value=0.0, value=-1.0, step=0.1, format="%.1f", key="dn_thr")
        params["거래량돌파"] = {"win": int(vol_window), "mult": float(vol_mult), "up": float(up_thr_pct), "dn": float(down_thr_pct)}

if "OBV" in chosen:
    with st.sidebar.expander("OBV 추세", expanded=True):
        obv_short = st.number_input("OBV 단기", min_value=2, max_value=200, value=20, step=1, key="obv_s")
        obv_long  = st.number_input("OBV 장기", min_value=3, max_value=600, value=60, step=1, key="obv_l")
        if obv_short >= obv_long:
            st.warning("OBV 단기는 장기보다 작아야 합니다.")
        params["OBV"] = {"short": int(obv_short), "long": int(obv_long)}

if "VWAP" in chosen:
    with st.sidebar.expander("VWAP", expanded=True):
        vwap_window = st.number_input("VWAP 기간 (0=누적)", min_value=0, max_value=300, value=0, step=1, key="vwap_w")
        params["VWAP"] = {"window": int(vwap_window)}

# ============================================================
# 4) 전략 함수 (returns, signal 반환)
# ============================================================
def compute_ma(df, short: int, long: int, slp: float):
    d = df.copy()
    d["S"] = d["Close"].rolling(short).mean()
    d["L"] = d["Close"].rolling(long).mean()
    sig = np.where(d["S"] > d["L"], 1, -1)
    sig_s = pd.Series(sig, index=d.index)
    ret = sig_s.shift(1) * d["Close"].pct_change()
    ret -= slp * np.abs(sig_s.diff().fillna(0))
    return ret.dropna(), sig_s

def compute_vol(df, win: int, mult: float, up: float, dn: float, slp: float):
    d = df.copy()
    d["avgV"] = d["Volume"].rolling(win).mean()
    pct = d["Close"].pct_change()
    upc = (d["Volume"] > mult * d["avgV"]) & (pct >  up / 100.0)
    dnc = (d["Volume"] > mult * d["avgV"]) & (pct <  dn / 100.0)
    sig = np.where(upc, 1, np.where(dnc, -1, 0))
    sig_s = pd.Series(sig, index=d.index)
    ret = sig_s.shift(1) * pct
    ret -= slp * np.abs(sig_s.diff().fillna(0))
    return ret.dropna(), sig_s

def compute_obv(df, short: int, long: int, slp: float):
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
    sig_s = pd.Series(sig, index=d.index)
    ret = sig_s.shift(1) * d["Close"].pct_change()
    ret -= slp * np.abs(sig_s.diff().fillna(0))
    return ret.dropna(), sig_s

def compute_vwap(df, window: int, slp: float):
    d = df.copy()
    if window == 0:
        d["VWAP"] = (d["Close"] * d["Volume"]).cumsum() / d["Volume"].cumsum()
    else:
        d["VWAP"] = (d["Close"] * d["Volume"]).rolling(window).sum() / d["Volume"].rolling(window).sum()
    sig = np.where(d["Close"] > d["VWAP"], 1, -1)
    sig_s = pd.Series(sig, index=d.index)
    ret = sig_s.shift(1) * d["Close"].pct_change()
    ret -= slp * np.abs(sig_s.diff().fillna(0))
    return ret.dropna(), sig_s

# ============================================================
# 5) 매매내역 (롱만, 요청 포맷)
# ============================================================
def extract_long_trades(df: pd.DataFrame, signal: pd.Series, slp: float) -> pd.DataFrame:
    sig = signal.fillna(0).astype(float)
    buys  = sig[(sig.shift(1) <= 0) & (sig > 0)].index
    sells = sig[(sig.shift(1) > 0) & (sig <= 0)].index

    trades = []
    si = 0
    for b in buys:
        # b 이후 첫 매도
        future_sells = sells[sells > b]
        if len(future_sells) == 0:
            s_date = df.index[-1]
        else:
            s_date = future_sells[0]

        buy_price  = float(df.loc[b, "Close"])
        sell_price = float(df.loc[s_date, "Close"])
        raw = (sell_price / buy_price) - 1.0
        adj = raw - 2.0 * slp  # 진입 1회 + 청산 1회 슬리피지
        trades.append({
            "매수일": b.date(),
            "매수가": buy_price,
            "매도일": s_date.date(),
            "매도가": sell_price,
            "수익률(%)": raw * 100.0,
            "슬리피지반영 수익률(%)": adj * 100.0
        })
        si += 1

    return pd.DataFrame(trades)

# ============================================================
# 6) 성과 계산 (빈 데이터 안전처리)
# ============================================================
def curve_from_returns(returns: pd.Series, init_cap: float) -> pd.Series:
    if returns is None or len(returns) == 0:
        # 최소 한 점이라도 만들어서 후속 로직 에러 방지
        return pd.Series([init_cap], index=[data.index.min()])
    return (1 + returns).cumprod() * init_cap

def metrics(curve: pd.Series):
    if curve is None or curve.empty:
        return 0, 0, 0, 0, 0
    final = float(curve.iloc[-1])
    years = max(len(curve) / 252, 1)
    cagr = (final / float(curve.iloc[0])) ** (1 / years) - 1
    mdd = float((curve / curve.cummax() - 1).min())
    daily = curve.pct_change().dropna()
    if daily.empty or daily.std() == 0:
        sharpe = 0.0
        win = 0.0
    else:
        sharpe = float((daily.mean() / daily.std()) * np.sqrt(252))
        win = float((daily > 0).mean())
    return final, cagr, mdd, sharpe, win

# ============================================================
# 7) 실행 (안전 가드 포함)
# ============================================================
if not chosen:
    st.warning("전략을 최소 1개 이상 선택하세요.")
    st.stop()

curves, results, trade_logs = {}, [], {}

for s in chosen:
    try:
        if s == "이동평균" and "이동평균" in params and isinstance(params["이동평균"], dict):
            r, sig = compute_ma(data, **params["이동평균"], slp=slippage)
        elif s == "거래량돌파" and "거래량돌파" in params and isinstance(params["거래량돌파"], dict):
            r, sig = compute_vol(data, **params["거래량돌파"], slp=slippage)
        elif s == "OBV" and "OBV" in params and isinstance(params["OBV"], dict):
            r, sig = compute_obv(data, **params["OBV"], slp=slippage)
        elif s == "VWAP" and "VWAP" in params and isinstance(params["VWAP"], dict):
            r, sig = compute_vwap(data, **params["VWAP"], slp=slippage)
        else:
            # 파라미터가 없거나 잘못된 경우 건너뛰기
            st.info(f"ℹ️ '{s}' 전략 파라미터가 설정되지 않아 건너뜁니다.")
            continue

        curve = curve_from_returns(r, initial_capital)
        final, cagr, mdd, sharpe, win = metrics(curve)

        curves[s] = curve
        results.append([
            s, f"{initial_capital:,.0f}", f"{final:,.2f}",
            f"{cagr*100:.2f}%", f"{mdd*100:.2f}%", f"{sharpe:.2f}", f"{win*100:.1f}%"
        ])

        # 매매내역 (롱만, 요청 포맷)
        trade_logs[s] = extract_long_trades(data, sig, slippage)

    except Exception as e:
        st.warning(f"⚠️ '{s}' 전략 실행 중 오류: {e}")

# ============================================================
# 8) 시각화 & 결과표 & CSV 다운로드
# ============================================================
fig = go.Figure()
for name, curve in curves.items():
    if not curve.empty:
        fig.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=name))
fig.update_layout(
    title=f"누적 자산곡선 (초기자금 {initial_capital:,.0f}, 슬리피지 {int(slippage_pct)}%, 전체기간)",
    xaxis_title="날짜", yaxis_title="자산가치", template="plotly_white", legend_title="전략"
)
st.plotly_chart(fig, use_container_width=True)

if results:
    st.subheader("📊 전략별 성과 요약")
    st.dataframe(pd.DataFrame(results, columns=["전략", "초기자금", "최종자금", "CAGR", "MDD", "샤프", "승률"]), use_container_width=True)
else:
    st.info("표시할 성과가 없습니다. 전략과 파라미터를 확인하세요.")

st.subheader("⬇️ 전략별 매매내역 CSV 다운로드 (롱 트레이드)")
for name, tdf in trade_logs.items():
    if tdf is not None and not tdf.empty:
        st.download_button(
            label=f"{name} 매매내역 다운로드",
            data=tdf.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"trades_{name}.csv",
            mime="text/csv",
            key=f"dl_{name}"
        )
    else:
        st.write(f"📄 {name}: 거래 없음")

st.success("✅ 완료! CSV 전체기간 기준 백테스트/로그 생성 성공")
