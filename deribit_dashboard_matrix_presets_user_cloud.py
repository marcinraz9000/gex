
# deribit_dashboard_matrix_presets_user_cloud.py
# Cloud-safe Streamlit dashboard (Deribit options) with retry + endpoint fallbacks.
# Features:
# - Net GEX by strike (+ cumulative line)
# - GEX heatmap (Price × IV), IV anchored to DVOL
# - Greeks by strike
# - IV smiles + RR25/BF25
# - OI/GEX matrix (strike × expiry) with auto-filter presets
# - Expiry presets buttons (KF View, Monthlies, Full Curve)
#
# Notes:
# - This 'cloud' version is defensive: all Deribit HTTP calls have retries,
#   two endpoint fallbacks, and return empty frames on persistent failure so
#   the app renders with warnings instead of crashing.
# - Reduce 'Threads' when running on Streamlit Cloud to be friendly.

import os
import time
import math
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------
# Config
# ----------------------------
DERIBIT_HOSTS = [
    os.environ.get("DERIBIT_HOST", "https://www.deribit.com/api/v2").rstrip("/"),
    "https://deribit.com/api/v2",
]
HEADERS = {"User-Agent": "gex-streamlit/1.0 (+streamlit-cloud)"}

st.set_page_config(page_title="BTC/ETH – Options Metrics (Cloud)", layout="wide")

def _utcnow_str() -> str:
    return f"{datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC"

# ----------------------------
# HTTP helper with retries
# ----------------------------
def http_get(path: str, params=None, *, retries=3, timeout=(12, 20)):
    """
    Try all hosts with simple backoff; raise last error if all fail.
    """
    last_exc = None
    for host in DERIBIT_HOSTS:
        url = f"{host}{path if path.startswith('/') else '/' + path}"
        for attempt in range(retries):
            try:
                r = requests.get(url, params=params or {}, headers=HEADERS, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                last_exc = e
                # small exponential backoff
                sleep = 0.6 * (2 ** attempt)
                time.sleep(sleep)
                continue
        # try next host
    # All failed
    if last_exc:
        raise last_exc
    raise RuntimeError("HTTP failure")

# ----------------------------
# Data fetchers (defensive)
# ----------------------------
@st.cache_data(ttl=60 * 15, show_spinner=False)
def get_instruments(currency: str) -> pd.DataFrame:
    try:
        res = http_get("/public/get_instruments",
                       {"currency": currency, "kind": "option", "expired": False})
    except Exception:
        # fallback: without 'expired' filter
        try:
            res = http_get("/public/get_instruments",
                           {"currency": currency, "kind": "option"})
        except Exception:
            return pd.DataFrame(columns=["instrument_name","strike","option_type","expiration_timestamp","expiry_ymd"])
    items = res.get("result", []) or []
    rows = []
    for x in items:
        ts_s = float(x.get("expiration_timestamp", 0) or 0) / 1000.0
        rows.append({
            "instrument_name": x.get("instrument_name"),
            "strike": float(x.get("strike", 0) or 0),
            "option_type": ("C" if str(x.get("option_type","")).lower().startswith("call") else "P"),
            "expiration_timestamp": ts_s,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["expiry_ymd"] = pd.to_datetime(df["expiration_timestamp"], unit="s", utc=True).dt.strftime("%y%m%d")
    return df.sort_values(["expiration_timestamp","strike"]).reset_index(drop=True)

@st.cache_data(ttl=30, show_spinner=False)
def get_book_summary(currency: str) -> pd.DataFrame:
    try:
        res = http_get("/public/get_book_summary_by_currency",
                       {"currency": currency, "kind": "option"})
    except Exception:
        return pd.DataFrame(columns=["instrument_name","open_interest","underlying_price"])
    items = res.get("result", []) or []
    rows = []
    for x in items:
        rows.append({
            "instrument_name": x.get("instrument_name"),
            "open_interest": float(x.get("open_interest", 0) or 0),
            "underlying_price": float(x.get("underlying_price", 0) or 0),
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=20, show_spinner=False)
def get_index_price(currency: str) -> float:
    idx = "btc_usd" if currency.upper()=="BTC" else "eth_usd"
    try:
        res = http_get("/public/get_index_price", {"index_name": idx})
        return float(res["result"]["index_price"])
    except Exception:
        return float("nan")

@st.cache_data(ttl=30, show_spinner=False)
def get_mark_iv_batch(instrument_names, threads: int = 8) -> pd.DataFrame:
    names = list(dict.fromkeys(instrument_names))
    def _one(nm):
        try:
            res = http_get("/public/ticker", {"instrument_name": nm})
            r = res.get("result", {}) or {}
            return {"instrument_name": nm, "mark_iv": float(r.get("mark_iv", 0) or 0)}
        except Exception:
            return {"instrument_name": nm, "mark_iv": 0.0}
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
        futures = {ex.submit(_one, nm): nm for nm in names}
        for fut in as_completed(futures):
            rows.append(fut.result())
    return pd.DataFrame(rows)

@st.cache_data(ttl=120, show_spinner=False)
def get_dvol_anchor(currency: str) -> float:
    try:
        res = http_get("/public/get_volatility_index_data", {
            "currency": currency, "start_timestamp": int((time.time()-2*86400)*1000)
        })
        data = res.get("result")
        if isinstance(data, list) and data:
            last = data[-1]
            if isinstance(last, dict):
                v = float(last.get("volatility", last.get("value", np.nan)))
            elif isinstance(last, (list, tuple)) and len(last) >= 2:
                v = float(last[1])
            else:
                v = float(last)
            return v/100.0 if v > 3 else v
    except Exception:
        pass
    return float("nan")

# ----------------------------
# BS helpers
# ----------------------------
def days_to_years(exp_ts_s: float) -> float:
    now = time.time()
    T = max(float(exp_ts_s) - now, 0.0) / (365.0*24.0*3600.0)
    return T if T > 1e-6 else 1e-6

def n_pdf(x):
    x = np.asarray(x, float)
    return np.exp(-0.5*x*x) / np.sqrt(2.0*np.pi)

def n_cdf(x):
    x = np.asarray(x, float)
    t = 1.0/(1.0 + 0.2316419*np.abs(x))
    a1,a2,a3,a4,a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
    m = 1.0 - n_pdf(x)*poly
    return np.where(x>=0.0, m, 1.0-m)

def d1_d2(S,K,T,sigma):
    S = np.asarray(S,float); K = np.asarray(K,float)
    T = np.maximum(np.asarray(T,float), 1e-6)
    sigma = np.maximum(np.asarray(sigma,float), 1e-6)
    d1 = (np.log(S/K) + 0.5*sigma*sigma*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return d1,d2

def bs_gamma(S,K,T,sigma):
    d1,_ = d1_d2(S,K,T,sigma)
    return n_pdf(d1)/(S*sigma*np.sqrt(T))

def bs_delta_call(S,K,T,sigma):
    d1,_ = d1_d2(S,K,T,sigma)
    return n_cdf(d1)

def bs_vega(S,K,T,sigma):
    d1,_ = d1_d2(S,K,T,sigma)
    return S*n_pdf(d1)*np.sqrt(T)

def bs_theta_call(S,K,T,sigma,r=0.0,q=0.0):
    d1,d2 = d1_d2(S,K,T,sigma)
    term1 = -(S*np.exp(-q*T)*n_pdf(d1)*sigma)/(2*np.sqrt(T))
    term2 = - r*K*np.exp(-r*T)*(1.0 - n_cdf(d2))
    term3 = + q*S*np.exp(-q*T)*n_cdf(d1)
    return term1 + term2 + term3

# ----------------------------
# Builders
# ----------------------------
def build_chain_df(currency: str, expiries_sel, strike_window_pct: float, threads:int=8):
    ins = get_instruments(currency)
    if expiries_sel:
        ins = ins[ins["expiry_ymd"].isin(expiries_sel)].copy()
    if ins.empty:
        return pd.DataFrame(), float("nan")
    spot = get_index_price(currency)
    if not np.isfinite(spot):
        return pd.DataFrame(), float("nan")
    if strike_window_pct and strike_window_pct > 0:
        lo,hi = spot*(1.0-strike_window_pct), spot*(1.0+strike_window_pct)
        ins = ins[(ins["strike"]>=lo) & (ins["strike"]<=hi)].copy()
    bs = get_book_summary(currency)
    df = ins.merge(bs, on="instrument_name", how="left")
    df["open_interest"] = df["open_interest"].fillna(0.0)
    ivs = get_mark_iv_batch(df["instrument_name"].tolist(), threads=threads)
    df = df.merge(ivs, on="instrument_name", how="left")
    df["mark_iv"] = df["mark_iv"].replace([None,np.nan], 0.0).clip(1e-4, 500.0)
    df["mark_iv"] = np.where(df["mark_iv"]>3.0, df["mark_iv"]/100.0, df["mark_iv"])
    return df, float(spot)

def build_gex_by_strike(df_chain, spot, *, contract_size=1.0, per_1pct=False, sign_mode="calls_plus_puts_minus"):
    df = df_chain.copy()
    side = df["option_type"].str.upper().str[0].values
    if sign_mode=="calls_plus_puts_minus":
        sign = np.where(side=="C", +1.0, -1.0)
    elif sign_mode=="both_positive":
        sign = np.ones(len(df), float)
    elif sign_mode=="dealer_short_all":
        sign = -np.ones(len(df), float)
    else:
        sign = np.where(side=="C", +1.0, -1.0)
    T = df["expiration_timestamp"].apply(days_to_years).values
    gamma = bs_gamma(float(spot), df["strike"].values.astype(float), T, df["mark_iv"].values.astype(float))
    scale = (float(spot)*0.01) if per_1pct else 1.0
    contrib = gamma * (float(spot)**2) * scale * df["open_interest"].values.astype(float) * float(contract_size) * sign
    df["_gex"] = contrib
    agg = df.groupby("strike")["_gex"].sum().rename("net_gex").to_frame().reset_index().sort_values("strike")
    if not agg.empty:
        agg["cum_gex"] = agg["net_gex"].cumsum()
    flip = float(agg.loc[agg["net_gex"].abs().idxmin(), "strike"]) if not agg.empty else float("nan")
    hdr = {
        "calls_oi": float(df.loc[side=="C","open_interest"].sum()),
        "puts_oi": float(df.loc[side=="P","open_interest"].sum()),
        "flip": flip,
        "sum_net": float(agg["net_gex"].sum())
    }
    return agg, hdr

def build_iv_smile(df_chain, spot, *, mny_range=(0.85,1.20), mode="rolling", window=5, min_iv=0.01):
    if df_chain.empty:
        return pd.DataFrame(columns=["expiry_ymd","moneyness","mark_iv"])
    df = df_chain.copy()
    df = df[(df["mark_iv"]>min_iv) & (df["mark_iv"]<2.0)].copy()
    df["moneyness"] = df["strike"].astype(float) / float(spot)
    lo,hi = float(mny_range[0]), float(mny_range[1])
    df = df[(df["moneyness"]>=lo) & (df["moneyness"]<=hi)]
    rows = []
    for exp,d in df.groupby("expiry_ymd"):
        d = d.sort_values("moneyness")
        if d.empty: continue
        x = d["moneyness"].to_numpy()
        y = d["mark_iv"].to_numpy()
        if mode=="rolling":
            ser = pd.Series(y)
            y_s = ser.rolling(window=window, center=True, min_periods=2).median().to_numpy()
            mask = ~np.isfinite(y_s); y_s[mask] = y[mask]
        else:
            y_s = y
        rows.append(pd.DataFrame({"expiry_ymd": exp, "moneyness": x, "mark_iv": y_s}))
    if not rows:
        return pd.DataFrame(columns=["expiry_ymd","moneyness","mark_iv"])
    return pd.concat(rows, ignore_index=True)

def find_nearest_by_delta(df_exp, spot):
    T = days_to_years(df_exp["expiration_timestamp"].iloc[0])
    K = df_exp["strike"].values.astype(float)
    iv = df_exp["mark_iv"].values.astype(float)
    d_c = bs_delta_call(float(spot), K, T, iv)
    idx_c25 = int(np.nanargmin(np.abs(d_c-0.25)))
    idx_p25 = int(np.nanargmin(np.abs(d_c-0.75)))
    idx_atm = int(np.nanargmin(np.abs(d_c-0.50)))
    return {"iv_call25": float(iv[idx_c25]), "iv_put25": float(iv[idx_p25]), "iv_atm": float(iv[idx_atm])}

def build_rr_bf_by_expiry(df_chain, spot):
    rows = []
    for exp, df_exp in df_chain.groupby("expiry_ymd"):
        try:
            sel = find_nearest_by_delta(df_exp.sort_values("strike"), spot)
            rr = sel["iv_call25"] - sel["iv_put25"]
            bf = 0.5*(sel["iv_call25"] + sel["iv_put25"]) - sel["iv_atm"]
            rows.append({"expiry_ymd": exp, "rr25": rr, "bf25": bf})
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("expiry_ymd")

def build_oi_gex_matrix(df_chain, spot, sign_mode="calls_plus_puts_minus"):
    if df_chain.empty:
        return pd.DataFrame(columns=["expiry_ymd","strike","Call_OI","Put_OI","Call_GEX","Put_GEX","Net_GEX"])
    df = df_chain.copy()
    S = float(spot)
    T = df["expiration_timestamp"].apply(days_to_years).values
    K = df["strike"].values.astype(float)
    iv = df["mark_iv"].values.astype(float)
    gamma = bs_gamma(S, K, T, iv) * (S**2)

    is_call = df["option_type"].str.upper().str[0].values == "C"
    call_oi = np.where(is_call, df["open_interest"].values, 0.0)
    put_oi  = np.where(~is_call, df["open_interest"].values, 0.0)

    call_gex = gamma * call_oi
    put_gex  = gamma * put_oi * (-1.0 if sign_mode=="calls_plus_puts_minus" else +1.0)
    net_gex  = call_gex + put_gex

    out = pd.DataFrame({
        "expiry_ymd": df["expiry_ymd"].values,
        "strike": K,
        "Call_OI": call_oi,
        "Put_OI":  put_oi,
        "Call_GEX": call_gex,
        "Put_GEX":  put_gex,
        "Net_GEX":  net_gex,
    })
    return out

# ----------------------------
# Plots
# ----------------------------
def plot_gex_bar(df_strike, spot, header):
    d = df_strike.sort_values("strike").reset_index(drop=True)
    colors = np.where(d["net_gex"]>=0, "rgba(39,174,96,0.9)", "rgba(231,76,60,0.9)")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["strike"], y=d["net_gex"], marker_color=colors, name="Per-strike GEX"))
    if "cum_gex" in d.columns:
        fig.add_trace(go.Scatter(x=d["strike"], y=d["cum_gex"], mode="lines+markers",
                                 name="Cumulative GEX", yaxis="y2",
                                 line=dict(color="rgba(255,127,80,0.95)", width=2)))
    fig.add_vline(x=float(spot), line=dict(color="rgba(52,152,219,0.95)", width=2))
    if math.isfinite(header.get("flip", float("nan"))):
        fig.add_vline(x=header["flip"], line=dict(color="rgba(255,255,255,0.7)", width=1.5, dash="dot"))
    fig.update_layout(template="plotly_dark",
                      title=("Net GEX by Strike + Cumulative"
                             f"<br><sup>Calls OI: {header['calls_oi']:,.0f} · Puts OI: {header['puts_oi']:,.0f} · "
                             f"Σ Net GEX: {header['sum_net']:,.0f} · {_utcnow_str()}</sup>"),
                      xaxis=dict(title="Strike (USD)", tickformat=",.0f"),
                      yaxis=dict(title="GEX ($ per $1 move)"),
                      yaxis2=dict(title="Cumulative GEX", overlaying="y", side="right", showgrid=False),
                      bargap=0.08, showlegend=True, margin=dict(l=60,r=60,t=95,b=60))
    return fig

def plot_oi_profile(df_chain):
    df = df_chain.copy()
    calls = df[df["option_type"].str.upper().str[0]=="C"].groupby("strike")["open_interest"].sum().rename("Calls")
    puts  = df[df["option_type"].str.upper().str[0]=="P"].groupby("strike")["open_interest"].sum().rename("Puts")
    prof = pd.concat([calls, puts], axis=1).fillna(0.0).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=prof["strike"], y=prof["Calls"], name="Calls OI"))
    fig.add_trace(go.Bar(x=prof["strike"], y=prof["Puts"], name="Puts OI"))
    fig.update_layout(template="plotly_dark", barmode="stack", title="Open Interest by Strike",
                      xaxis=dict(title="Strike", tickformat=",.0f"),
                      yaxis=dict(title="Contracts"))
    return fig

def make_kf_style_heatmap(df_chain, spot, *, price_span=0.10, price_steps=60, iv_pad=0.10, iv_steps=40,
                          sign_mode="calls_plus_puts_minus", per_1pct=False, contract_size=1.0,
                          invert_colors=False, iv_anchor=None):
    spot = float(spot)
    p_lo,p_hi = spot*(1.0-price_span), spot*(1.0+price_span)
    P = np.linspace(p_lo, p_hi, int(price_steps))
    iv_vals = df_chain["mark_iv"].astype(float).clip(1e-4,2.0)
    if iv_anchor is not None and np.isfinite(iv_anchor) and iv_anchor>0:
        iv_med = float(iv_anchor)
    else:
        iv_med = float(np.median(iv_vals)) if len(iv_vals) else 0.5
    V = np.linspace(max(0.01, iv_med*(1.0-iv_pad)), min(2.0, iv_med*(1.0+iv_pad)), int(iv_steps))
    K = df_chain["strike"].astype(float).values[:,None,None]
    T = df_chain["expiration_timestamp"].apply(days_to_years).values[:,None,None]
    OI = df_chain["open_interest"].astype(float).values[:,None,None]
    side = df_chain["option_type"].astype(str).str.upper().str[0].values
    if sign_mode=="calls_plus_puts_minus":
        sign_vec = np.where(side=="C", 1.0, -1.0)[:,None,None]
    elif sign_mode=="both_positive":
        sign_vec = np.ones_like(OI)
    elif sign_mode=="dealer_short_all":
        sign_vec = -np.ones_like(OI)
    else:
        sign_vec = np.where(side=="C", 1.0, -1.0)[:,None,None]
    Pgrid = P[None,:,None]; Vgrid = V[None,None,:]
    gamma_grid = bs_gamma(Pgrid, K, T, Vgrid)
    scale = (spot*0.01) if per_1pct else 1.0
    gex_grid = np.nansum(OI*contract_size*sign_vec*gamma_grid*(Pgrid**2)*scale, axis=0)
    if invert_colors: gex_grid = -gex_grid
    fig = go.Figure(data=go.Heatmap(x=V, y=P, z=gex_grid, colorscale="RdBu", zmid=0,
                                    colorbar=dict(title=("GEX $/1%" if per_1pct else "GEX $/$1"))))
    fig.add_vline(x=iv_med, line=dict(color="lime", width=2, dash="dot"))
    fig.add_hline(y=spot, line=dict(color="lime", width=2, dash="dot"))
    fig.update_layout(template="plotly_dark", title=f"Dealer GEX (Price × IV) • {_utcnow_str()}",
                      xaxis=dict(title="Implied Vol", tickformat=".0%"),
                      yaxis=dict(title="Price (USD)", tickformat=",.0f"))
    return fig

def greeks_panel_by_strike(df_chain, spot, *, contract_size=1.0, sign_mode="calls_plus_puts_minus"):
    df = df_chain.copy()
    S = np.full(len(df), float(spot))
    K = df["strike"].values.astype(float)
    T = df["expiration_timestamp"].apply(days_to_years).values
    iv = df["mark_iv"].values.astype(float)
    side = df["option_type"].str.upper().str[0].values
    if sign_mode=="calls_plus_puts_minus":
        sign = np.where(side=="C", +1.0, -1.0)
    elif sign_mode=="both_positive":
        sign = np.ones(len(df), float)
    elif sign_mode=="dealer_short_all":
        sign = -np.ones(len(df), float)
    else:
        sign = np.where(side=="C", +1.0, -1.0)
    call_delta = bs_delta_call(S,K,T,iv)
    delta = np.where(side=="P", call_delta-1.0, call_delta)
    gamma = bs_gamma(S,K,T,iv)
    vega  = bs_vega(S,K,T,iv)
    theta = bs_theta_call(S,K,T,iv)
    OI = df["open_interest"].values.astype(float) * float(contract_size)
    out = pd.DataFrame({
        "strike": K,
        "Delta$": delta*S*OI*sign,
        "Gamma$/1%": gamma*(S**2)*0.01*OI*sign,
        "Vega$/vol%": vega*0.01*OI*np.sign(sign),
        "Theta$/day": (theta/365.0)*OI*np.sign(sign),
    })
    agg = out.groupby("strike").sum().reset_index().sort_values("strike")
    fig = go.Figure()
    for col in ["Delta$","Gamma$/1%","Vega$/vol%","Theta$/day"]:
        fig.add_trace(go.Scatter(x=agg["strike"], y=agg[col], mode="lines", name=col))
    fig.update_layout(template="plotly_dark", title="Greeks by Strike (net)",
                      xaxis=dict(title="Strike", tickformat=",.0f"),
                      yaxis=dict(title="Exposure (USD units)"))
    return fig

def plot_iv_smile(smile_df):
    fig = go.Figure()
    for exp,d in smile_df.groupby("expiry_ymd"):
        d = d.sort_values("moneyness")
        fig.add_trace(go.Scatter(x=d["moneyness"], y=d["mark_iv"], mode="lines", name=str(exp), connectgaps=True))
    fig.update_layout(template="plotly_dark", title="IV Smile (IV vs K/S)",
                      xaxis=dict(title="Moneyness (K/S)", tickformat=".2f"),
                      yaxis=dict(title="Implied Vol", tickformat=".0%"))
    return fig

def plot_rr_bf(rrbf_df):
    fig_rr = go.Figure(); fig_bf = go.Figure()
    if not rrbf_df.empty:
        fig_rr.add_trace(go.Bar(x=rrbf_df["expiry_ymd"], y=rrbf_df["rr25"], name="RR25 (C25−P25)"))
        fig_bf.add_trace(go.Bar(x=rrbf_df["expiry_ymd"], y=rrbf_df["bf25"], name="BF25"))
    fig_rr.update_layout(template="plotly_dark", title="Risk Reversal 25Δ by Expiry",
                         xaxis=dict(title="Expiry (YYMMDD)"), yaxis=dict(title="RR25 (vol pts)", tickformat=".2%"))
    fig_bf.update_layout(template="plotly_dark", title="Butterfly 25Δ by Expiry",
                         xaxis=dict(title="Expiry (YYMMDD)"), yaxis=dict(title="BF25 (vol pts)", tickformat=".2%"))
    return fig_rr, fig_bf

# ----------------------------
# UI
# ----------------------------
st.title("Deribit Options – Metrics (Cloud-safe)")

with st.sidebar:
    st.header("Controls")
    currency = st.selectbox("Asset", ["BTC","ETH"], index=0)
    ins_all = get_instruments(currency)
    if ins_all.empty:
        st.error("Deribit API unreachable from cloud right now. Try again or run locally. The app will render placeholders.")
        exps_all = []
    else:
        exps_all = list(dict.fromkeys(ins_all["expiry_ymd"].tolist()))
    default_exps = exps_all[:3] if exps_all else []
    expiries_sel = st.multiselect("Expiries (YYMMDD)", exps_all, default=default_exps)
    strike_win = st.slider("Strike window ±%", 5, 80, 40, help="Filter strikes around spot to speed up (±%)") / 100.0
    sign_mode = st.selectbox("GEX sign mode", ["calls_plus_puts_minus","both_positive","dealer_short_all"], index=0)
    price_span = st.slider("Heatmap Price span ±%", 5, 30, 10) / 100.0
    smile_mode = st.selectbox("Smile smoothing", ["rolling","raw"], index=0)
    threads = st.slider("Threads", 1, 12, 6)

    st.markdown("**Expiry presets**")
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        if st.button("KF View", use_container_width=True, help="Current weekly + next major"):
            if exps_all:
                sel = exps_all[:1]
                if len(exps_all) > 1:
                    sel.append(exps_all[1])
                st.session_state["preset_exps"] = sel
    with colp2:
        if st.button("Monthlies", use_container_width=True):
            st.session_state["preset_exps"] = exps_all[:3] if len(exps_all)>=3 else exps_all
    with colp3:
        if st.button("Full Curve", use_container_width=True):
            st.session_state["preset_exps"] = exps_all

    if "preset_exps" in st.session_state and exps_all:
        expiries_sel = st.session_state["preset_exps"]
        st.caption(f"Preset applied: {', '.join(expiries_sel)}")

with st.spinner("Fetching chain & building exposures..."):
    df_chain, spot = build_chain_df(currency, expiries_sel, strike_window_pct=strike_win, threads=threads)

if df_chain.empty or not np.isfinite(spot):
    st.warning("No live data available. If on Streamlit Cloud, this can be a transient network block. Try again, or run locally.")
    st.stop()

k1,k2,k3,k4 = st.columns(4)
k1.metric("Spot", f"{spot:,.0f} USD")
k2.metric("Calls OI (Σ)", f"{int(df_chain.loc[df_chain['option_type']=='C','open_interest'].sum()):,}")
k3.metric("Puts OI (Σ)", f"{int(df_chain.loc[df_chain['option_type']=='P','open_interest'].sum()):,}")
k4.metric("Expiries", f"{len(set(df_chain['expiry_ymd']))}")

gex_by_strike, hdr = build_gex_by_strike(df_chain, spot, per_1pct=False, sign_mode=sign_mode)
c1,c2 = st.columns([1.25,1.0])
with c1:
    st.plotly_chart(plot_gex_bar(gex_by_strike, spot, hdr), use_container_width=True)
with c2:
    st.plotly_chart(plot_oi_profile(df_chain), use_container_width=True)

iv_anchor = get_dvol_anchor(currency)
st.plotly_chart(make_kf_style_heatmap(df_chain, spot, price_span=price_span, iv_pad=0.10,
                                      sign_mode=sign_mode, per_1pct=False, iv_anchor=iv_anchor),
                use_container_width=True)

st.plotly_chart(greeks_panel_by_strike(df_chain, spot, sign_mode=sign_mode), use_container_width=True)

smile_df = build_iv_smile(df_chain, spot, mny_range=(0.85,1.20), mode=smile_mode, window=5)
st.plotly_chart(plot_iv_smile(smile_df), use_container_width=True)
rrbf = build_rr_bf_by_expiry(df_chain, spot)
fig_rr, fig_bf = plot_rr_bf(rrbf)
st.plotly_chart(fig_rr, use_container_width=True)
st.plotly_chart(fig_bf, use_container_width=True)

st.subheader("OI / GEX Matrix (strike × expiry)")
matrix = build_oi_gex_matrix(df_chain, spot, sign_mode=sign_mode)
exp_totals = matrix.groupby("expiry_ymd")[["Call_OI","Put_OI"]].sum().sum(axis=1)
thr = st.slider("Auto-filter: min Σ OI per expiry", 0, int(exp_totals.max() or 0)+1, 1000, step=500)
keep = exp_totals[exp_totals >= thr].index.tolist()
mat_filt = matrix[matrix["expiry_ymd"].isin(keep)] if keep else matrix

if mat_filt.empty:
    st.info("No expiries matched the OI threshold. Lower the filter or widen strike window.")
else:
    piv_oi = mat_filt.pivot_table(index="strike", columns="expiry_ymd",
                                  values=["Call_OI","Put_OI"], aggfunc="sum").fillna(0.0)
    piv_gex = mat_filt.pivot_table(index="strike", columns="expiry_ymd",
                                   values="Net_GEX", aggfunc="sum").fillna(0.0)
    st.dataframe(piv_oi.round(0), use_container_width=True, height=300)
    st.dataframe(piv_gex.round(0), use_container_width=True, height=300)

st.caption(f"Updated: {_utcnow_str()} • Expiries: {', '.join(sorted(set(df_chain['expiry_ymd'])))} • Instruments: {len(df_chain):,}")
