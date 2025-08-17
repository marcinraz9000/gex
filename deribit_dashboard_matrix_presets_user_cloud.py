# deribit_dashboard_matrix_presets_user.py
# Streamlit dashboard for Deribit options (BTC/ETH)
# - Net GEX by strike (USD per $1 move) + cumulative line
# - Dealer GEX heatmap (Price × IV), IV anchored to DVOL by default
# - Greeks by strike (Delta$, Gamma$/1%, Vega$/vol pt, Theta$/day)
# - IV Smiles (RR25/BF25)
# - Expiry presets: KF / Monthlies / Full + User-defined save/load/delete
# - Strike × Expiry Matrix for OI & GEX with auto-filter & monthly/quarterly grouping
#
# Run:
#   python -m streamlit run deribit_dashboard_matrix_presets_user.py

import time
import math
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st

DERIBIT = "https://www.deribit.com/api/v2"
st.set_page_config(page_title="BTC/ETH – Options Metrics", layout="wide")

# Where user presets are stored (local-friendly: next to script; cloud variant overrides this)
PRESETS_PATH = (Path.home() / ".streamlit" / "options_presets.json")
# ensure folder exists
PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _utcnow_str() -> str:
    return f"{datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC"


def _st_rerun():
    try:
        st.rerun()
    except Exception:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()


def load_presets() -> dict:
    try:
        data = json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_presets(data: dict) -> None:
    try:
        PRESETS_PATH.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass


# ----------------------------
# Data fetchers
# ----------------------------
@st.cache_data(ttl=60 * 15, show_spinner=False)
def get_instruments(currency: str) -> pd.DataFrame:
    url = f"{DERIBIT}/public/get_instruments"
    params = {"currency": currency, "kind": "option", "expired": False}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        items = r.json().get("result", []) or []
    except Exception:
        r = requests.get(url, params={"currency": currency, "kind": "option"}, timeout=15)
        r.raise_for_status()
        items = r.json().get("result", []) or []
    rows = []
    for x in items:
        ts_s = float(x.get("expiration_timestamp", 0) or 0) / 1000.0
        rows.append(
            {
                "instrument_name": x.get("instrument_name"),
                "strike": float(x.get("strike", 0) or 0),
                "option_type": ("C" if str(x.get("option_type", "")).lower().startswith("call") else "P"),
                "expiration_timestamp": ts_s,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["expiry_ymd"] = pd.to_datetime(df["expiration_timestamp"], unit="s", utc=True).dt.strftime("%y%m%d")
    return df.sort_values(["expiration_timestamp", "strike"]).reset_index(drop=True)


@st.cache_data(ttl=20, show_spinner=False)
def get_book_summary(currency: str) -> pd.DataFrame:
    params = {"currency": currency, "kind": "option"}
    r = requests.get(f"{DERIBIT}/public/get_book_summary_by_currency", params=params, timeout=20)
    r.raise_for_status()
    items = r.json().get("result", []) or []
    rows = []
    for x in items:
        rows.append(
            {
                "instrument_name": x.get("instrument_name"),
                "open_interest": float(x.get("open_interest", 0) or 0),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=20, show_spinner=False)
def get_index_price(currency: str) -> float:
    index = "btc_usd" if currency.upper() == "BTC" else "eth_usd"
    r = requests.get(f"{DERIBIT}/public/get_index_price", params={"index_name": index}, timeout=10)
    r.raise_for_status()
    return float(r.json()["result"]["index_price"])


@st.cache_data(ttl=20, show_spinner=False)
def get_mark_iv_batch(instrument_names, threads: int = 12) -> pd.DataFrame:
    names = list(dict.fromkeys(instrument_names))

    def _one(nm):
        try:
            r = requests.get(f"{DERIBIT}/public/ticker", params={"instrument_name": nm}, timeout=10)
            r.raise_for_status()
            res = r.json().get("result", {}) or {}
            return {"instrument_name": nm, "mark_iv": float(res.get("mark_iv", 0) or 0)}
        except Exception:
            return {"instrument_name": nm, "mark_iv": 0.0}

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
        futures = {ex.submit(_one, nm): nm for nm in names}
        for fut in as_completed(futures):
            rows.append(fut.result())
    return pd.DataFrame(rows)


@st.cache_data(ttl=60, show_spinner=False)
def get_dvol_anchor(currency: str) -> float:
    """Fetch DVOL-like index (decimal). If fails, returns NaN."""
    try:
        url = f"{DERIBIT}/public/get_volatility_index_data"
        params = {"currency": currency, "start_timestamp": int((time.time() - 2 * 86400) * 1000)}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        res = r.json().get("result")
        if isinstance(res, list) and len(res) > 0:
            last = res[-1]
            if isinstance(last, dict):
                v = float(last.get("volatility", last.get("value", np.nan)))
            elif isinstance(last, (list, tuple)) and len(last) >= 2:
                v = float(last[1])
            else:
                v = float(last)
            return v / 100.0 if v > 3 else v
    except Exception:
        pass
    return float("nan")


# ----------------------------
# Black–Scholes helpers (vectorized)
# ----------------------------
def days_to_years(exp_ts_s: float) -> float:
    now = time.time()
    T = max(float(exp_ts_s) - now, 0.0) / (365.0 * 24.0 * 3600.0)
    return T if T > 1e-6 else 1e-6


def n_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def n_cdf(x):
    """Standard normal CDF (vectorized, Abramowitz–Stegun 7.1.26)."""
    x = np.asarray(x, dtype=float)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(x))
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    m = 1.0 - n_pdf(x) * poly
    return np.where(x >= 0.0, m, 1.0 - m)


def d1_d2(S, K, T, sigma):
    S = np.asarray(S, float)
    K = np.asarray(K, float)
    T = np.maximum(np.asarray(T, float), 1e-6)
    sigma = np.maximum(np.asarray(sigma, float), 1e-6)
    d1 = (np.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_gamma(S, K, T, sigma):
    d1, _ = d1_d2(S, K, T, sigma)
    return n_pdf(d1) / (S * sigma * np.sqrt(T))


def bs_delta_call(S, K, T, sigma):
    d1, _ = d1_d2(S, K, T, sigma)
    return n_cdf(d1)


def bs_vega(S, K, T, sigma):
    d1, _ = d1_d2(S, K, T, sigma)
    return S * n_pdf(d1) * np.sqrt(T)  # per 1.0 vol (multiply by 0.01 for vol point)


def bs_theta_call(S, K, T, sigma, r=0.0, q=0.0):
    d1, d2 = d1_d2(S, K, T, sigma)
    term1 = - (S * np.exp(-q * T) * n_pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = - r * K * np.exp(-r * T) * (1.0 - n_cdf(d2))
    term3 = + q * S * np.exp(-q * T) * n_cdf(d1)
    return term1 + term2 + term3  # per year


# ----------------------------
# Builders & plots
# ----------------------------
def build_chain_df(currency: str, expiries_sel, strike_window_pct: float, threads: int = 12):
    ins = get_instruments(currency)
    if expiries_sel:
        ins = ins[ins["expiry_ymd"].isin(expiries_sel)].copy()
    if ins.empty:
        return pd.DataFrame(), np.nan
    spot = get_index_price(currency)
    if strike_window_pct and strike_window_pct > 0:
        lo, hi = spot * (1.0 - strike_window_pct), spot * (1.0 + strike_window_pct)
        ins = ins[(ins["strike"] >= lo) & (ins["strike"] <= hi)].copy()
    bs = get_book_summary(currency)
    df = ins.merge(bs, on="instrument_name", how="left")
    df["open_interest"] = df["open_interest"].fillna(0.0)
    ivs = get_mark_iv_batch(df["instrument_name"].tolist(), threads=threads)
    df = df.merge(ivs, on="instrument_name", how="left")
    df["mark_iv"] = df["mark_iv"].replace([None, np.nan], 0.0).clip(1e-4, 500.0)
    df["mark_iv"] = np.where(df["mark_iv"] > 3.0, df["mark_iv"] / 100.0, df["mark_iv"])
    return df, float(spot)


def build_iv_smile(df_chain, spot, *, mny_range=(0.85, 1.20), mode="rolling", window=5, min_iv=0.01):
    if df_chain.empty:
        return pd.DataFrame(columns=["expiry_ymd", "moneyness", "mark_iv"])
    df = df_chain.copy()
    df = df[(df["mark_iv"] > min_iv) & (df["mark_iv"] < 2.0)].copy()
    df["moneyness"] = df["strike"].astype(float) / float(spot)
    lo, hi = float(mny_range[0]), float(mny_range[1])
    df = df[(df["moneyness"] >= lo) & (df["moneyness"] <= hi)]
    rows = []
    for exp, d in df.groupby("expiry_ymd"):
        d = d.sort_values("moneyness")
        if d.empty:
            continue
        x = d["moneyness"].to_numpy()
        y = d["mark_iv"].to_numpy()
        if mode == "rolling":
            ser = pd.Series(y)
            y_s = ser.rolling(window=window, center=True, min_periods=2).median().to_numpy()
            mask = ~np.isfinite(y_s)
            y_s[mask] = y[mask]
        else:
            y_s = y
        rows.append(pd.DataFrame({"expiry_ymd": exp, "moneyness": x, "mark_iv": y_s}))
    if not rows:
        return pd.DataFrame(columns=["expiry_ymd", "moneyness", "mark_iv"])
    return pd.concat(rows, ignore_index=True)


def find_nearest_by_delta(df_exp, spot):
    T = days_to_years(df_exp["expiration_timestamp"].iloc[0])
    K = df_exp["strike"].values.astype(float)
    iv = df_exp["mark_iv"].values.astype(float)
    d_c = bs_delta_call(float(spot), K, T, iv)
    idx_c25 = int(np.nanargmin(np.abs(d_c - 0.25)))
    idx_p25 = int(np.nanargmin(np.abs(d_c - 0.75)))
    idx_atm = int(np.nanargmin(np.abs(d_c - 0.50)))
    return {"iv_call25": float(iv[idx_c25]), "iv_put25": float(iv[idx_p25]), "iv_atm": float(iv[idx_atm])}


def build_rr_bf_by_expiry(df_chain, spot):
    rows = []
    for exp, df_exp in df_chain.groupby("expiry_ymd"):
        try:
            sel = find_nearest_by_delta(df_exp.sort_values("strike"), spot)
            rr = sel["iv_call25"] - sel["iv_put25"]
            bf = 0.5 * (sel["iv_call25"] + sel["iv_put25"]) - sel["iv_atm"]
            rows.append({"expiry_ymd": exp, "rr25": rr, "bf25": bf})
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("expiry_ymd")


def build_gex_by_strike(
    df_chain, spot, *, contract_size=1.0, per_1pct=False, sign_mode="calls_plus_puts_minus"
):
    df = df_chain.copy()
    side = df["option_type"].str.upper().str[0].values
    if sign_mode == "calls_plus_puts_minus":
        sign = np.where(side == "C", +1.0, -1.0)
    elif sign_mode == "both_positive":
        sign = np.ones(len(df), float)
    elif sign_mode == "dealer_short_all":
        sign = -np.ones(len(df), float)
    else:
        sign = np.where(side == "C", +1.0, -1.0)
    T = df["expiration_timestamp"].apply(days_to_years).values
    gamma = bs_gamma(float(spot), df["strike"].values.astype(float), T, df["mark_iv"].values.astype(float))
    scale = (float(spot) * 0.01) if per_1pct else 1.0
    contrib = gamma * (float(spot) ** 2) * scale * df["open_interest"].values.astype(float) * float(contract_size) * sign
    df["_gex"] = contrib
    agg = (
        df.groupby("strike")["_gex"].sum().rename("net_gex").to_frame().reset_index().sort_values("strike")
    )
    if not agg.empty:
        agg["cum_gex"] = agg["net_gex"].cumsum()
    flip = float(agg.loc[agg["net_gex"].abs().idxmin(), "strike"]) if not agg.empty else float("nan")
    hdr = {
        "calls_oi": float(df.loc[side == "C", "open_interest"].sum()),
        "puts_oi": float(df.loc[side == "P", "open_interest"].sum()),
        "flip": flip,
        "sum_net": float(agg["net_gex"].sum()),
    }
    return agg, hdr


def plot_gex_bar(df_strike, spot, header):
    d = df_strike.sort_values("strike").reset_index(drop=True)
    colors = np.where(d["net_gex"] >= 0, "rgba(39,174,96,0.9)", "rgba(231,76,60,0.9)")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["strike"], y=d["net_gex"], marker_color=colors, name="Per-strike GEX"))
    if "cum_gex" in d.columns:
        fig.add_trace(go.Scatter(x=d["strike"], y=d["cum_gex"], mode="lines+markers",
                                 name="Cumulative GEX", line=dict(color="rgba(255,127,80,0.95)", width=2), yaxis="y2"))
    fig.add_vline(x=float(spot), line=dict(color="rgba(52,152,219,0.95)", width=2))
    if math.isfinite(header.get("flip", float("nan"))):
        fig.add_vline(x=header["flip"], line=dict(color="rgba(255,255,255,0.7)", width=1.5, dash="dot"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="black", plot_bgcolor="black",
                      title=("Net GEX by Strike + Cumulative (Σ selected)"
                             f"<br><sup>Calls OI: {header['calls_oi']:,.0f} · Puts OI: {header['puts_oi']:,.0f} · "
                             f"Σ Net GEX: {header['sum_net']:,.0f} · {_utcnow_str()}</sup>"),
                      xaxis=dict(title="Strike (USD)", tickformat=",.0f"),
                      yaxis=dict(title="GEX ($ per $1 move)"),
                      yaxis2=dict(title="Cumulative GEX", overlaying="y", side="right", showgrid=False),
                      bargap=0.08, showlegend=True, margin=dict(l=60, r=60, t=95, b=60))
    return fig


def plot_oi_profile(df_chain):
    df = df_chain.copy()
    calls = df[df["option_type"].str.upper().str[0] == "C"].groupby("strike")["open_interest"].sum().rename("Calls")
    puts = df[df["option_type"].str.upper().str[0] == "P"].groupby("strike")["open_interest"].sum().rename("Puts")
    prof = pd.concat([calls, puts], axis=1).fillna(0.0).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=prof["strike"], y=prof["Calls"], name="Calls OI"))
    fig.add_trace(go.Bar(x=prof["strike"], y=prof["Puts"], name="Puts OI"))
    fig.update_layout(template="plotly_dark", barmode="stack", title="Open Interest by Strike",
                      xaxis=dict(title="Strike", tickformat=",.0f"), yaxis=dict(title="Contracts"))
    return fig


def make_kf_style_heatmap(
    df_chain, spot, *, price_span=0.10, price_steps=60, iv_pad=0.10, iv_steps=40,
    sign_mode="calls_plus_puts_minus", per_1pct=False, contract_size=1.0, invert_colors=False, iv_anchor=None
):
    spot = float(spot)
    p_lo, p_hi = spot * (1.0 - price_span), spot * (1.0 + price_span)
    P = np.linspace(p_lo, p_hi, int(price_steps))
    iv_vals = df_chain["mark_iv"].astype(float).clip(1e-4, 2.0)
    if iv_anchor is not None and np.isfinite(iv_anchor) and iv_anchor > 0:
        iv_med = float(iv_anchor)
    else:
        iv_med = float(np.median(iv_vals)) if len(iv_vals) else 0.5
    V = np.linspace(max(0.01, iv_med*(1.0 - iv_pad)), min(2.0, iv_med*(1.0 + iv_pad)), int(iv_steps))

    K = df_chain["strike"].astype(float).values[:, None, None]
    T = df_chain["expiration_timestamp"].apply(days_to_years).values[:, None, None]
    OI = df_chain["open_interest"].astype(float).values[:, None, None]
    side = df_chain["option_type"].astype(str).str.upper().str[0].values
    if sign_mode == "calls_plus_puts_minus":
        sign_vec = np.where(side == "C", 1.0, -1.0)[:, None, None]
    elif sign_mode == "both_positive":
        sign_vec = np.ones_like(OI)
    elif sign_mode == "dealer_short_all":
        sign_vec = -np.ones_like(OI)
    else:
        sign_vec = np.where(side == "C", 1.0, -1.0)[:, None, None]

    Pgrid = P[None, :, None]
    Vgrid = V[None, None, :]
    gamma_grid = bs_gamma(Pgrid, K, T, Vgrid)
    scale = (spot * 0.01) if per_1pct else 1.0
    gex_grid = np.nansum(OI * contract_size * sign_vec * gamma_grid * (Pgrid**2) * scale, axis=0)
    if invert_colors:
        gex_grid = -gex_grid

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
    if sign_mode == "calls_plus_puts_minus":
        sign = np.where(side == "C", +1.0, -1.0)
    elif sign_mode == "both_positive":
        sign = np.ones(len(df), float)
    elif sign_mode == "dealer_short_all":
        sign = -np.ones(len(df), float)
    else:
        sign = np.where(side == "C", +1.0, -1.0)

    call_delta = bs_delta_call(S, K, T, iv)
    delta = np.where(side == "P", call_delta - 1.0, call_delta)
    gamma = bs_gamma(S, K, T, iv)
    vega = bs_vega(S, K, T, iv)
    theta = bs_theta_call(S, K, T, iv)
    OI = df["open_interest"].values.astype(float) * float(contract_size)

    delta_usd = delta * S * OI * sign
    gamma_usd_per1pct = gamma * (S**2) * 0.01 * OI * sign
    vega_usd_per_volpt = vega * 0.01 * OI * np.sign(sign)
    theta_usd_per_day = (theta / 365.0) * OI * np.sign(sign)

    out = pd.DataFrame({"strike": K, "Delta$": delta_usd, "Gamma$/1%": gamma_usd_per1pct,
                        "Vega$/vol%": vega_usd_per_volpt, "Theta$/day": theta_usd_per_day})
    agg = out.groupby("strike").sum().reset_index().sort_values("strike")
    fig = go.Figure()
    for col in ["Delta$", "Gamma$/1%", "Vega$/vol%", "Theta$/day"]:
        fig.add_trace(go.Scatter(x=agg["strike"], y=agg[col], mode="lines", name=col))
    fig.update_layout(template="plotly_dark", title="Greeks by Strike (net)",
                      xaxis=dict(title="Strike", tickformat=",.0f"), yaxis=dict(title="Exposure (USD units)"))
    return fig


# ----------------------------
# NEW: Strike × Expiry matrix
# ----------------------------
def _period_label(ts: pd.Timestamp, mode: str) -> str:
    if mode == "Monthly":
        return ts.strftime("%y%m")
    if mode == "Quarterly":
        q = (ts.month - 1) // 3 + 1
        return f"{ts.strftime('%y')}Q{q}"
    return ts.strftime("%y%m%d")


def build_oi_gex_matrix(df_chain: pd.DataFrame, spot: float, sign_mode: str = "calls_plus_puts_minus",
                        group_mode: str = "None"):
    if df_chain.empty:
        return pd.DataFrame(columns=["expiry_key","strike","call_oi","put_oi","call_gex","put_gex","net_gex"]), \
               pd.DataFrame(columns=["expiry_key","sum_oi","share_oi","flip_near_spot"])
    df = df_chain.copy()
    T = df["expiration_timestamp"].apply(days_to_years).values
    S = float(spot)
    gamma = bs_gamma(S, df["strike"].values.astype(float), T, df["mark_iv"].values.astype(float))
    raw_gex = gamma * (S**2)
    side = df["option_type"].str.upper().str[0].values
    call_mask, put_mask = (side == "C"), (side != "C")
    exp_ts = pd.to_datetime(df["expiration_timestamp"], unit="s", utc=True)
    expiry_key = [ _period_label(ts, group_mode) for ts in exp_ts ] if group_mode in ("Monthly","Quarterly") else df["expiry_ymd"].tolist()
    df["_expiry_key"] = expiry_key
    base = pd.DataFrame({
        "expiry_key": df["_expiry_key"].values,
        "strike": df["strike"].values.astype(float),
        "call_oi": np.where(call_mask, df["open_interest"].values.astype(float), 0.0),
        "put_oi": np.where(put_mask, df["open_interest"].values.astype(float), 0.0),
        "call_gex": np.where(call_mask, raw_gex * df["open_interest"].values.astype(float), 0.0),
        "put_gex": np.where(put_mask, raw_gex * df["open_interest"].values.astype(float), 0.0),
    })
    agg = base.groupby(["expiry_key","strike"]).sum(numeric_only=True).reset_index()
    if sign_mode == "calls_plus_puts_minus":
        agg["net_gex"] = agg["call_gex"] - agg["put_gex"]
    elif sign_mode == "both_positive":
        agg["net_gex"] = agg["call_gex"] + agg["put_gex"]
    elif sign_mode == "dealer_short_all":
        agg["net_gex"] = -(agg["call_gex"] + agg["put_gex"])
    else:
        agg["net_gex"] = agg["call_gex"] - agg["put_gex"]
    exp_sum = agg.groupby("expiry_key").agg(sum_call=('call_oi','sum'), sum_put=('put_oi','sum')).reset_index()
    exp_sum["sum_oi"] = exp_sum["sum_call"] + exp_sum["sum_put"]
    total_oi = float(exp_sum["sum_oi"].sum()) if len(exp_sum) else 1.0
    exp_sum["share_oi"] = np.where(total_oi > 0, exp_sum["sum_oi"]/total_oi, 0.0)
    flips = []
    for exp, sub in agg.groupby("expiry_key"):
        below = sub.loc[sub["strike"] < S, "net_gex"]
        above = sub.loc[sub["strike"] > S, "net_gex"]
        has_flip = (below.size > 0 and above.size > 0 and np.sign(below.mean()) * np.sign(above.mean()) < 0)
        flips.append({"expiry_key": exp, "flip_near_spot": bool(has_flip)})
    flip_df = pd.DataFrame(flips)
    exp_sum = exp_sum.merge(flip_df, on="expiry_key", how="left").fillna({"flip_near_spot": False})
    exp_sum = exp_sum[["expiry_key","sum_oi","share_oi","flip_near_spot"]].sort_values("expiry_key")
    return agg, exp_sum


def pivot_heatmap(df: pd.DataFrame, value_col: str, title: str):
    if df.empty:
        return go.Figure().update_layout(template="plotly_dark", title=title)
    piv = df.pivot(index="expiry_key", columns="strike", values=value_col)
    piv = piv.sort_index().sort_index(axis=1)
    fig = go.Figure(
        data=go.Heatmap(
            z=piv.values,
            x=piv.columns.astype(float),
            y=piv.index.astype(str),
            colorscale="Viridis",
            colorbar=dict(title=value_col),
        )
    )
    fig.update_layout(template="plotly_dark", title=title,
                      xaxis=dict(title="Strike", tickformat=",.0f"),
                      yaxis=dict(title="Expiry / Group"))
    return fig


# ----------------------------
# UI
# ----------------------------
st.title("Deribit Options – Metrics")

with st.sidebar:
    st.header("Controls")
    currency = st.selectbox("Asset", ["BTC", "ETH"], index=0)

    ins_all = get_instruments(currency)
    exps_all = sorted(list(dict.fromkeys(ins_all["expiry_ymd"].tolist())))
    now_utc = pd.Timestamp.now(tz="UTC")

    # Preset buttons
    st.subheader("Expiry presets")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] div.stButton > button {
            padding: .25rem .6rem;
            font-size: 0.85rem;
            line-height: 1.1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    exps_dt = [(e, pd.to_datetime("20" + str(e), format="%Y%m%d", utc=True)) for e in exps_all]
    upcoming = [(e, t) for (e, t) in exps_dt if t >= now_utc - pd.Timedelta(hours=1)]
    cur_e = upcoming[0][0] if upcoming else None
    cur_t = upcoming[0][1] if upcoming else None

    month_buckets = {}
    for e, t in exps_dt:
        month_buckets.setdefault((t.year, t.month), []).append((e, t))
    monthlies = []
    for _, lst in month_buckets.items():
        lst.sort(key=lambda x: x[1])
        if lst[-1][1] >= now_utc - pd.Timedelta(hours=1):
            monthlies.append(lst[-1])
    monthlies.sort(key=lambda x: x[1])

    cA, cB, cC = st.columns(3)
    if cA.button("KF"):
        sel = []
        if cur_e is not None:
            sel.append(cur_e)
            nxt_major = None
            for e, t in monthlies:
                if t > cur_t:
                    nxt_major = e
                    break
            if not nxt_major and monthlies:
                for e, _ in monthlies:
                    if e != cur_e:
                        nxt_major = e
                        break
            if nxt_major:
                sel.append(nxt_major)
        st.session_state["expiries_sel"] = sel
        _st_rerun()

    if cB.button("Monthlies"):
        st.session_state["expiries_sel"] = [e for (e, _) in monthlies[:3]]
        _st_rerun()

    if cC.button("Full"):
        horizon = now_utc + pd.Timedelta(days=180)
        sel = [e for (e, t) in exps_dt if (now_utc - pd.Timedelta(hours=1)) <= t <= horizon]
        st.session_state["expiries_sel"] = sel
        _st_rerun()

    # User-defined presets
    if "user_presets" not in st.session_state:
        st.session_state["user_presets"] = load_presets()
    st.subheader("My presets")
    presets: dict = st.session_state["user_presets"]
    preset_names = sorted(presets.keys())
    pc1, pc2 = st.columns([2, 1])
    sel_preset = pc1.selectbox("Preset", preset_names, index=0 if preset_names else None,
                               placeholder="No saved presets yet", key="sel_preset_name")
    if pc2.button("Load", disabled=not preset_names):
        st.session_state["expiries_sel"] = list(presets[sel_preset])
        _st_rerun()
    with st.expander("Save current as new preset"):
        new_name = st.text_input("Preset name", placeholder="e.g. BTC_KF_view")
        if st.button("Save preset"):
            name = new_name.strip()
            if not name:
                st.warning("Give your preset a name.")
            else:
                presets[name] = list(st.session_state.get("expiries_sel", []))
                st.session_state["user_presets"] = presets
                save_presets(presets)
                st.success(f"Saved preset “{name}”")
                _st_rerun()
    po1, po2 = st.columns(2)
    if po1.button("Overwrite", disabled=not preset_names):
        presets[sel_preset] = list(st.session_state.get("expiries_sel", []))
        st.session_state["user_presets"] = presets
        save_presets(presets)
        st.success(f"Overwrote “{sel_preset}”")
    if po2.button("Delete", disabled=not preset_names, type="secondary"):
        presets.pop(sel_preset, None)
        st.session_state["user_presets"] = presets
        save_presets(presets)
        st.success(f"Deleted “{sel_preset}”")
        _st_rerun()
    st.caption(f"Presets file: {PRESETS_PATH}")

    # Multiselect (uses same key)
    default_exps = st.session_state.get("expiries_sel", [e for (e, _) in exps_dt[:3]])
    expiries_sel = st.multiselect("Expiries (YYMMDD)", exps_all, default=default_exps, key="expiries_sel")

    # Remaining controls
    strike_win = st.slider("Strike window ±%", 5, 80, 40, help="Filter strikes around spot to speed up (±%)") / 100.0
    sign_mode = st.selectbox("GEX sign mode", ["calls_plus_puts_minus", "both_positive", "dealer_short_all"], index=0)
    price_span = st.slider("Heatmap Price span ±%", 5, 30, 10) / 100.0
    smile_mode = st.selectbox("Smile smoothing", ["rolling", "raw"], index=0)
    threads = st.slider("Threads", 1, 24, 12)

with st.spinner("Fetching chain & building exposures..."):
    df_chain, spot = build_chain_df(currency, expiries_sel, strike_window_pct=strike_win, threads=threads)

if df_chain.empty or not np.isfinite(spot):
    st.error("No data returned. Try different expiries or widen strike window.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Spot", f"{spot:,.0f} USD")
k2.metric("Calls OI (Σ)", f"{int(df_chain.loc[df_chain['option_type']=='C','open_interest'].sum()):,}")
k3.metric("Puts OI (Σ)", f"{int(df_chain.loc[df_chain['option_type']=='P','open_interest'].sum()):,}")
k4.metric("Expiries", f"{len(set(df_chain['expiry_ymd']))}")

# GEX by strike (per $1 move) + cumulative line
gex_by_strike, hdr = build_gex_by_strike(df_chain, spot, per_1pct=False, sign_mode=sign_mode)

c1, c2 = st.columns([1.25, 1.0])
with c1:
    st.plotly_chart(plot_gex_bar(gex_by_strike, spot, hdr), use_container_width=True)
with c2:
    st.plotly_chart(plot_oi_profile(df_chain), use_container_width=True)

# Heatmap (per $1 move), IV anchored to DVOL by default
iv_anchor = get_dvol_anchor(currency)
st.plotly_chart(
    make_kf_style_heatmap(
        df_chain, spot, price_span=price_span, iv_pad=0.10, sign_mode=sign_mode, per_1pct=False, iv_anchor=iv_anchor
    ),
    use_container_width=True,
)

# Greeks panel
st.plotly_chart(greeks_panel_by_strike(df_chain, spot, sign_mode=sign_mode), use_container_width=True)

# Skews
smile_df = build_iv_smile(df_chain, spot, mny_range=(0.85, 1.20), mode=smile_mode, window=5)
st.plotly_chart(plot_iv_smile(smile_df), use_container_width=True)
rrbf = build_rr_bf_by_expiry(df_chain, spot)
fig_rr, fig_bf = plot_rr_bf(rrbf)
st.plotly_chart(fig_rr, use_container_width=True)
st.plotly_chart(fig_bf, use_container_width=True)

# ----------------------------
# NEW SECTION: Strike × Expiry Matrix
# ----------------------------
st.header("Strike × Expiry Matrix (OI & GEX)")

col_m1, col_m2, col_m3, col_m4 = st.columns([1.2, 1.2, 1, 1])
group_mode = col_m1.selectbox("Group by", ["None", "Monthly", "Quarterly"], index=0, help="Aggregate expiries")
auto_filter = col_m2.checkbox("Auto-filter expiries", value=True)
min_oi = col_m3.number_input("Min Σ OI", min_value=0, value=10000, step=1000)
min_share = col_m4.slider("Min % of total OI", 0, 20, 5, help="Include expiries above this share")

# Build matrix
matrix_df, exp_summary = build_oi_gex_matrix(df_chain, spot, sign_mode=sign_mode, group_mode=group_mode)

include_keys = set(exp_summary["expiry_key"].tolist())
if auto_filter and not exp_summary.empty:
    mask = (exp_summary["sum_oi"] >= float(min_oi)) | (exp_summary["share_oi"] >= (float(min_share)/100.0)) | (exp_summary["flip_near_spot"])
    include_keys = set(exp_summary.loc[mask, "expiry_key"].tolist())

st.dataframe(
    exp_summary.assign(
        include=exp_summary["expiry_key"].isin(include_keys),
        share_pct=(exp_summary["share_oi"]*100.0).round(2)
    )[["expiry_key","sum_oi","share_pct","flip_near_spot","include"]].sort_values(["include","sum_oi"], ascending=[False,False]),
    use_container_width=True
)

apply_btn = st.button("Apply included expiries to selection", disabled=(group_mode!="None"))
if apply_btn and group_mode=="None":
    st.session_state["expiries_sel"] = sorted(list(include_keys))
    st.success(f"Applied {len(include_keys)} expiries to selection")
    _st_rerun()

if matrix_df.empty:
    st.info("No matrix data.")
else:
    filtered_df = matrix_df[matrix_df["expiry_key"].isin(include_keys)].copy()
    hm1 = pivot_heatmap(filtered_df.assign(total_oi=filtered_df["call_oi"]+filtered_df["put_oi"]), "total_oi",
                        "OI Heatmap (Calls+Puts)")
    hm2 = pivot_heatmap(filtered_df, "net_gex", "Net Dealer GEX Heatmap")
    st.plotly_chart(hm1, use_container_width=True)
    st.plotly_chart(hm2, use_container_width=True)
    with st.expander("Matrix data (sample)"):
        show_rows = min(2000, len(filtered_df))
        st.dataframe(filtered_df.sort_values(["expiry_key","strike"]).head(show_rows), use_container_width=True)

st.caption(f"Updated: {_utcnow_str()} • Matrix built on current selection • Grouping: {group_mode}")
