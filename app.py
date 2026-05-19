import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

from backend.pricing_engine import BlackScholes

# ── PAGE CONFIG — must be the very first Streamlit call ───────────────────────
st.set_page_config(
    page_title="Options Pricing Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── HELPERS ───────────────────────────────────────────────────────────────────

TICKERS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "INFOSYS": "INFY.NS",
}


@st.cache_data(ttl=300)
def get_live_price(ticker: str):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        return float(data["Close"].iloc[-1]) if not data.empty else None
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_hist_vol(ticker: str, days: int = 30):
    try:
        data = yf.Ticker(ticker).history(period=f"{days + 10}d")
        if len(data) > days:
            log_ret = np.log(data["Close"] / data["Close"].shift(1))
            return float(log_ret.std() * np.sqrt(252))
        return None
    except Exception:
        return None


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Options Calculator")
    st.caption("Black-Scholes · European Options")
    st.divider()

    use_live = st.checkbox("📡 Use Live Market Data")
    default_spot = 21500.0

    if use_live:
        ticker_name = st.selectbox("Ticker", list(TICKERS.keys()))
        ticker = TICKERS[ticker_name]
        live_price = get_live_price(ticker)
        if live_price:
            st.success(f"Live: ₹{live_price:,.2f}")
            default_spot = live_price
            hv = get_hist_vol(ticker, 30)
            if hv:
                st.info(f"30-day HV: {hv * 100:.1f}%")
        else:
            st.warning("Could not fetch — using default")

    st.divider()
    st.subheader("Parameters")

    S = st.number_input("Spot Price (S)", min_value=1.0, value=float(round(default_spot)), step=100.0)
    K = st.number_input("Strike Price (K)", min_value=1.0, value=float(round(default_spot)), step=100.0)

    today = datetime.today().date()
    expiry = st.date_input(
        "Expiry Date",
        value=today + timedelta(days=30),
        min_value=today + timedelta(days=1),
    )
    days_left = max((expiry - today).days, 1)
    T = days_left / 365
    st.caption(f"⏳ {days_left} calendar days to expiry")

    r = st.slider("Risk-Free Rate (%)", 0.0, 15.0, 7.0, 0.5) / 100
    sigma = st.slider("Volatility (%)", 1.0, 100.0, 18.0, 0.5) / 100

    st.divider()
    st.subheader("P/L Analysis")
    st.caption("Enter your purchase prices to switch the heatmap to P/L mode")
    call_purchase = st.number_input("Call Purchase Price (₹)", min_value=0.0, value=0.0, step=5.0)
    put_purchase = st.number_input("Put Purchase Price (₹)", min_value=0.0, value=0.0, step=5.0)

    # Author links
    st.divider()
    st.markdown(
        """
        <div style="text-align:center; padding:6px 0 2px;">
            <a href="https://www.linkedin.com/in/avishkar-lokhande-9b68b024a/"
               target="_blank" style="text-decoration:none; margin-right:14px;">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png"
                     width="26" style="vertical-align:middle; margin-right:4px;"/>
                <span style="font-size:13px; color:#0A66C2; vertical-align:middle;">LinkedIn</span>
            </a>
            <a href="https://github.com/Avishkar-Lokhande"
               target="_blank" style="text-decoration:none;">
                <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png"
                     width="26" style="vertical-align:middle; margin-right:4px;"/>
                <span style="font-size:13px; color:#24292e; vertical-align:middle;">GitHub</span>
            </a>
        </div>
        <p style="text-align:center; font-size:11px; color:#888; margin-top:6px;">
            Built by Avishkar Lokhande
        </p>
        """,
        unsafe_allow_html=True,
    )

# ── COMPUTE ───────────────────────────────────────────────────────────────────

try:
    bs = BlackScholes(S, K, T, r, sigma)
except ValueError as e:
    st.error(f"Input error: {e}")
    st.stop()

call_price = bs.call_price()
put_price = bs.put_price()

# ── PAGE HEADER ───────────────────────────────────────────────────────────────

st.title("📊 Options Pricing Calculator")
st.caption("European options · Black-Scholes model · Prices in ₹")

moneyness_pct = (S - K) / K * 100
if abs(moneyness_pct) < 2:
    st.info("🟡 **At The Money (ATM)** — Spot ≈ Strike")
elif S > K:
    st.success(f"🟢 **In The Money (ITM)** — Call favourable · {moneyness_pct:+.1f}%")
else:
    st.warning(f"🔴 **Out of The Money (OTM)** — {moneyness_pct:.1f}% from strike")

# ── TABS ──────────────────────────────────────────────────────────────────────

tab_calc, tab_heat, tab_strat = st.tabs(["📐 Calculator", "🔥 Heatmap", "📊 Strategies"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — CALCULATOR
# ═════════════════════════════════════════════════════════════════════════════

with tab_calc:

    # Prices & break-even
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Call Price", f"₹{call_price:.2f}")
    c2.metric("Put Price", f"₹{put_price:.2f}")
    c3.metric("Call Break-even", f"₹{K + call_price:,.2f}")
    c4.metric("Put Break-even", f"₹{K - put_price:,.2f}")

    # P/L row — only when purchase prices are set
    if call_purchase > 0 or put_purchase > 0:
        st.divider()
        p1, p2, p3, p4 = st.columns(4)
        if call_purchase > 0:
            call_pl = call_price - call_purchase
            p1.metric("Call P/L", f"₹{call_pl:+.2f}", delta=f"{call_pl:+.2f}")
            p2.metric("Call Return", f"{call_pl / call_purchase * 100:+.1f}%")
        if put_purchase > 0:
            put_pl = put_price - put_purchase
            p3.metric("Put P/L", f"₹{put_pl:+.2f}", delta=f"{put_pl:+.2f}")
            p4.metric("Put Return", f"{put_pl / put_purchase * 100:+.1f}%")

    st.divider()

    # Greeks
    st.subheader("Greeks")

    call_delta = bs.delta("call")
    put_delta  = bs.delta("put")
    gamma      = bs.gamma()
    call_theta = bs.theta("call")
    put_theta  = bs.theta("put")
    vega       = bs.vega()
    call_rho   = bs.rho("call")
    put_rho    = bs.rho("put")

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Δ Call Delta", f"{call_delta:.4f}",
              help="Call price change per ₹1 move in spot")
    g2.metric("Δ Put Delta",  f"{put_delta:.4f}",
              help="Put price change per ₹1 move in spot")
    g3.metric("Γ Gamma",      f"{gamma:.6f}",
              help="Rate of change of delta per ₹1 move")
    g4.metric("ν Vega",       f"{vega:.4f}",
              help="Price change per 1 vol point (1%)")
    g5.metric("", "")

    g6, g7, g8, g9, g10 = st.columns(5)
    g6.metric("Θ Call Theta/day", f"₹{call_theta / 365:.4f}",
              help="Daily time decay for call")
    g7.metric("Θ Put Theta/day",  f"₹{put_theta / 365:.4f}",
              help="Daily time decay for put")
    g8.metric("ρ Call Rho", f"{call_rho:.4f}",
              help="Price change per 1% change in interest rate")
    g9.metric("ρ Put Rho",  f"{put_rho:.4f}",
              help="Price change per 1% change in interest rate")
    g10.metric("", "")

    st.divider()

    # Implied Volatility calculator (collapsed by default)
    with st.expander("🔍 Implied Volatility Calculator"):
        st.caption("Enter a market-observed option price to back out its implied volatility")
        iv1, iv2 = st.columns(2)

        with iv1:
            st.write("**Call IV**")
            mkt_call = st.number_input(
                "Market Call Price (₹)", min_value=0.01,
                value=float(round(call_price, 2)), step=1.0, key="iv_call",
            )
            try:
                iv_c = bs.implied_volatility(mkt_call, "call")
                st.success(f"Implied Vol: **{iv_c * 100:.2f}%**")
                diff_c = (iv_c - sigma) * 100
                direction = "higher" if diff_c > 0 else "lower"
                st.caption(
                    f"Your input vol: {sigma * 100:.1f}% · "
                    f"Market pricing {abs(diff_c):.1f}% {direction}"
                )
            except Exception:
                st.warning("Could not converge — try adjusting the price")

        with iv2:
            st.write("**Put IV**")
            mkt_put = st.number_input(
                "Market Put Price (₹)", min_value=0.01,
                value=float(round(put_price, 2)), step=1.0, key="iv_put",
            )
            try:
                iv_p = bs.implied_volatility(mkt_put, "put")
                st.success(f"Implied Vol: **{iv_p * 100:.2f}%**")
                diff_p = (iv_p - sigma) * 100
                direction = "higher" if diff_p > 0 else "lower"
                st.caption(
                    f"Your input vol: {sigma * 100:.1f}% · "
                    f"Market pricing {abs(diff_p):.1f}% {direction}"
                )
            except Exception:
                st.warning("Could not converge — try adjusting the price")

    st.divider()

    # Payoff diagram
    st.subheader("Payoff at Expiration")

    spot_plot    = np.linspace(K * 0.70, K * 1.30, 300)
    call_payoff  = np.maximum(spot_plot - K, 0) - call_price
    put_payoff   = np.maximum(K - spot_plot, 0) - put_price

    fig_p, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    for ax, payoff, color, label in [
        (ax1, call_payoff, "green",   "Call"),
        (ax2, put_payoff,  "crimson", "Put"),
    ]:
        ax.plot(spot_plot, payoff, color=color, linewidth=2.5)
        ax.axhline(0, color="black",  linestyle="--", linewidth=1, alpha=0.4)
        ax.axvline(K, color="gray",   linestyle="--", alpha=0.5, label=f"Strike ₹{K:,.0f}")
        ax.axvline(S, color="orange", linestyle="--", alpha=0.7, label=f"Spot ₹{S:,.0f}")
        ax.fill_between(spot_plot, payoff, 0, where=(payoff >= 0), alpha=0.2, color="green")
        ax.fill_between(spot_plot, payoff, 0, where=(payoff  < 0), alpha=0.2, color="red")
        ax.set_xlabel("Spot Price at Expiration")
        ax.set_ylabel("Profit / Loss (₹)")
        ax.set_title(f"{label} Option Payoff")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig_p.tight_layout()
    st.pyplot(fig_p)
    plt.close(fig_p)

    with st.expander("ℹ️ About the model & Greeks"):
        st.markdown("""
**Black-Scholes** prices European options assuming constant volatility and interest rate,
no dividends, and a log-normal price distribution. Options can only be exercised at expiry.

| Greek | Meaning |
|-------|---------|
| **Delta (Δ)** | Price change per ₹1 move in spot |
| **Gamma (Γ)** | Rate of change of Delta per ₹1 |
| **Theta (Θ)** | Daily time decay — value lost per day |
| **Vega (ν)**  | Price change per 1% change in volatility |
| **Rho (ρ)**   | Price change per 1% change in interest rate |
        """)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — HEATMAP
# ═════════════════════════════════════════════════════════════════════════════

with tab_heat:
    st.subheader("Scenario Analysis")

    if call_purchase == 0 and put_purchase == 0:
        st.info(
            "💡 Showing **option prices** across spot & volatility scenarios. "
            "Enter a purchase price in the sidebar to switch to **P/L mode**."
        )
    else:
        st.success("✅ P/L mode — green = profit · red = loss · yellow = near breakeven")

    # Range controls live inside this tab
    h1, h2, h3, h4 = st.columns(4)
    spot_lo = h1.number_input(
        "Min Spot", min_value=1.0,
        value=float(round(S * 0.85)),
        step=100.0,
    )
    spot_hi = h2.number_input(
        "Max Spot", min_value=spot_lo + 100,
        value=float(round(S * 1.15)),
        step=100.0,
    )

    sigma_pct     = sigma * 100
    vol_lo_def    = float(max(1.0,   round(sigma_pct * 0.5)))
    vol_hi_def    = float(min(100.0, max(vol_lo_def + 5.0, round(sigma_pct * 1.5))))

    vol_lo = h3.slider("Min Vol (%)", 1.0,              99.0,  vol_lo_def, 1.0)
    vol_hi = h4.slider("Max Vol (%)", float(vol_lo + 1), 100.0,
                        max(float(vol_lo + 1), vol_hi_def), 1.0)

    N          = 12
    spot_grid  = np.linspace(spot_lo, spot_hi, N)
    vol_grid   = np.linspace(vol_lo / 100, vol_hi / 100, N)
    x_labels   = [f"₹{s:,.0f}" for s in spot_grid]
    y_labels   = [f"{v * 100:.0f}%" for v in vol_grid]

    # Nearest grid cell for current position star
    xi = int(np.argmin(np.abs(spot_grid - S)))
    yi = int(np.argmin(np.abs(vol_grid - sigma)))

    def _build_matrix(opt_type: str, purchase: float):
        z = np.full((N, N), np.nan)
        for i, v in enumerate(vol_grid):
            for j, s in enumerate(spot_grid):
                try:
                    b = BlackScholes(s, K, T, r, v)
                    price = b.call_price() if opt_type == "call" else b.put_price()
                    z[i, j] = price - purchase if purchase > 0 else price
                except Exception:
                    pass
        return z

    def _heatmap_fig(z, purchase: float, opt_type: str):
        is_pl      = purchase > 0
        colorscale = "RdYlGn" if is_pl else ("Blues" if opt_type == "call" else "Purples")
        cb_title   = "P/L (₹)" if is_pl else "Price (₹)"
        text       = [[f"₹{z[i, j]:.1f}" for j in range(N)] for i in range(N)]

        fig = go.Figure(go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            zmid=0 if is_pl else None,
            colorbar=dict(title=cb_title),
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=9),
            hovertemplate="Spot: %{x}<br>Vol: %{y}<br>Value: %{text}<extra></extra>",
        ))

        # Blue star at current position
        fig.add_scatter(
            x=[x_labels[xi]],
            y=[y_labels[yi]],
            mode="markers",
            marker=dict(
                symbol="star", size=18, color="royalblue",
                line=dict(color="white", width=1.5),
            ),
            name="Your position",
            showlegend=True,
        )

        suffix = f"P/L  (bought ₹{purchase:.2f})" if is_pl else "Option Price"
        fig.update_layout(
            title=f"{opt_type.upper()} — {suffix}",
            xaxis_title="Spot Price",
            yaxis_title="Volatility",
            height=430,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            margin=dict(t=60, b=40, l=60, r=20),
        )
        return fig

    call_z = _build_matrix("call", call_purchase)
    put_z  = _build_matrix("put",  put_purchase)

    hc, hp = st.columns(2)
    with hc:
        st.plotly_chart(_heatmap_fig(call_z, call_purchase, "call"), use_container_width=True)
    with hp:
        st.plotly_chart(_heatmap_fig(put_z,  put_purchase,  "put"),  use_container_width=True)

    # Summary stats when in P/L mode
    if call_purchase > 0 or put_purchase > 0:
        st.divider()
        s1, s2, s3, s4 = st.columns(4)
        if call_purchase > 0:
            s1.metric("Call Max Profit", f"₹{np.nanmax(call_z):.2f}")
            s2.metric("Call Max Loss",   f"₹{np.nanmin(call_z):.2f}")
        if put_purchase > 0:
            s3.metric("Put Max Profit", f"₹{np.nanmax(put_z):.2f}")
            s4.metric("Put Max Loss",   f"₹{np.nanmin(put_z):.2f}")

    with st.expander("How to read this heatmap"):
        st.markdown("""
- **Each cell** shows the option price (or P/L if you entered a purchase price) at that spot × vol combination.
- **X-axis** — spot price range around the current level.
- **Y-axis** — volatility range.
- **Blue star** — your current position (current S and sigma).
- **P/L mode** — green = profit, red = loss, yellow = near breakeven.
- Hover over any cell for the exact value.
        """)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — STRATEGIES
# ═════════════════════════════════════════════════════════════════════════════

with tab_strat:
    st.subheader("Option Strategies")

    strategy = st.selectbox(
        "Strategy",
        ["Bull Call Spread", "Bear Put Spread", "Long Straddle", "Iron Condor"],
    )

    spot_arr = np.linspace(K * 0.70, K * 1.30, 300)

    # Default spread: 3% of strike, rounded to nearest 50
    default_spread = float(max(round(K * 0.03 / 50) * 50, 50.0))

    if strategy == "Long Straddle":
        st.caption("Buy a call **and** a put at the same strike — profits from a large move in either direction.")
        bs_k  = BlackScholes(S, K, T, r, sigma)
        cost  = bs_k.call_price() + bs_k.put_price()
        payoff = np.maximum(spot_arr - K, 0) + np.maximum(K - spot_arr, 0) - cost
        st.info(
            f"Total Cost: ₹{cost:.2f}  ·  "
            f"Upside B/E: ₹{K + cost:,.2f}  ·  "
            f"Downside B/E: ₹{K - cost:,.2f}"
        )

    else:
        sc1, sc2 = st.columns(2)
        lower_k = sc1.number_input(
            "Lower Strike", min_value=1.0,
            value=float(round((K - default_spread) / 50) * 50),
            step=50.0,
        )
        upper_k = sc2.number_input(
            "Upper Strike", min_value=lower_k + 1,
            value=float(round((K + default_spread) / 50) * 50),
            step=50.0,
        )

        if strategy == "Bull Call Spread":
            st.caption("Buy lower strike call, sell upper — limited profit & limited loss, bullish bias.")
            cost = (
                BlackScholes(S, lower_k, T, r, sigma).call_price()
                - BlackScholes(S, upper_k, T, r, sigma).call_price()
            )
            payoff = (
                np.maximum(spot_arr - lower_k, 0)
                - np.maximum(spot_arr - upper_k, 0)
                - cost
            )
            st.info(
                f"Net Cost: ₹{cost:.2f}  ·  "
                f"Max Profit: ₹{upper_k - lower_k - cost:.2f}  ·  "
                f"Max Loss: ₹{cost:.2f}"
            )

        elif strategy == "Bear Put Spread":
            st.caption("Buy upper strike put, sell lower — profits from a decline, bearish bias.")
            cost = (
                BlackScholes(S, upper_k, T, r, sigma).put_price()
                - BlackScholes(S, lower_k, T, r, sigma).put_price()
            )
            payoff = (
                np.maximum(upper_k - spot_arr, 0)
                - np.maximum(lower_k - spot_arr, 0)
                - cost
            )
            st.info(
                f"Net Cost: ₹{cost:.2f}  ·  "
                f"Max Profit: ₹{upper_k - lower_k - cost:.2f}  ·  "
                f"Max Loss: ₹{cost:.2f}"
            )

        elif strategy == "Iron Condor":
            wing = upper_k - lower_k
            k1, k2, k3, k4 = lower_k - wing, lower_k, upper_k, upper_k + wing
            st.caption(
                f"Sell put spread ({k1:,.0f}/{k2:,.0f}) + sell call spread "
                f"({k3:,.0f}/{k4:,.0f}) — profits when spot stays in range."
            )
            try:
                credit = (
                    BlackScholes(S, k2, T, r, sigma).put_price()
                    - BlackScholes(S, k1, T, r, sigma).put_price()
                    + BlackScholes(S, k3, T, r, sigma).call_price()
                    - BlackScholes(S, k4, T, r, sigma).call_price()
                )
                payoff = (
                    credit
                    - (np.maximum(k2 - spot_arr, 0) - np.maximum(k1 - spot_arr, 0))
                    - (np.maximum(spot_arr - k3, 0) - np.maximum(spot_arr - k4, 0))
                )
                st.info(
                    f"Net Credit: ₹{credit:.2f}  ·  Max Profit: ₹{credit:.2f}  ·  "
                    f"Max Loss: ₹{wing - credit:.2f}  ·  "
                    f"Profitable range: ₹{k2:,.0f}–₹{k3:,.0f}"
                )
            except ValueError as e:
                st.error(f"Error computing Iron Condor: {e}")
                payoff = np.zeros_like(spot_arr)

    # Payoff chart
    fig_s, ax_s = plt.subplots(figsize=(11, 4))
    payoff_arr = np.asarray(payoff)
    ax_s.plot(spot_arr, payoff_arr, "steelblue", linewidth=2.5)
    ax_s.axhline(0, color="black",  linestyle="--", linewidth=1, alpha=0.4)
    ax_s.axvline(S, color="orange", linestyle="--", alpha=0.7, label=f"Spot ₹{S:,.0f}")
    ax_s.axvline(K, color="gray",   linestyle="--", alpha=0.4, label=f"ATM ₹{K:,.0f}")
    ax_s.fill_between(spot_arr, payoff_arr, 0,
                      where=(payoff_arr >= 0), alpha=0.2, color="green", label="Profit")
    ax_s.fill_between(spot_arr, payoff_arr, 0,
                      where=(payoff_arr  < 0), alpha=0.2, color="red",   label="Loss")
    ax_s.set_xlabel("Spot Price at Expiration")
    ax_s.set_ylabel("Profit / Loss (₹)")
    ax_s.set_title(f"{strategy} — Payoff at Expiration")
    ax_s.legend(fontsize=9)
    ax_s.grid(True, alpha=0.2)
    fig_s.tight_layout()
    st.pyplot(fig_s)
    plt.close(fig_s)


# ── FOOTER ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<p style='text-align:center; color:#999; font-size:12px;'>"
    "Black-Scholes model · European options only · No dividends assumed · For educational purposes"
    "</p>",
    unsafe_allow_html=True,
)
