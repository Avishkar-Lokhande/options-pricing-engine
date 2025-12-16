"""
Options Pricing Calculator using Black-Scholes Model
Simple Streamlit app for calculating option prices and Greeks
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Black-Scholes implementation
class BlackScholes:
    """Black-Scholes model for pricing European options"""
    
    def __init__(self, S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            raise ValueError("T and sigma must be positive")
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    
    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def call_price(self):
        d1, d2 = self.d1(), self.d2()
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
    
    def put_price(self):
        d1, d2 = self.d1(), self.d2()
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
    
    def delta(self, option_type='call'):
        d1 = self.d1()
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    def gamma(self):
        return norm.pdf(self.d1()) / (self.S * self.sigma * np.sqrt(self.T))
    
    def theta(self, option_type='call'):
        d1, d2 = self.d1(), self.d2()
        common = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        if option_type == 'call':
            return common - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return common + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
    
    def vega(self):
        return self.S * np.sqrt(self.T) * norm.pdf(self.d1()) / 100
    
    def rho(self, option_type='call'):
        d2 = self.d2()
        if option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100


# Streamlit UI
st.set_page_config(page_title="Options Pricing Calculator", page_icon="📊", layout="wide")

st.title("📊 Options Pricing Calculator")
st.write("Calculate option prices and Greeks using the Black-Scholes model")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

S = st.sidebar.number_input("Spot Price (S)", min_value=1.0, value=21500.0, step=100.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=1.0, value=21500.0, step=100.0)
days = st.sidebar.slider("Days to Expiration", 1, 365, 30)
T = days / 365
r = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 15.0, 7.0, 0.5) / 100
sigma = st.sidebar.slider("Volatility (%)", 1.0, 100.0, 18.0, 1.0) / 100

st.sidebar.markdown("---")
st.sidebar.info(f"Time to expiry: {T:.4f} years ({days} days)")

st.sidebar.markdown("---")
st.sidebar.header("💰 Purchase Prices (Optional)")
st.sidebar.write("Enter your purchase price to see P/L analysis")

call_purchase = st.sidebar.number_input(
    "Call Purchase Price (₹)", 
    min_value=0.0, 
    value=0.0, 
    step=10.0,
    help="Leave as 0 to skip P/L analysis"
)

put_purchase = st.sidebar.number_input(
    "Put Purchase Price (₹)", 
    min_value=0.0, 
    value=0.0, 
    step=10.0,
    help="Leave as 0 to skip P/L analysis"
)

# Calculate prices and greeks
try:
    bs = BlackScholes(S, K, T, r, sigma)
    
    call_price = bs.call_price()
    put_price = bs.put_price()
    
    # Display prices
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📞 Call Price", f"₹{call_price:.2f}")
        if call_purchase > 0:
            call_pl = call_price - call_purchase
            st.metric("Call P/L", f"₹{call_pl:.2f}", delta=f"{call_pl:.2f}")
    with col2:
        st.metric("📉 Put Price", f"₹{put_price:.2f}")
        if put_purchase > 0:
            put_pl = put_price - put_purchase
            st.metric("Put P/L", f"₹{put_pl:.2f}", delta=f"{put_pl:.2f}")
    
    st.markdown("---")
    
    # Greeks
    st.subheader("Greeks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Call Greeks**")
        call_delta = bs.delta('call')
        gamma = bs.gamma()
        call_theta = bs.theta('call')
        vega = bs.vega()
        call_rho = bs.rho('call')
        
        greeks_call = pd.DataFrame({
            'Greek': ['Delta', 'Gamma', 'Theta (yearly)', 'Theta (daily)', 'Vega', 'Rho'],
            'Value': [
                f"{call_delta:.4f}",
                f"{gamma:.6f}",
                f"{call_theta:.2f}",
                f"{call_theta/365:.4f}",
                f"{vega:.4f}",
                f"{call_rho:.4f}"
            ]
        })
        st.dataframe(greeks_call, hide_index=True, use_container_width=True)
    
    with col2:
        st.write("**Put Greeks**")
        put_delta = bs.delta('put')
        put_theta = bs.theta('put')
        put_rho = bs.rho('put')
        
        greeks_put = pd.DataFrame({
            'Greek': ['Delta', 'Gamma', 'Theta (yearly)', 'Theta (daily)', 'Vega', 'Rho'],
            'Value': [
                f"{put_delta:.4f}",
                f"{gamma:.6f}",
                f"{put_theta:.2f}",
                f"{put_theta/365:.4f}",
                f"{vega:.4f}",
                f"{put_rho:.4f}"
            ]
        })
        st.dataframe(greeks_put, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Payoff diagram
    st.subheader("Payoff Diagram at Expiration")
    
    stock_range = np.linspace(K * 0.7, K * 1.3, 100)
    call_payoff = np.maximum(stock_range - K, 0) - call_price
    put_payoff = np.maximum(K - stock_range, 0) - put_price
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Call payoff
    ax1.plot(stock_range, call_payoff, 'g-', linewidth=2)
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(K, color='red', linestyle='--', alpha=0.5, label=f'Strike={K}')
    ax1.fill_between(stock_range, call_payoff, 0, where=(call_payoff > 0), alpha=0.3, color='green')
    ax1.fill_between(stock_range, call_payoff, 0, where=(call_payoff < 0), alpha=0.3, color='red')
    ax1.set_xlabel('Stock Price at Expiration')
    ax1.set_ylabel('Profit/Loss')
    ax1.set_title('Call Option Payoff')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Put payoff
    ax2.plot(stock_range, put_payoff, 'r-', linewidth=2)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(K, color='red', linestyle='--', alpha=0.5, label=f'Strike={K}')
    ax2.fill_between(stock_range, put_payoff, 0, where=(put_payoff > 0), alpha=0.3, color='green')
    ax2.fill_between(stock_range, put_payoff, 0, where=(put_payoff < 0), alpha=0.3, color='red')
    ax2.set_xlabel('Stock Price at Expiration')
    ax2.set_ylabel('Profit/Loss')
    ax2.set_title('Put Option Payoff')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # P/L Heatmap Section
    if call_purchase > 0 or put_purchase > 0:
        st.markdown("---")
        st.subheader("🔥 P/L Heatmap - Scenario Analysis")
        st.write("See how your profit/loss changes across different spot prices and volatilities")
        
        # Create heatmap data
        spot_range = np.linspace(S * 0.85, S * 1.15, 20)
        vol_range = np.linspace(sigma * 0.5, sigma * 1.5, 20)
        
        # Choose which option to display
        option_choice = st.radio("Select Option:", ["Call", "Put"], horizontal=True)
        
        # Calculate P/L for each combination
        pl_matrix = np.zeros((len(vol_range), len(spot_range)))
        
        for i, vol in enumerate(vol_range):
            for j, spot in enumerate(spot_range):
                try:
                    bs_temp = BlackScholes(spot, K, T, r, vol)
                    if option_choice == "Call":
                        theoretical_price = bs_temp.call_price()
                        pl_matrix[i, j] = theoretical_price - call_purchase if call_purchase > 0 else 0
                    else:
                        theoretical_price = bs_temp.put_price()
                        pl_matrix[i, j] = theoretical_price - put_purchase if put_purchase > 0 else 0
                except:
                    pl_matrix[i, j] = 0
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use RdYlGn colormap (Red-Yellow-Green)
        im = ax.imshow(pl_matrix, cmap='RdYlGn', aspect='auto', 
                      extent=[spot_range[0], spot_range[-1], vol_range[0]*100, vol_range[-1]*100],
                      origin='lower')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('P/L (₹)', rotation=270, labelpad=20, fontsize=11)
        
        # Add contour lines for zero P/L
        contour = ax.contour(spot_range, vol_range*100, pl_matrix, levels=[0], colors='black', linewidths=2)
        ax.clabel(contour, inline=True, fontsize=10, fmt='Breakeven')
        
        # Labels and title
        ax.set_xlabel('Spot Price (₹)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volatility (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{option_choice} Option P/L Heatmap\nGreen = Profit | Red = Loss', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Add current position marker
        ax.plot(S, sigma*100, 'b*', markersize=20, label='Current Position', markeredgecolor='white', markeredgewidth=2)
        ax.legend(fontsize=11)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Profit", f"₹{np.max(pl_matrix):.2f}")
        with col2:
            st.metric("Max Loss", f"₹{np.min(pl_matrix):.2f}")
        with col3:
            current_pl = call_price - call_purchase if option_choice == "Call" else put_price - put_purchase
            st.metric("Current P/L", f"₹{current_pl:.2f}")
        
        with st.expander("📊 How to Read This Heatmap"):
            st.write("""
            **This heatmap shows your profit/loss for different market scenarios:**
            
            - **X-axis (Horizontal):** Different spot prices (stock/index levels)
            - **Y-axis (Vertical):** Different volatility levels
            - **Colors:** 
              - 🟢 **Green** = Profit (theoretical price > your purchase price)
              - 🟡 **Yellow** = Near breakeven
              - 🔴 **Red** = Loss (theoretical price < your purchase price)
            - **Blue Star (⭐):** Your current position
            - **Black Line:** Breakeven points (zero P/L)
            
            **Example:** If the spot moves to ₹22,000 and volatility increases to 25%, 
            find that cell to see your expected P/L.
            
            **Use Case:** Before buying an option, check different scenarios to understand your risk.
            """)
    
    # Info section
    st.markdown("---")
    with st.expander("ℹ️ About the Model"):
        st.write("""
        **Black-Scholes Model** is used for pricing European options. 
        
        **Greeks:**
        - **Delta**: Change in option price for ₹1 change in stock price
        - **Gamma**: Rate of change of Delta
        - **Theta**: Time decay (how much value the option loses per day)
        - **Vega**: Sensitivity to volatility changes
        - **Rho**: Sensitivity to interest rate changes
        
        **Assumptions:**
        - European options (can only be exercised at expiration)
        - No dividends
        - Constant volatility and interest rate
        - Log-normal stock price distribution
        """)

except ValueError as e:
    st.error(f"Error: {e}")
    st.warning("Please check your input values.")

# Footer
st.markdown("---")
st.markdown("Built with Python, NumPy, SciPy, and Streamlit")
