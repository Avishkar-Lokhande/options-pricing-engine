# Options Pricing Calculator

A simple options pricing calculator using the Black-Scholes model. Built with Python and Streamlit.

## What it does

- Calculates European call and put option prices
- Shows all the Greeks (Delta, Gamma, Theta, Vega, Rho)
- Visualizes payoff diagrams
- Interactive web interface

## How to use

### Running locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the calculator

1. Enter your parameters in the sidebar:
   - Spot price (current stock/index price)
   - Strike price
   - Days to expiration
   - Risk-free rate (usually around 7% for India)
   - Volatility (historical or implied)

2. The app will show you:
   - Call and put prices
   - All the Greeks
   - Payoff diagrams showing profit/loss at expiration

## The Math

The Black-Scholes formula for a call option:

```
C = S×N(d₁) - K×e^(-rT)×N(d₂)
```

Where:
- S = spot price
- K = strike price
- T = time to expiration (years)
- r = risk-free rate
- σ = volatility
- N(x) = cumulative normal distribution

And:
```
d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

## Project Structure

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Technologies

- Python 3.x
- Streamlit (web interface)
- NumPy (numerical calculations)
- SciPy (normal distribution functions)
- Pandas (data handling)
- Matplotlib (charts)

## Notes

- This implements the Black-Scholes model for **European options** (can only be exercised at expiration)
- Doesn't account for dividends
- Assumes constant volatility and interest rates
- Real market prices may differ due to factors like supply/demand, volatility smile, early exercise, etc.

## References

- [Options, Futures, and Other Derivatives](https://www.amazon.com/Options-Futures-Other-Derivatives-9th/dp/0133456315) by John Hull
- [Black-Scholes Model on Wikipedia](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
- [Investopedia: Greeks](https://www.investopedia.com/terms/g/greeks.asp)

## License

Feel free to use this for learning and portfolio projects.
