# Options Pricing and Greeks Calculator

Black-Scholes based options calculator with:
- Streamlit app (`app.py`) for full-feature interactive usage
- FastAPI backend (`backend/api.py`) for API-first architecture
- Ad-friendly static frontend (`web/`) for Netlify/Vercel hosting and monetization

## Features

- European call/put pricing
- Greeks: Delta, Gamma, Theta, Vega, Rho
- Implied volatility (Newton-Raphson)
- Streamlit UI with live market helpers
- API endpoints for web/mobile integration
- Static wrapper page with ad slot and legal pages

## Project Structure

```text
.
├── app.py
├── backend/
│   ├── __init__.py
│   ├── api.py
│   └── pricing_engine.py
├── web/
│   ├── index.html
│   ├── script.js
│   ├── styles.css
│   ├── privacy.html
│   ├── terms.html
│   └── disclaimer.html
├── requirements.txt
├── main.ipynb
├── TUTORIAL.md
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Streamlit app (existing full app)

```bash
streamlit run app.py
```

### 3. Run API backend (for frontend/API mode)

```bash
uvicorn backend.api:app --reload
```

API docs open at `http://127.0.0.1:8000/docs`.

### 4. Run static monetization frontend

Use any static server from the `web` folder. Example with Python:

```bash
python -m http.server 8080
```

Then open `http://127.0.0.1:8080/web/`.

## Deploy for Ads

### Fastest path (no heavy rewrite)

1. Deploy Streamlit app (Streamlit Cloud / Render / Railway).
2. Deploy `web/` to Netlify or Vercel.
3. In `web/index.html`:
- Add your real AdSense script (`ca-pub-...`).
- Create and insert real ad units where `AdSense slot` placeholder exists.
- Set your deployed Streamlit URL in Embedded Streamlit Mode.

This gives monetization with minimal risk: ads on your static website, calculator in iframe.

### API-first path (better long-term)

1. Deploy FastAPI backend (Render/Railway/Fly.io).
2. In `web/index.html`, set API Base URL to your deployed API.
3. Use API Calculator Mode for direct backend calls.

## API Endpoints

- `GET /health`
- `POST /price`
- `POST /greeks`
- `POST /iv`

Example payload:

```json
{
  "S": 21500,
  "K": 21500,
  "T": 0.0822,
  "r": 0.07,
  "sigma": 0.18
}
```

## SEO and Approval Checklist

- Keep legal pages: `web/privacy.html`, `web/terms.html`, `web/disclaimer.html`
- Add original educational content pages (important for AdSense approval)
- Add analytics and Search Console
- Target intent keywords like "options greeks calculator India"

## Notes

- Black-Scholes assumptions: European options, constant volatility/rates, no dividends.
- Model output is theoretical and may differ from market prices.

