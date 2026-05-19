"""FastAPI backend for options pricing and Greeks."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.pricing_engine import BlackScholes


class OptionInputs(BaseModel):
    S: float = Field(..., gt=0, description="Spot price")
    K: float = Field(..., gt=0, description="Strike price")
    T: float = Field(..., gt=0, description="Time to expiry in years")
    r: float = Field(..., ge=0, le=1, description="Risk-free rate as decimal")
    sigma: float = Field(..., gt=0, description="Volatility as decimal")


class IVInputs(OptionInputs):
    market_price: float = Field(..., gt=0)
    option_type: str = Field("call", pattern="^(call|put)$")


app = FastAPI(
    title="Options Pricing API",
    description="Black-Scholes pricing, Greeks, and implied volatility endpoints",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/price")
def price(payload: OptionInputs) -> dict[str, float]:
    try:
        bs = BlackScholes(payload.S, payload.K, payload.T, payload.r, payload.sigma)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "call_price": bs.call_price(),
        "put_price": bs.put_price(),
    }


@app.post("/greeks")
def greeks(payload: OptionInputs) -> dict[str, float]:
    try:
        bs = BlackScholes(payload.S, payload.K, payload.T, payload.r, payload.sigma)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "call_delta": bs.delta("call"),
        "put_delta": bs.delta("put"),
        "gamma": bs.gamma(),
        "call_theta_yearly": bs.theta("call"),
        "put_theta_yearly": bs.theta("put"),
        "call_theta_daily": bs.theta("call") / 365,
        "put_theta_daily": bs.theta("put") / 365,
        "vega": bs.vega(),
        "call_rho": bs.rho("call"),
        "put_rho": bs.rho("put"),
    }


@app.post("/iv")
def implied_volatility(payload: IVInputs) -> dict[str, float]:
    try:
        bs = BlackScholes(payload.S, payload.K, payload.T, payload.r, payload.sigma)
        iv = bs.implied_volatility(payload.market_price, payload.option_type)
        return {"implied_volatility": iv}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
