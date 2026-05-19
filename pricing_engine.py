"""Core Black-Scholes pricing engine for API and future frontends."""

from __future__ import annotations

import math

from scipy.stats import norm


class BlackScholes:
    """Black-Scholes model for pricing European options and Greeks."""

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float) -> None:
        if S <= 0 or K <= 0:
            raise ValueError("S and K must be positive")
        if T <= 0:
            raise ValueError("T must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)

    def d1(self) -> float:
        return (
            math.log(self.S / self.K)
            + (self.r + 0.5 * self.sigma**2) * self.T
        ) / (self.sigma * math.sqrt(self.T))

    def d2(self) -> float:
        return self.d1() - self.sigma * math.sqrt(self.T)

    def call_price(self) -> float:
        d1, d2 = self.d1(), self.d2()
        return self.S * norm.cdf(d1) - self.K * math.exp(-self.r * self.T) * norm.cdf(d2)

    def put_price(self) -> float:
        d1, d2 = self.d1(), self.d2()
        return self.K * math.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    def delta(self, option_type: str = "call") -> float:
        d1 = self.d1()
        if option_type == "call":
            return norm.cdf(d1)
        if option_type == "put":
            return norm.cdf(d1) - 1
        raise ValueError("option_type must be 'call' or 'put'")

    def gamma(self) -> float:
        return norm.pdf(self.d1()) / (self.S * self.sigma * math.sqrt(self.T))

    def theta(self, option_type: str = "call") -> float:
        d1, d2 = self.d1(), self.d2()
        common = -(self.S * norm.pdf(d1) * self.sigma) / (2 * math.sqrt(self.T))
        if option_type == "call":
            return common - self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(d2)
        if option_type == "put":
            return common + self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(-d2)
        raise ValueError("option_type must be 'call' or 'put'")

    def vega(self) -> float:
        # Returned per 1 vol point (1%) to match common trading convention.
        return self.S * math.sqrt(self.T) * norm.pdf(self.d1()) / 100

    def rho(self, option_type: str = "call") -> float:
        d2 = self.d2()
        if option_type == "call":
            return self.K * self.T * math.exp(-self.r * self.T) * norm.cdf(d2) / 100
        if option_type == "put":
            return -self.K * self.T * math.exp(-self.r * self.T) * norm.cdf(-d2) / 100
        raise ValueError("option_type must be 'call' or 'put'")

    def implied_volatility(
        self,
        market_price: float,
        option_type: str = "call",
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> float:
        sigma = 0.5

        for _ in range(max_iterations):
            bs_temp = BlackScholes(self.S, self.K, self.T, self.r, sigma)
            price = bs_temp.call_price() if option_type == "call" else bs_temp.put_price()
            vega = bs_temp.vega() * 100

            diff = market_price - price
            if abs(diff) < tolerance:
                return sigma

            if vega == 0:
                break

            sigma = sigma + diff / vega
            if sigma <= 0:
                sigma = 0.01

        raise ValueError("Implied volatility did not converge")
