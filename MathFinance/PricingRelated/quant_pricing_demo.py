
from math import log, sqrt, exp, erf
import numpy as np

SQRT_2 = sqrt(2.0)
def norm_cdf(x): return 0.5*(1.0 + erf(x / SQRT_2))
def norm_pdf(x): return (1.0/np.sqrt(2.0*np.pi))*np.exp(-0.5*x*x)

def d1_d2(S0, K, r, sigma, T, q=0.0):
    if sigma <= 0 or T <= 0 or S0 <= 0 or K <= 0:
        raise ValueError("S0, K, sigma, T must be positive (sigma, T > 0).")
    d1 = (log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2

def bs_call(S0, K, r, sigma, T, q=0.0):
    d1, d2 = d1_d2(S0, K, r, sigma, T, q)
    return S0 * exp(-q * T) * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)

def bs_put(S0, K, r, sigma, T, q=0.0):
    d1, d2 = d1_d2(S0, K, r, sigma, T, q)
    return K * exp(-r * T) * norm_cdf(-d2) - S0 * exp(-q * T) * norm_cdf(-d1)

def mc_euro_call(S0, K, r, sigma, T, q=0.0, paths=100_000, seed=123):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(paths)
    drift = (r - q - 0.5 * sigma * sigma) * T
    diff = sigma * sqrt(T) * Z
    ST = S0 * np.exp(drift + diff)
    payoff = np.maximum(ST - K, 0.0)
    return np.exp(-r * T) * payoff.mean()
