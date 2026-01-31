# Example: Portfolio optimization & evaluation in Python (synthetic data you can replace with Bloomberg data)
# - Builds an efficient frontier
# - Compares equal-weight, min-variance, and max-Sharpe portfolios
# - Computes Sharpe, Treynor, Beta, and Jensen's Alpha
#
# How to use with your own data later:
# 1) Replace 'prices_df' with a DataFrame of weekly or monthly adjusted close prices for 6–10 tickers (columns = tickers; index = DateTime)
# 2) Replace 'benchmark_prices' with your benchmark’s prices (e.g., S&P 500 / FTSE 100)
# 3) Replace rf_annual with your annualised risk-free rate (e.g., 3M T-bill annualised)
#
# This cell also writes:
# - '/mnt/data/portfolio_example_outputs.xlsx' with weights and metrics tables
# - '/mnt/data/template_prices.csv' as a CSV template for your Bloomberg export format

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from math import sqrt
from datetime import datetime, timedelta

from caas_jupyter_tools import display_dataframe_to_user

# ----------------------
# 0) Helper functions
# ----------------------
def annualize_return(mean_periodic, periods_per_year):
    return mean_periodic * periods_per_year

def annualize_vol(vol_periodic, periods_per_year):
    return vol_periodic * np.sqrt(periods_per_year)

def portfolio_stats(weights, mu_vec, cov_mat, rf_annual, periods_per_year):
    w = np.array(weights)
    mu_p = float(np.dot(w, mu_vec))  # periodic expected return
    vol_p = float(np.sqrt(np.dot(w, cov_mat).dot(w)))
    mu_ann = annualize_return(mu_p, periods_per_year)
    vol_ann = annualize_vol(vol_p, periods_per_year)
    sharpe = (mu_ann - rf_annual) / vol_ann if vol_ann > 0 else np.nan
    return mu_ann, vol_ann, sharpe

def compute_beta_and_alpha(port_rets, bench_rets, rf_periodic, periods_per_year):
    # Use periodic returns for regression-style beta and alpha (no internet, do simple formulas)
    # Beta = Cov(Rp, Rm) / Var(Rm). Jensen's alpha (annualized): 
    # alpha_ann = [ (mean(Rp) - rf_periodic) - beta*(mean(Rm) - rf_periodic) ] * periods_per_year
    var_m = np.var(bench_rets, ddof=1)
    cov_pm = np.cov(port_rets, bench_rets, ddof=1)[0,1]
    beta = cov_pm / var_m if var_m > 0 else np.nan
    mean_p = np.mean(port_rets)
    mean_m = np.mean(bench_rets)
    alpha_periodic = (mean_p - rf_periodic) - beta * (mean_m - rf_periodic)
    alpha_annual = alpha_periodic * periods_per_year
    return beta, alpha_annual

def solve_min_variance(cov_mat, bounds, periods_per_year):
    n = cov_mat.shape[0]
    init = np.repeat(1.0/n, n)
    cons = ({'type':'eq','fun': lambda w: np.sum(w) - 1})
    res = minimize(lambda w: np.dot(w, cov_mat).dot(w),
                   init, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x

def solve_max_sharpe(mu_vec, cov_mat, rf_annual, periods_per_year, bounds):
    n = len(mu_vec)
    init = np.repeat(1.0/n, n)
    cons = ({'type':'eq','fun': lambda w: np.sum(w) - 1})
    # minimize negative Sharpe
    def neg_sharpe(w):
        mu_ann, vol_ann, sharpe = portfolio_stats(w, mu_vec, cov_mat, rf_annual, periods_per_year)
        return -sharpe
    res = minimize(neg_sharpe, init, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x

def solve_efficient_frontier(mu_vec, cov_mat, target_returns, bounds):
    n = len(mu_vec)
    init = np.repeat(1.0/n, n)
    cons_base = ({'type':'eq','fun': lambda w: np.sum(w) - 1},)
    sols = []
    vols = []
    for tr in target_returns:
        def port_var(w): return np.dot(w, cov_mat).dot(w)
        cons = cons_base + ({'type':'eq','fun': lambda w, tr=tr: np.dot(w, mu_vec) - tr},)
        res = minimize(port_var, init, method='SLSQP', bounds=bounds, constraints=cons)
        if res.success:
            w = res.x
            sols.append(w)
            vols.append(np.sqrt(np.dot(w, cov_mat).dot(w)))
        else:
            sols.append(np.full(n, np.nan))
            vols.append(np.nan)
    return np.array(sols), np.array(vols)

# ----------------------
# 1) Create synthetic price data you can replace later
# ----------------------
rng = default_rng(42)
tickers = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "KO", "NVDA", "AMZN"]  # 8 names (fits 6–10 requirement)
periods_per_year = 52  # we'll use weekly returns
years = 4
n = len(tickers)

dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=years*periods_per_year, freq="W-FRI")

# Build a random but stable covariance using a simple factor model
k = 3  # factors
loadings = rng.normal(0, 0.5, size=(n, k))
factor_cov = np.diag(rng.uniform(0.5, 1.5, size=k))
idiosyncratic = np.diag(rng.uniform(0.05, 0.15, size=n))
cov_weekly = loadings @ factor_cov @ loadings.T + idiosyncratic

# Simulate weekly drifts and then prices via GBM-ish compounding
mu_weekly = rng.uniform(0.0008, 0.004, size=n)  # ~4% to 20% annualised ballpark
L = np.linalg.cholesky(cov_weekly)
Z = rng.normal(size=(len(dates), n))
eps = Z @ L.T  # correlated shocks
rets = mu_weekly + eps  # weekly arithmetic returns (small enough)

# Convert to price paths starting at arbitrary base (100)
prices = np.exp(np.log(100) + np.cumsum(np.log1p(rets), axis=0))
prices_df = pd.DataFrame(prices, index=dates, columns=tickers)

# Benchmark (market) synthetic series correlated with average of assets
bench_mu = 0.0025
bench_sigma = 0.05
bench_noise = rng.normal(bench_mu, bench_sigma, size=len(dates))
# add some correlation by mixing with cross-section average returns
bench_rets = 0.6 * rets.mean(axis=1) + 0.4 * bench_noise
benchmark_prices = 100 * (1 + pd.Series(bench_rets, index=dates)).cumprod()

# Risk-free (annualised). Convert to periodic for some calcs.
rf_annual = 0.03
rf_periodic = rf_annual / periods_per_year

# ----------------------
# 2) Compute returns and inputs for MPT
# ----------------------
# Use simple % weekly returns to align with Bloomberg weekly sampling option
R = prices_df.pct_change().dropna()
R_bench = pd.Series(benchmark_prices).pct_change().dropna()
mu_vec = R.mean().values  # periodic mean vector
cov_mat = R.cov().values  # periodic cov matrix

# ----------------------
# 3) Solve portfolios
# ----------------------
bounds = tuple((0.0, 1.0) for _ in range(n))  # long-only, no shorting
w_equal = np.repeat(1.0/n, n)
w_minvar = solve_min_variance(cov_mat, bounds, periods_per_year)
w_maxsharpe = solve_max_sharpe(mu_vec, cov_mat, rf_annual, periods_per_year, bounds)

# Efficient frontier across a grid of target periodic returns
target_periodic = np.linspace(mu_vec.min()*0.8, mu_vec.max()*1.2, 30)
ef_weights, ef_vols = solve_efficient_frontier(mu_vec, cov_mat, target_periodic, bounds)
ef_returns_ann = annualize_return(target_periodic, periods_per_year)
ef_vols_ann = annualize_vol(ef_vols, periods_per_year)

# ----------------------
# 4) Evaluate performance metrics
# ----------------------
def portfolio_periodic_returns(weights, R):
    return (R.values @ np.array(weights)).astype(float)

# Equal-weight
mu_eq_ann, vol_eq_ann, sharpe_eq = portfolio_stats(w_equal, mu_vec, cov_mat, rf_annual, periods_per_year)
rp_eq = portfolio_periodic_returns(w_equal, R)
beta_eq, alpha_eq_ann = compute_beta_and_alpha(rp_eq, R_bench.values, rf_periodic, periods_per_year)
treynor_eq = (mu_eq_ann - rf_annual) / beta_eq if beta_eq and beta_eq != 0 else np.nan

# Min-variance
mu_mv_ann, vol_mv_ann, sharpe_mv = portfolio_stats(w_minvar, mu_vec, cov_mat, rf_annual, periods_per_year)
rp_mv = portfolio_periodic_returns(w_minvar, R)
beta_mv, alpha_mv_ann = compute_beta_and_alpha(rp_mv, R_bench.values, rf_periodic, periods_per_year)
treynor_mv = (mu_mv_ann - rf_annual) / beta_mv if beta_mv and beta_mv != 0 else np.nan

# Max-Sharpe
mu_ms_ann, vol_ms_ann, sharpe_ms = portfolio_stats(w_maxsharpe, mu_vec, cov_mat, rf_annual, periods_per_year)
rp_ms = portfolio_periodic_returns(w_maxsharpe, R)
beta_ms, alpha_ms_ann = compute_beta_and_alpha(rp_ms, R_bench.values, rf_periodic, periods_per_year)
treynor_ms = (mu_ms_ann - rf_annual) / beta_ms if beta_ms and beta_ms != 0 else np.nan

# ----------------------
# 5) Tables for weights and metrics
# ----------------------
weights_table = pd.DataFrame({
    "Ticker": tickers,
    "Equal_Weight": w_equal,
    "Min_Var": w_minvar,
    "Max_Sharpe": w_maxsharpe
})

metrics_table = pd.DataFrame([
    {"Portfolio":"Equal-Weight", "Exp Return (ann)": mu_eq_ann, "Vol (ann)": vol_eq_ann, "Sharpe": sharpe_eq, "Beta": beta_eq, "Treynor": treynor_eq, "Jensen Alpha (ann)": alpha_eq_ann},
    {"Portfolio":"Min-Variance", "Exp Return (ann)": mu_mv_ann, "Vol (ann)": vol_mv_ann, "Sharpe": sharpe_mv, "Beta": beta_mv, "Treynor": treynor_mv, "Jensen Alpha (ann)": alpha_mv_ann},
    {"Portfolio":"Max-Sharpe", "Exp Return (ann)": mu_ms_ann, "Vol (ann)": vol_ms_ann, "Sharpe": sharpe_ms, "Beta": beta_ms, "Treynor": treynor_ms, "Jensen Alpha (ann)": alpha_ms_ann},
])

# Display to user
display_dataframe_to_user("Portfolio Weights (Example)", weights_table.round(4))
display_dataframe_to_user("Portfolio Metrics (Example)", metrics_table.round(4))

# ----------------------
# 6) Plots (per tool rules: matplotlib only, one plot per figure, no custom colors)
# ----------------------
# Efficient frontier
plt.figure()
plt.plot(ef_vols_ann, ef_returns_ann, marker='o', linestyle='-')
plt.xlabel("Volatility (Annualised)")
plt.ylabel("Expected Return (Annualised)")
plt.title("Efficient Frontier (Synthetic Example)")
plt.scatter([vol_eq_ann, vol_mv_ann, vol_ms_ann],
            [mu_eq_ann,  mu_mv_ann,  mu_ms_ann])
plt.legend(["Frontier","Equal-Weight","Min-Var","Max-Sharpe"])
plt.show()

# Weights bar chart for Max-Sharpe
plt.figure()
plt.bar(tickers, w_maxsharpe)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Weight")
plt.title("Max-Sharpe Portfolio Weights")
plt.tight_layout()
plt.show()

# Risk/Return scatter of individual assets
asset_returns_ann = annualize_return(mu_vec, periods_per_year)
asset_vols_ann = annualize_vol(np.sqrt(np.diag(cov_mat)), periods_per_year)
plt.figure()
plt.scatter(asset_vols_ann, asset_returns_ann)
for i, t in enumerate(tickers):
    plt.annotate(t, (asset_vols_ann[i], asset_returns_ann[i]))
plt.xlabel("Asset Volatility (Annualised)")
plt.ylabel("Asset Expected Return (Annualised)")
plt.title("Assets: Risk vs Return")
plt.show()

# ----------------------
# 7) Save outputs and CSV template
# ----------------------
with pd.ExcelWriter("/mnt/data/portfolio_example_outputs.xlsx", engine="xlsxwriter") as writer:
    weights_table.to_excel(writer, sheet_name="Weights", index=False)
    metrics_table.to_excel(writer, sheet_name="Metrics", index=False)
    pd.DataFrame({"Date": R.index}).to_excel(writer, sheet_name="Notes", index=False)

# CSV template the group can fill with Bloomberg prices
template = prices_df.copy()
template.insert(0, "Date", template.index.strftime("%Y-%m-%d"))
template["Benchmark"] = benchmark_prices.values
template.to_csv("/mnt/data/template_prices.csv", index=False)

print("Files written:\n- /mnt/data/portfolio_example_outputs.xlsx\n- /mnt/data/template_prices.csv")
