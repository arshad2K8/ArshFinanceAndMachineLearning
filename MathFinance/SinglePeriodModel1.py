#!/usr/bin/env python3
"""
Single-period (one-period) market model — 2 states (Up/Down), 3 assets (A, B, Risk-free).

Example data:
- Asset A: P0 = 100, payoffs = [120 (Up), 80 (Down)]
- Asset B: P0 = 50,  payoffs = [60 (Up),  40 (Down)]
- Risk-free: P0 = 1, payoff = [1.05, 1.05]

What this script does:
1) Solves for state prices q (one per state) from the risky assets.
2) Checks non-negativity of q (no-arbitrage condition).
3) Compares the **implied** risk-free price from q to the **given** risk-free price (consistency check).
4) Provides helpers to price any new payoff vector and to report findings.
"""

import numpy as np

# -----------------------------
# 1) MODEL INPUTS (edit freely)
# -----------------------------

# Prices today (time 0)
P0 = np.array([
    100.0,  # Asset A
    50.0,   # Asset B
    1.0     # Risk-free asset
])

# Payoffs at time 1 (rows = assets, cols = states: [Up, Down])
X = np.array([
    [120.0,  80.0],   # Asset A
    [ 60.0,  40.0],   # Asset B
    [  1.05,  1.05],  # Risk-free
])

asset_names = ["A", "B", "Risk-free"]
state_names = ["Up", "Down"]

# Numerical tolerance for comparisons
TOL = 1e-9


# ---------------------------------------
# 2) Solve for state prices from risky set
# ---------------------------------------
# Use only risky assets to identify state prices (square system: 2 assets x 2 states)
X_risky = X[:2, :]          # shape (2, 2)
P0_risky = P0[:2]           # shape (2,)

# Ensure the risky payoff matrix is invertible
if np.linalg.matrix_rank(X_risky) < 2:
    raise ValueError("Risky payoff matrix is not full rank; cannot uniquely recover state prices.")

q = np.linalg.solve(X_risky, P0_risky)   # state price vector, shape (2,)

# ---------------------------------------
# 3) Diagnostics: NA check & consistency
# ---------------------------------------
no_arbitrage_state_prices = np.all(q >= -TOL)  # allow tiny numerical negatives
q = np.maximum(q, 0) if no_arbitrage_state_prices else q  # clip tiny negatives for display

# Implied risk-free price from state prices:
# Price = sum_s q_s * payoff_s  (here payoff_s for RF is the same in both states)
rf_payoff = X[2, :]                 # [1.05, 1.05]
P0_rf_implied = rf_payoff @ q       # dot product

# Implied discount factor and risk-free rate from state prices:
discount_factor_implied = np.sum(q)             # because a $1 in each state prices the risk-free promise of 1 next period
r_f_implied = (1.0 / discount_factor_implied) - 1.0

# Consistency check with the GIVEN risk-free asset
rf_consistent = abs(P0_rf_implied - P0[2]) <= 1e-8


# ---------------------------------------
# 4) Helper: price any new payoff vector
# ---------------------------------------
def price_from_state_prices(payoff_by_state: np.ndarray) -> float:
    """
    payoff_by_state: shape (2,), payoffs in [Up, Down]
    returns: time-0 price using state prices q
    """
    payoff_by_state = np.asarray(payoff_by_state, dtype=float)
    if payoff_by_state.shape != (2,):
        raise ValueError("payoff_by_state must be a length-2 vector: [Up, Down].")
    return float(payoff_by_state @ q)


# ---------------------------------------
# 5) Report
# ---------------------------------------
print("=== Single-Period Market Model (2 states) ===\n")
print("Assets and payoffs:")
for i, name in enumerate(asset_names):
    print(f"  {name:10s} | P0 = {P0[i]:8.4f} | Payoffs (Up, Down) = {X[i, 0]:8.4f}, {X[i, 1]:8.4f}")
print()

print("State names:", state_names)
print("Solved state prices q (price today of $1 in each state):")
for s, qs in zip(state_names, q):
    print(f"  q[{s}] = {qs:.10f}")
print()

print(f"No-arbitrage condition q >= 0? -> {no_arbitrage_state_prices}")
if not no_arbitrage_state_prices:
    print("  WARNING: Negative state prices detected -> arbitrage exists.")
print()

print("Implied discount factor and risk-free rate from q:")
print(f"  Sum(q) = {discount_factor_implied:.10f}  -> implied gross RF = 1/Sum(q) = {(1/discount_factor_implied):.10f}")
print(f"  Implied r_f = {r_f_implied*100:.6f}%")
print()

print("Risk-free consistency check:")
print(f"  Given RF price P0 = {P0[2]:.10f}")
print(f"  Implied RF price from q and payoff [1.05, 1.05] = {P0_rf_implied:.10f}")
print(f"  Consistent? -> {rf_consistent}")
if not rf_consistent:
    diff = P0_rf_implied - P0[2]
    direction = "overpriced vs q" if diff > 0 else "underpriced vs q"
    print(f"  -> Inconsistent by {diff:.10e} ({direction}); indicates an arbitrage or data inconsistency.")
print()

# ---------------------------------------
# 6) Example: price some new payoffs
# ---------------------------------------
examples = {
    "Digital (pays 10 if Up, 0 if Down)": np.array([10.0, 0.0]),
    "Binary Down (0 if Up, 10 if Down)":  np.array([0.0, 10.0]),
    "Straddle-like (pays 5 in both states)": np.array([5.0, 5.0]),
    "Custom (pays 15 if Up, 2 if Down)":  np.array([15.0, 2.0]),
}

print("Pricing examples using state prices q:")
for name, payoff in examples.items():
    print(f"  {name:35s} -> P0 = {price_from_state_prices(payoff):.6f}")
print()

# ---------------------------------------
# 7) Optional: sanity check — reprice traded assets
# ---------------------------------------
print("Repricing traded assets from q (should match P0 if consistent):")
for i, name in enumerate(asset_names):
    p0_implied = float(X[i, :] @ q)
    print(f"  {name:10s}: Given P0 = {P0[i]:8.6f} | Implied P0 = {p0_implied:8.6f} | Diff = {p0_implied - P0[i]: .3e}")
print("\nDone.")
