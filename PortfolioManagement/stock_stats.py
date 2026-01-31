import math

def compute_returns(prices):
    """
    Given a list of prices, compute simple returns:
    r_t = (P_t / P_{t-1}) - 1
    """
    returns = []
    for i in range(1, len(prices)):
        r_t = prices[i] / prices[i - 1] - 1
        returns.append(r_t)
    return returns

def mean(data):
    """Simple arithmetic mean."""
    return sum(data) / len(data)

def sample_std(data):
    """
    Sample standard deviation (denominator n - 1).
    """
    n = len(data)
    if n < 2:
        raise ValueError("Need at least two data points for standard deviation")
    mu = mean(data)
    squared_diffs = [(x - mu) ** 2 for x in data]
    variance = sum(squared_diffs) / (n - 1)
    return math.sqrt(variance)

def annualise_return(daily_mean_return, trading_days=252):
    """
    Annualised expected return from daily mean using compounding:
    (1 + r_daily)^N - 1
    """
    return (1 + daily_mean_return) ** trading_days - 1

def annualise_volatility(daily_std, trading_days=252):
    """
    Annualised volatility (std dev):
    sigma_annual = sigma_daily * sqrt(N)
    """
    return daily_std * math.sqrt(trading_days)

if __name__ == "__main__":
    # Example price data
    prices = [100, 101, 99, 103]
    print("Prices:", prices)

    # 1) Daily returns
    daily_returns = compute_returns(prices)
    print("\nDaily returns (decimal):")
    for i, r in enumerate(daily_returns, start=1):
        print(f" Day {i} -> {i+1}: {r:.6f} ({r*100:.4f}%)")

    # 2) Expected daily return (mean)
    mu_daily = mean(daily_returns)
    print(f"\nExpected daily return (mean): {mu_daily:.6f} ({mu_daily*100:.4f}%)")

    # 3) Daily standard deviation (risk)
    sigma_daily = sample_std(daily_returns)
    print(f"Daily standard deviation: {sigma_daily:.6f} ({sigma_daily*100:.4f}%)")

    # 4) Annualised metrics
    mu_annual = annualise_return(mu_daily)
    sigma_annual = annualise_volatility(sigma_daily)

    print(f"\nAnnualised expected return: {mu_annual:.6f} ({mu_annual*100:.4f}%)")
    print(f"Annualised volatility: {sigma_annual:.6f} ({sigma_annual*100:.4f}%)")
