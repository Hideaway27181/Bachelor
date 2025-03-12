import numpy as np
from numpy.polynomial import Polynomial


def ls_american_option_quadratic_iter(X, t, r, strike):
    """
    Generator that performs the Longstaff-Schwartz backward iteration
    for an American put using a 2nd-degree polynomial (Quadratic) in x.
    """
    # At final time, payoff is just the European payoff
    cashflow = np.maximum(strike - X[-1, :], 0.0)
    
    # Go backwards from next-to-last time step down to the first
    for i in reversed(range(1, X.shape[0] - 1)):
        # discount factor between t[i] and t[i+1]
        dt = (t[i+1] - t[i])
        df = np.exp(-r * dt)
        
        # discount future cashflows from next period
        cashflow = cashflow * df
        
        x = X[i, :]
        exercise = np.maximum(strike - x, 0.0)
        
        # in-the-money paths
        itm = (exercise > 0)
        
        # Fit a 2nd-degree polynomial on the in-the-money subset
        # x[itm], cashflow[itm]
        fitted = Polynomial.fit(x[itm], cashflow[itm], deg=2)
        continuation = fitted(x)
        
        # early-exercise decision
        ex_idx = itm & (exercise > continuation)
        cashflow[ex_idx] = exercise[ex_idx]
        
        yield cashflow, x, fitted, continuation, exercise, ex_idx


def longstaff_schwartz_american_option_quadratic(X, t, r, strike):
    """
    Runs the LSM iteration (Quadratic) to price an American put.
    Returns the present value at time t[0].
    """
    gen = ls_american_option_quadratic_iter(X, t, r, strike)
    # run the generator to completion
    cashflow = None
    for c, *_ in gen:
        cashflow = c
    
    if cashflow is None:
        raise ValueError("No cashflow computed in the iteration!")
    
    # discount from t[1] back to t[0]
    dt = (t[1] - t[0])
    return cashflow.mean() * np.exp(-r * dt)


# Helper: simulate GBM


def simulate_geometric_brownian_paths(S0, r, sigma, T, N, n_steps, seed=None):
    """
    Simulate N GBM paths with n_steps over [0, T], returning an array
    of shape (n_steps+1, N) -- time in first dimension, path in second.
    This aligns with X[i, :] = stock prices at time step i.
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    # shape = (n_steps+1, N)
    S = np.zeros((n_steps + 1, N))
    S[0, :] = S0
    
    for i in range(n_steps):
        z = np.random.normal(size=N)
        S[i+1, :] = S[i, :] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
    
    return S


# Main demonstration


if __name__ == "__main__":
    # Parameters from the paper's Table 1 for the American put
    S0     = 36.0 # price on the underlying 
    K      = 40.0 # Strike price
    r      = 0.06 # risk free rate
    sigma  = 0.20 # volatility
    T      = 2.0 
    n_steps = 50     # 50 time steps 
    N       = 100000 # number of paths
    
    # Build time grid
    t = np.linspace(0, T, n_steps + 1)
    
    # Simulate paths: shape is (n_steps+1, N)
    X = simulate_geometric_brownian_paths(S0, r, sigma, T, N, n_steps, seed=52)
    
    # Price the American put using your LSM with quadratic polynomials
    price = longstaff_schwartz_american_option_quadratic(X, t, r, K)
    
    print(f"Parameters => S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}, n_steps={n_steps}, N={N}")
    print(f"American put via LSM Quadratic = {price:.4f}")