import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
DELTA = 0.99  # Time discount factor
LAMBDA = 2.0  # Loss aversion parameter
GAMMA = 1.0   # Risk aversion parameter
GROWTH_MEAN = 0.02  # Mean consumption growth
GROWTH_STD = 0.02   # Consumption growth volatility
RF = np.exp(0.0198) / DELTA  # Risk-free rate

def simulate_consumption_growth(num_draws: int = int(1e4)) -> np.ndarray:
    """Simulate consumption growth using normal distribution."""
    epsilon = np.random.randn(num_draws)
    return np.exp(GROWTH_MEAN + GROWTH_STD * epsilon)

def utility_function(R: float, Rf: float, lambd: float = LAMBDA) -> float:
    """Calculate utility v(R) based on return R."""
    return R - Rf if R >= Rf else lambd * (R - Rf)

def compute_ex(x: float, b0: float, g_tilde: np.ndarray, delta: float, Rf: float) -> float:
    """Compute e(x) function value."""
    R_tilde = x * g_tilde
    v_vals = np.array([utility_function(R, Rf) for R in R_tilde])
    Ev = np.mean(v_vals)
    return delta * b0 * Ev + delta * x - 1

def find_equilibrium_x(b0: float, g_tilde: np.ndarray, delta: float, Rf: float,
                      x_minus: float = 1.0, x_plus: float = 1.1,
                      tol: float = 1e-5, max_iter: int = 1000) -> float:
    """Find equilibrium x using bisection method."""
    e_minus = compute_ex(x_minus, b0, g_tilde, delta, Rf)
    e_plus = compute_ex(x_plus, b0, g_tilde, delta, Rf)
    
    if e_minus >= 0 or e_plus <= 0:
        return np.nan
        
    iter_count = 0
    while True:
        x_0 = 0.5 * (x_minus + x_plus)
        e_0 = compute_ex(x_0, b0, g_tilde, delta, Rf)
        
        if abs(e_0) < tol or iter_count >= max_iter:
            break
        elif e_0 < 0:
            x_minus = x_0
        else:
            x_plus = x_0
        iter_count += 1
        
    return x_0

def calculate_metrics(x: float, Rf: float) -> tuple:
    """Calculate P/D ratio and equity premium."""
    if np.isnan(x):
        return np.nan, np.nan
    pd_ratio = 1.0 / (x - 1.0)
    E_Rm = x * np.exp(0.0202)
    equity_premium = E_Rm - Rf
    return pd_ratio, equity_premium

def plot_results(results_df: pd.DataFrame) -> None:
    """Plot P/D ratio and equity premium vs b0."""
    plt.figure()
    plt.plot(results_df['b0'], results_df['Price-Dividend Ratio (P/D)'])
    plt.xlabel('b0')
    plt.ylabel('Price-Dividend Ratio (P/D)')
    plt.title('Price-Dividend Ratio vs b0')
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(results_df['b0'], results_df['Equity Premium'])
    plt.xlabel('b0')
    plt.ylabel('Equity Premium')
    plt.title('Equity Premium vs b0')
    plt.grid(True)
    plt.show()

def main():
    # Parameters
    b0_values = np.arange(0, 10.1, 0.1)
    
    # Simulate consumption growth
    g_tilde = simulate_consumption_growth()
    
    # Initialize results lists
    pd_ratios = []
    equity_premiums = []
    equilibrium_x_values = []
    
    # Calculate results for each b0
    for b0 in b0_values:
        x = find_equilibrium_x(b0, g_tilde, DELTA, RF)
        pd_ratio, equity_premium = calculate_metrics(x, RF)
        
        pd_ratios.append(pd_ratio)
        equity_premiums.append(equity_premium)
        equilibrium_x_values.append(x)
        
        if np.isnan(x):
            print(f"No solution for b0 = {b0}")
    
    # Create and display results
    results_df = pd.DataFrame({
        'b0': b0_values,
        'Price-Dividend Ratio (P/D)': pd_ratios,
        'Equity Premium': equity_premiums,
        'Simulated Equilibrium x': equilibrium_x_values
    })
    
    plot_results(results_df)
    print(results_df)

if __name__ == "__main__":
    main()
