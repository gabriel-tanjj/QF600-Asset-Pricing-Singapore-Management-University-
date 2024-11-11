import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Constants
MARKET_FILE = "excel_data/Market_Portfolio.xlsx"
INDUSTRY_FILE = "excel_data/Industry_Portfolios.xlsx"
NUM_ASSETS = 10
NUM_SIMULATIONS = 100000
MU_START = 0
MU_END = 0.0010
MU_STEP = 0.000005
EPSILON = 1e-6
FIGURE_SIZE = (10, 6)
SCATTER_SIZE = 1
SCATTER_ALPHA = 0.5

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def load_and_preprocess_data():
    """Load and preprocess market and industry return data."""
    market_returns = pd.read_excel(MARKET_FILE, index_col='Date', parse_dates=True)
    industry_returns = pd.read_excel(INDUSTRY_FILE, index_col='Date', parse_dates=True)
    
    # Align dates and convert to decimal values
    industry_returns = industry_returns.loc[market_returns.index]
    market_returns['Market'] = market_returns['Market'] / 100.0
    industry_returns = industry_returns / 100.0
    
    return market_returns, industry_returns

def calculate_tracking_error_frontier(market_returns, industry_returns):
    """Calculate minimum tracking error frontier parameters."""
    deviations = industry_returns.subtract(market_returns['Market'], axis=0)
    
    # Calculate mean deviations and covariance matrix
    Ri = deviations.mean(axis=0)
    Ri_array = Ri.values.reshape(-1, 1)
    V = deviations.cov()
    V_array = V.values
    V_inv = np.linalg.inv(V_array)
    ones = np.ones((NUM_ASSETS, 1))
    
    # Compute scalar values
    delta = float(ones.T @ V_inv @ ones)
    alpha = float(ones.T @ V_inv @ Ri_array)
    zeta = float(Ri_array.T @ V_inv @ Ri_array)
    denom = delta * zeta - alpha ** 2
    
    return Ri_array, V_array, V_inv, ones, delta, alpha, zeta, denom

def generate_frontier_points(mu_values, Ri_array, V_array, V_inv, ones, delta, alpha, zeta, denom):
    """Generate points along the minimum tracking error frontier."""
    sigma_p = []
    expected_return_deviation = []
    
    for mu in mu_values:
        numerator = (zeta - alpha * mu) * ones + (delta * mu - alpha) * Ri_array
        w = V_inv @ numerator / denom
        E_delta_Rp = float(w.T @ Ri_array)
        sigma_squared = float(w.T @ V_array @ w)
        sigma = np.sqrt(sigma_squared)
        expected_return_deviation.append(E_delta_Rp)
        sigma_p.append(sigma)
        
    return [s * 100 for s in sigma_p], [e * 100 for e in expected_return_deviation]

def calculate_tangency_portfolio(V_inv, Ri_array, V_array, ones):
    """Calculate tangency portfolio weights and metrics."""
    w_tangency_numerator = V_inv @ Ri_array
    w_tangency_denominator = float(ones.T @ V_inv @ Ri_array)
    w_tangency = w_tangency_numerator / w_tangency_denominator
    
    E_delta_Rp_tangency = float(w_tangency.T @ Ri_array)
    sigma_tangency = np.sqrt(float(w_tangency.T @ V_array @ w_tangency))
    IR_tangency = E_delta_Rp_tangency / sigma_tangency
    
    return w_tangency, E_delta_Rp_tangency * 100, sigma_tangency * 100, IR_tangency

def plot_tracking_error_frontier(sigma_p, expected_return_deviation, E_delta_Rp_max, sigma_max_on_tangent):
    """Plot the minimum tracking error frontier with tangency line."""
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(sigma_p, expected_return_deviation, label='Minimum-Tracking-Error Frontier')
    plt.plot([0, sigma_max_on_tangent], [0, E_delta_Rp_max], 'r--', label='Tangency Line')
    plt.xlabel('Tracking Error')
    plt.ylabel('Expected Return Deviation')
    plt.ylim(0, E_delta_Rp_max * 1.1)
    plt.xlim(0, max(sigma_p) * 1.1)
    plt.yticks(np.arange(0, E_delta_Rp_max * 1.1, 0.005))
    plt.title('Minimum-Tracking-Error Frontier with Tangency Line')
    plt.grid(True)
    plt.legend()
    plt.show()

def print_tangency_weights(w_tangency, industry_names, IR_tangency):
    """Print tangency portfolio weights and information ratio."""
    w_tangency_percent = w_tangency.flatten() * 100
    print(f"Information Ratio of Tangency Portfolio: {IR_tangency:.6f}")
    print("Portfolio Weights for the Tangency Portfolio:")
    for industry, weight in zip(industry_names, w_tangency_percent):
        print(f"{industry}: {weight:.2f}%")

def monte_carlo_simulation(Ri, V, weight_method='uniform'):
    """Perform Monte Carlo simulation with specified weight generation method."""
    portfolio_returns = np.zeros(NUM_SIMULATIONS)
    portfolio_std_devs = np.zeros(NUM_SIMULATIONS)
    
    for i in range(NUM_SIMULATIONS):
        if weight_method == 'uniform':
            weights = np.random.uniform(0, 1, NUM_ASSETS)
        else:  # reciprocal
            u = np.random.uniform(EPSILON, 1, NUM_ASSETS)
            weights = 1 / u
            
        weights /= np.sum(weights)
        portfolio_returns[i] = np.dot(weights, Ri.flatten())
        portfolio_std_devs[i] = np.sqrt(weights @ V @ weights.T)
    
    return portfolio_std_devs * 100, portfolio_returns * 100

def plot_monte_carlo_frontier(std_devs, returns, weight_method):
    """Plot Monte Carlo simulation results."""
    plt.figure(figsize=FIGURE_SIZE)
    plt.scatter(std_devs, returns, s=SCATTER_SIZE, alpha=SCATTER_ALPHA, 
               color='blue' if weight_method == 'uniform' else 'green')
    plt.xlabel('Portfolio Standard Deviation [%]')
    plt.ylabel('Portfolio Mean Return [%]')
    plt.title(f'Minimum-Variance Frontier without Short Sales ({weight_method.title()} Weights)')
    plt.grid(True)
    plt.show()

def main():
    # Load and preprocess data
    market_returns, industry_returns = load_and_preprocess_data()
    
    # Calculate tracking error frontier
    Ri_array, V_array, V_inv, ones, delta, alpha, zeta, denom = calculate_tracking_error_frontier(
        market_returns, industry_returns)
    
    # Generate frontier points
    mu_values = np.arange(MU_START, MU_END, MU_STEP)
    sigma_p, expected_return_deviation = generate_frontier_points(
        mu_values, Ri_array, V_array, V_inv, ones, delta, alpha, zeta, denom)
    
    # Calculate tangency portfolio
    w_tangency, E_delta_Rp_tangency, sigma_tangency, IR_tangency = calculate_tangency_portfolio(
        V_inv, Ri_array, V_array, ones)
    
    # Calculate maximum values
    E_delta_Rp_max = max(expected_return_deviation)
    sigma_max_on_tangent = E_delta_Rp_max / IR_tangency
    
    # Plot frontier and print weights
    plot_tracking_error_frontier(sigma_p, expected_return_deviation, E_delta_Rp_max, sigma_max_on_tangent)
    print_tangency_weights(w_tangency, industry_returns.columns, IR_tangency)
    
    # Monte Carlo simulations
    Ri = industry_returns.mean().values.reshape(-1, 1)
    V = industry_returns.cov().values
    
    # Uniform weights simulation
    std_devs_uniform, returns_uniform = monte_carlo_simulation(Ri, V, 'uniform')
    plot_monte_carlo_frontier(std_devs_uniform, returns_uniform, 'uniform')
    
    # Reciprocal weights simulation
    std_devs_reciprocal, returns_reciprocal = monte_carlo_simulation(Ri, V, 'reciprocal')
    plot_monte_carlo_frontier(std_devs_reciprocal, returns_reciprocal, 'reciprocal')

if __name__ == "__main__":
    main()
