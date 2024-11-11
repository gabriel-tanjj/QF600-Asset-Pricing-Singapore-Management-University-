import numpy as np
import matplotlib.pyplot as plt

def black_litterman(P, Q, Omega, tau, sigma, pi):
    """
    Calculate Black-Litterman returns and covariance matrix.
    
    Parameters:
    P: Matrix of views
    Q: Expected returns for views
    Omega: Uncertainty in views
    tau: Investor's confidence level
    sigma: Covariance matrix of asset returns
    pi: Market equilibrium excess returns
    
    Returns:
    adjusted_pi: Black-Litterman expected returns
    M: Black-Litterman covariance matrix
    """
    # Step 1: Compute the posterior mean (Black-Litterman returns)
    middle_term = np.linalg.inv(np.dot(np.dot(P, tau * sigma), P.T) + Omega)
    adjusted_pi = pi + np.dot(np.dot(tau * sigma, P.T), np.dot(middle_term, (Q - np.dot(P, pi))))
    
    # Step 2: Compute the posterior covariance
    M = np.linalg.inv(np.linalg.inv(tau * sigma) + np.dot(np.dot(P.T, np.linalg.inv(Omega)), P))
    return adjusted_pi, M

def plot_returns_comparison(original_returns, bl_returns, asset_names=None):
    """
    Plot comparison of original vs Black-Litterman returns.
    
    Parameters:
    original_returns: Original market equilibrium returns
    bl_returns: Black-Litterman adjusted returns
    asset_names: List of asset names (optional)
    """
    if asset_names is None:
        asset_names = [f'Asset {i+1}' for i in range(len(original_returns))]
        
    x = np.arange(len(original_returns))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, original_returns, width, label='Original Returns')
    ax.bar(x + width/2, bl_returns, width, label='Black-Litterman Returns')
    
    ax.set_ylabel('Expected Returns')
    ax.set_title('Original vs Black-Litterman Returns')
    ax.set_xticks(x)
    ax.set_xticklabels(asset_names)
    ax.legend()
    
    plt.grid(True, alpha=0.3)
    plt.show()

def get_example_data():
    """Generate example data for Black-Litterman model."""
    n_assets = 3
    # Market equilibrium excess returns
    pi = np.array([0.05, 0.03, 0.04])
    # Covariance matrix of asset returns
    sigma = np.array([[0.1, 0.05, 0.02],
                     [0.05, 0.08, 0.03],
                     [0.02, 0.03, 0.06]])
    # Investor's confidence level
    tau = 0.025
    # Matrix of views (2 views here)
    P = np.array([[1, 0, 0],  # View 1 on Asset 1
                  [0, 1, -1]])  # View 2 on Assets 2 vs. 3
    # Expected returns for the views
    Q = np.array([0.05, 0.01])
    # Uncertainty in the views
    Omega = np.diag([0.0001, 0.0001])
    
    return P, Q, Omega, tau, sigma, pi

def main():
    # Get example data
    P, Q, Omega, tau, sigma, pi = get_example_data()
    
    # Calculate Black-Litterman results
    bl_returns, bl_covariance = black_litterman(P, Q, Omega, tau, sigma, pi)
    
    # Print results
    print("Adjusted Expected Returns (Black-Litterman):", bl_returns)
    print("\nAdjusted Covariance Matrix (Black-Litterman):\n", bl_covariance)
    
    # Plot comparison
    plot_returns_comparison(pi, bl_returns)

if __name__ == "__main__":
    main()
