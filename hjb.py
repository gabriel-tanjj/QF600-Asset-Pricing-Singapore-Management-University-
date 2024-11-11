import numpy as np
import matplotlib.pyplot as plt

# Constants
N_SIMULATIONS = 10000
GAMMA_START = 1
GAMMA_END = 4.01
GAMMA_STEP = 0.001
JUMP_PROBABILITY = 0.017
JUMP_SIZE = 0.65
GROWTH_MEAN = 0.02
GROWTH_STD = 0.02
DISCOUNT_FACTOR = 0.99
VOLATILITY_THRESHOLD = 0.4
FIGURE_SIZE = (12, 8)
EXCLUDE_JUMPS = False

def simulate_growth_rates(n_simulations: int = N_SIMULATIONS) -> np.ndarray:
    """
    Simulate growth rates using normal and jump components.
    
    Args:
        n_simulations: Number of simulations to run
        
    Returns:
        Array of simulated growth rates
    """
    epsilon = np.random.randn(n_simulations)
    if not EXCLUDE_JUMPS:
        U = np.random.rand(n_simulations)
        nu = np.where(U < JUMP_PROBABILITY, np.log(JUMP_SIZE), 0)
    else:
        nu = 0
    ln_g = GROWTH_MEAN + GROWTH_STD * epsilon + nu
    return np.exp(ln_g)

def calculate_volatility_ratios(g: np.ndarray, gamma_values: np.ndarray) -> np.ndarray:
    """
    Calculate volatility ratios for different gamma values.
    
    Args:
        g: Array of growth rates
        gamma_values: Array of risk aversion parameters
        
    Returns:
        Array of volatility ratios
    """
    volatility_ratios = []
    for gamma in gamma_values:
        M = DISCOUNT_FACTOR * g ** (-gamma)
        volatility_ratio = np.std(M) / np.mean(M)
        volatility_ratios.append(volatility_ratio)
    return np.array(volatility_ratios)

def plot_volatility_ratios(gamma_values: np.ndarray, volatility_ratios: np.ndarray) -> None:
    """
    Plot volatility ratios against gamma values.
    
    Args:
        gamma_values: Array of risk aversion parameters
        volatility_ratios: Array of volatility ratios
    """
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(gamma_values, volatility_ratios)
    plt.xlabel('Risk Aversion Parameter (γ)')
    plt.ylabel('Volatility Ratio (σM / μM)')
    plt.title('Volatility Ratio of Pricing Kernel vs Risk Aversion Parameter')
    plt.axhline(y=VOLATILITY_THRESHOLD, color='red', linestyle='--')
    plt.grid(True)
    plt.show()

def find_threshold_gamma(gamma_values: np.ndarray, volatility_ratios: np.ndarray, 
                        threshold: float = VOLATILITY_THRESHOLD) -> float:
    """
    Find smallest gamma value where volatility ratio exceeds threshold.
    
    Args:
        gamma_values: Array of risk aversion parameters
        volatility_ratios: Array of volatility ratios
        threshold: Volatility ratio threshold
        
    Returns:
        Smallest gamma value exceeding threshold
    """
    for gamma, vr in zip(gamma_values, volatility_ratios):
        if vr > threshold:
            return gamma
    return None

def main():
    gamma_values = np.arange(GAMMA_START, GAMMA_END, GAMMA_STEP)
    
    g = simulate_growth_rates(N_SIMULATIONS)
    volatility_ratios = calculate_volatility_ratios(g, gamma_values)
    
    plot_volatility_ratios(gamma_values, volatility_ratios)
    
    threshold_gamma = find_threshold_gamma(gamma_values, volatility_ratios)
    print(f"The smallest value of γ where σM/μM > {VOLATILITY_THRESHOLD} is {threshold_gamma:.3f}")

if __name__ == "__main__":
    main()
