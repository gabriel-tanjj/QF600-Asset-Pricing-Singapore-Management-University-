import numpy as np

# Constants
S = 6        # Initial stock price
u = 10 / S   # Upward movement factor
d = 5 / S    # Downward movement factor
K = 6        # Strike price
Rf = 1.05    # One plus risk-free rate

def calculate_state_prices(S, u, d, Rf):
    """Calculate state prices and display matrices."""
    # Create payoff matrix X showing final values of assets in each state
    X = np.array([[u * S, 1], [d * S, 1]])
    P = np.array([S, 1 / Rf])  # Vector of current asset prices
    
    # Display the payoff matrix
    print("\nFinal Payoffs Matrix X:")
    print("     Stock  Bond")
    print(f"Up   {X[0,0]:.2f}   {X[0,1]:.2f}")
    print(f"Down {X[1,0]:.2f}   {X[1,1]:.2f}")

    # Display current prices
    print("\nInitial Prices Vector P:")
    print(f"Stock: {P[0]:.2f}")
    print(f"Bond:  {P[1]:.2f}")
    
    # Calculate state prices
    X_inv = np.linalg.inv(X)
    state_prices = P @ X_inv
    
    # Display the inverse matrix
    print("\nInverse of Payoffs Matrix X:")
    print("      Stock    Bond")
    print(f"Up   {X_inv[0,0]:.4f}  {X_inv[0,1]:.4f}")
    print(f"Down {X_inv[1,0]:.4f}  {X_inv[1,1]:.4f}")
    
    return state_prices

def price_call_option(S, K, u, d, pu, pd):
    """Calculate call option price and display payoffs."""
    payoff_up = max(u * S - K, 0)
    payoff_down = max(d * S - K, 0)
    
    print("\nCall Option Payoffs:")
    print(f"Up state:   {payoff_up:.4f}")
    print(f"Down state: {payoff_down:.4f}")
    
    return pu * payoff_up + pd * payoff_down

def price_put_option(S, K, u, d, pu, pd):
    """Calculate put option price and display payoffs."""
    payoff_up = max(K - u * S, 0)
    payoff_down = max(K - d * S, 0)
    
    print("\nPut Option Payoffs:")
    print(f"Up state:   {payoff_up:.4f}")
    print(f"Down state: {payoff_down:.4f}")
    
    return pu * payoff_up + pd * payoff_down

def main():
    # Calculate state prices
    pu, pd = calculate_state_prices(S, u, d, Rf)

    # Display state prices
    print("\nState Prices [p_u, p_d]:")
    print(f"p_u: {pu:.4f}")
    print(f"p_d: {pd:.4f}")

    # Calculate risk-neutral probabilities
    pi_u = Rf * pu
    pi_d = Rf * pd
    print("\nRisk-neutral Probabilities [π_u, π_d]:")
    print(f"π_u: {pi_u:.4f}")
    print(f"π_d: {pi_d:.4f}")

    # Calculate option prices
    call_price = price_call_option(S, K, u, d, pu, pd)
    put_price = price_put_option(S, K, u, d, pu, pd)

    print("\nOption Prices:")
    print(f"Call: {call_price:.4f}")
    print(f"Put:  {put_price:.4f}")

if __name__ == "__main__":
    main()
