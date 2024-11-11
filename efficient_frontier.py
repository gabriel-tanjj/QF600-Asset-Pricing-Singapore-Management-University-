import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Constants
RISK_FREE_RATE = 0.13
Y_START = 0.13  # Starting point for return range
Y_END = 2.0    # Ending point for return range 
NUM_ITERATIONS = 200  # Number of points to generate

def calculate_greeks(mean, covm_inv, e):
    alpha = np.dot(mean.T, np.dot(covm_inv, e))
    zeta = np.dot(mean.T, np.dot(covm_inv, mean))
    delta = np.dot(e.T, np.dot(covm_inv, e))
    return alpha, zeta, delta

def calculate_weights(alpha, zeta, delta, mean, covm_inv, e):
    term1 = ((delta * mean - alpha) / (zeta * delta - alpha ** 2)) * np.dot(covm_inv, mean)
    term2 = ((zeta - alpha * mean) / (zeta * delta - alpha ** 2)) * np.dot(covm_inv, e)
    weights = term1 + term2
    weights /= np.sum(weights)  # Normalize weights
    return term1, term2, weights

def calculate_portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def calculate_portfolio_return(weights, returns):
    return np.dot(weights.T, returns)

def generate_mvf(df, ystart, yend, iter, rf):
    mean_ret = df.mean()
    covm = df.cov()
    e = np.ones(mean_ret.shape)
    covm_inv = np.linalg.inv(covm)

    alpha, zeta, delta = calculate_greeks(mean_ret, covm_inv, e)
    ret_mv = alpha / delta

    ret1 = np.linspace(start=0, stop=yend, num=iter)
    var1 = (1 / delta) + (delta / (zeta * delta - alpha ** 2)) * (ret1 - ret_mv) ** 2
    sigma1 = np.sqrt(var1)

    mvf1 = pd.DataFrame(ret1, columns=["rets"])
    mvf1["sigma1"] = sigma1

    ret2 = np.linspace(start=ystart, stop=yend, num=iter)
    sigma2 = np.sqrt((ret2 - rf) ** 2 / (zeta - 2 * alpha * rf + delta * rf ** 2))

    mvf2 = pd.DataFrame(ret2, columns=['rets_wra'])
    mvf2['sigma2'] = sigma2

    ret_tg = ret_mv - ((zeta * delta - alpha ** 2) / (delta ** 2 * (rf - ret_mv)))
    a = (zeta * covm_inv @ e - alpha * covm_inv @ mean_ret) / (zeta * delta - alpha ** 2)
    b = (delta * covm_inv @ mean_ret - alpha * covm_inv @ e) / (zeta * delta - alpha ** 2)
    weight_tg = a + b * ret_tg
    rp_tg = ret_tg - rf
    sigma_tg = -(zeta - 2 * alpha * rf + delta * rf ** 2) ** (1 / 2) / (delta * (rf - ret_mv))
    sharpe_tg = rp_tg / sigma_tg

    var_vector = [covm.iloc[i, i] for i in range(len(mean_ret.index))]
    table = pd.DataFrame(mean_ret, index=mean_ret.index, columns=['Mean Returns'])
    table['SD of Returns'] = np.sqrt(var_vector)
    table['Weight Tangency'] = weight_tg
    table = table.round(2)

    print('Risk premium for tangency portfolio: ', round(rp_tg, 2))
    print('Sharpe ratio for tangency portfolio: ', round(sharpe_tg, 2))

    # Calculate minimum variance point
    min_var_sigma = np.sqrt(1/delta)  # Standard deviation at minimum variance
    min_var_return = ret_mv  # Return at minimum variance
    
    # Print minimum variance frontier values
    print('\nMinimum Variance Portfolio:')
    print(f'Standard Deviation: {min_var_sigma:.4f}')
    print(f'Expected Return: {min_var_return:.4f}')

    # Calculate weights for minimum variance portfolio
    min_var_weights = covm_inv @ e / (e.T @ covm_inv @ e)
    print('\nMinimum Variance Portfolio Weights:')
    for asset_name, weight in zip(mean_ret.index, min_var_weights):
        print(f'{asset_name}: {weight:.4f}')
    print(f'Total weights: {np.sum(min_var_weights):.4f}')

    # Plotting the minimum variance frontier
    plt.figure(figsize=(10, 6))
    plt.plot(mvf1['sigma1'], mvf1['rets'], label='Minimum Variance Frontier', color='blue')
    plt.plot(mvf2['sigma2'], mvf2['rets_wra'], label='Weighted Return Frontier', color='orange')
    plt.axvline(x=min_var_sigma, color='red', linestyle='--', label='Minimum Variance Point')
    plt.plot(min_var_sigma, min_var_return, 'ro', label='Minimum Variance Portfolio')
    plt.title('Minimum Variance Frontier')
    plt.xlabel('Standard Deviation (Risk)')
    plt.ylabel('Expected Return')
    plt.legend()
    plt.grid()
    plt.show()

    return mvf1, mvf2, table, rp_tg, ret_tg, sigma_tg

if __name__ == "__main__":
    # Example usage to generate outputs
    df = pd.read_excel("excel_data/Industry_Portfolios.xlsx").drop(columns="Date")

    # Generate outputs using constants
    mvf1, mvf2, table, rp_tg, ret_tg, sigma_tg = generate_mvf(df, ystart=Y_START, yend=Y_END, iter=NUM_ITERATIONS, rf=RISK_FREE_RATE)

    # Print the results
    print(table)
    print(f"Sum of portfolio weights: {np.sum(table['Weight Tangency'])}")
    print(f"Risk Premium (Tangency): {round(rp_tg, 2)}")
    print(f"Return (Tangency): {round(ret_tg, 2)}")
    print(f"Standard Deviation (Tangency): {round(sigma_tg, 2)}")
