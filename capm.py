import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings

# Constants
MARKET_FILE = "excel_data/Market_Portfolio.xlsx"
INDUSTRY_FILE = "excel_data/Industry_Portfolios.xlsx"
RISK_FREE_RATE = 0.13
BETA_RANGE_START = 0
BETA_RANGE_END = 2
BETA_RANGE_POINTS = 100
FIGURE_SIZE = (10, 6)
MARKET_POINT_SIZE = 100

def load_data():
    """Load market and industry portfolio data from Excel files."""
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
    marketdf = pd.read_excel(MARKET_FILE).drop(columns="Date")
    industrydf = pd.read_excel(INDUSTRY_FILE).drop(columns="Date") 
    return marketdf, industrydf

def MarketModel(data_portfolio, data_market):
    """
    Calculate the Alpha and Beta for each portfolio using the Market Model.
    
    Parameters:
    data_portfolio: DataFrame containing portfolio returns
    data_market: DataFrame containing market returns
    
    Returns:
    DataFrame with Alpha and Beta for each portfolio
    """
    # Align market data with portfolio data by index
    data_market = data_market.reindex(data_portfolio.index)
    
    # Calculate excess returns for market and portfolios
    excess_market_returns = data_market.squeeze() - RISK_FREE_RATE 
    excess_portfolio_returns = data_portfolio - RISK_FREE_RATE   
    
    # Initialize results DataFrame to store Alpha and Beta
    results = pd.DataFrame(index=excess_portfolio_returns.columns, columns=['Alpha', 'Beta'])
    
    # Loop through each portfolio to calculate Alpha and Beta
    for portfolio in results.index:
        y = excess_portfolio_returns[portfolio]  # Portfolio excess returns
        x = excess_market_returns  # Market excess returns
        df = pd.concat([x, y], axis=1).dropna()  # Combine and drop NaN values
        model = LinearRegression().fit(df.iloc[:, 0].values.reshape(-1, 1), df.iloc[:, 1])  # Fit linear regression
        results.loc[portfolio] = [model.intercept_, model.coef_[0]]  # Store Alpha and Beta
    
    # Set the Alpha and Beta for the market
    results.loc['Market'] = [0.0, 1.0]
    return results.round(6)  # Return results rounded to 6 decimal places

def plot_SML(beta_range, sml_line, capm_data):
    """
    Plot the Security Market Line (SML).
    
    Parameters:
    beta_range: array, range of Beta values
    sml_line: array, corresponding SML values
    capm_data: DataFrame containing CAPM data for portfolios
    """
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(beta_range, sml_line, label='Security Market Line (SML)', color='red')  # Plot SML
    plt.scatter(capm_data['Beta'], capm_data['Mean_Return'], color='blue', label='Portfolios and Market')  # Plot portfolios
    plt.xlabel('Beta')
    plt.ylabel('Mean Monthly Return (%)')
    plt.title('Security Market Line (SML)')
    plt.legend()
    plt.grid(True)
    plt.show()

def CAPM_SML(data_portfolio, data_market, market_model_results):
    """
    Calculate the CAPM Security Market Line (SML) and plot it.
    
    Parameters:
    data_portfolio: DataFrame containing portfolio returns
    data_market: DataFrame containing market returns
    market_model_results: DataFrame containing Alpha and Beta from the Market Model
    
    Returns:
    DataFrame with CAPM data including Mean Returns and Betas
    """
    # Calculate mean returns for portfolios and market
    mean_returns_portfolio = data_portfolio.mean()
    mean_return_market = data_market.mean().iloc[0]

    # Create a DataFrame for CAPM data
    capm_data = pd.DataFrame({
        'Mean_Return': mean_returns_portfolio,
        'Beta': market_model_results.loc[data_portfolio.columns, 'Beta']
    })
    
    # Create a DataFrame for the market row and concatenate it with capm_data
    market_row = pd.DataFrame({'Mean_Return': [mean_return_market], 'Beta': [1]}, index=['Market'])
    capm_data = pd.concat([capm_data, market_row])

    # Prepare data for regression
    X = capm_data['Beta'].values.reshape(-1, 1)  # Reshape Beta for regression
    Y = capm_data['Mean_Return'].values  # Mean Returns for regression
    capm_model = LinearRegression().fit(X, Y)  # Fit linear regression model

    # Print CAPM regression results
    print(f"CAPM Regression Results:")
    print(f"Alpha (Risk-Free Rate Estimate): {capm_model.intercept_:.6f}%")
    print(f"Beta (Market Risk Premium Estimate): {capm_model.coef_[0]:.6f}")

    # Plotting the SML
    beta_range = np.linspace(BETA_RANGE_START, BETA_RANGE_END, BETA_RANGE_POINTS)  # Generate a range of Beta values
    sml_line = capm_model.intercept_ + capm_model.coef_[0] * beta_range  # Calculate SML line

    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(beta_range, sml_line, label='Security Market Line', color='blue')  # Plot SML
    plt.scatter(capm_data['Beta'], capm_data['Mean_Return'], color='green', label='Portfolios')  # Plot portfolios
    plt.scatter(1, mean_return_market, color='red', label='Market Portfolio', s=MARKET_POINT_SIZE)  # Highlight market portfolio
    plt.title('Security Market Line (SML)')
    plt.xlabel('Beta')
    plt.ylabel('Mean Return')
    plt.legend()
    plt.grid(True)
    plt.show()

    return capm_data  # Return CAPM data

def main():
    # Load data
    marketdf, industrydf = load_data()
    
    # Calculate the Market Model results
    mm = MarketModel(industrydf, marketdf)
    mm['Alpha'] = mm['Alpha'].apply(lambda x: '{:.6f}'.format(x))  # Format Alpha values
    print(mm)  # Print Market Model results

    # Calculate and print CAPM SML results
    capm_sml = CAPM_SML(industrydf, marketdf, mm)
    print(capm_sml)  # Print CAPM SML results

if __name__ == "__main__":
    main()
