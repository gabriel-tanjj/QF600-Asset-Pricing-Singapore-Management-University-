import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings

# Constants
MARKET_FILE = "excel_data/Market_Portfolio.xlsx"
INDUSTRY_FILE = "excel_data/Industry_Portfolios.xlsx"
RISK_FREE_RATE = 0.13
FIGURE_SIZE = (10, 6)

# Load market and industry portfolio data from Excel files, dropping the 'Date' column
marketdf = pd.read_excel(MARKET_FILE).drop(columns="Date")
industrydf = pd.read_excel(INDUSTRY_FILE).drop(columns="Date")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def calculate_coskewness(portfolio_returns, market_returns):
    """
    Calculate the coskewness between portfolio returns and market returns.
    
    Parameters:
    portfolio_returns: Series containing portfolio returns
    market_returns: Series containing market returns
    
    Returns:
    float: Coskewness measure
    """
    # Standardize returns
    p_std = (portfolio_returns - portfolio_returns.mean()) / portfolio_returns.std()
    m_std = (market_returns - market_returns.mean()) / market_returns.std()
    
    # Calculate coskewness
    coskewness = np.mean(p_std * m_std**2)
    return coskewness

def ThreeMomentCAPM(data_portfolio, data_market):
    """
    Calculate Alpha, Beta and Gamma (coskewness) for each portfolio using Three-Moment CAPM.
    
    Parameters:
    data_portfolio: DataFrame containing portfolio returns
    data_market: DataFrame containing market returns
    
    Returns:
    DataFrame with Alpha, Beta and Gamma for each portfolio
    """
    # Align market data with portfolio data by index
    data_market = data_market.reindex(data_portfolio.index)
    
    # Calculate excess returns
    excess_market_returns = data_market.squeeze() - RISK_FREE_RATE
    excess_portfolio_returns = data_portfolio - RISK_FREE_RATE
    
    # Initialize results DataFrame
    results = pd.DataFrame(index=excess_portfolio_returns.columns, 
                         columns=['Alpha', 'Beta', 'Gamma'])
    
    # Calculate for each portfolio
    for portfolio in results.index:
        y = excess_portfolio_returns[portfolio]
        x = excess_market_returns
        df = pd.concat([x, y], axis=1).dropna()
        
        # Calculate Beta using linear regression
        model = LinearRegression().fit(df.iloc[:, 0].values.reshape(-1, 1), df.iloc[:, 1])
        alpha, beta = model.intercept_, model.coef_[0]
        
        # Calculate Gamma (coskewness)
        gamma = calculate_coskewness(df.iloc[:, 1], df.iloc[:, 0])
        
        results.loc[portfolio] = [alpha, beta, gamma]
    
    # Set market values
    results.loc['Market'] = [0.0, 1.0, 1.0]
    return results.round(6)

def plot_3M_SML(beta_range, gamma_range, sml_surface, capm_data):
    """
    Plot the Three-Moment Security Market Line Surface.
    
    Parameters:
    beta_range: array of Beta values
    gamma_range: array of Gamma values
    sml_surface: 2D array of expected returns
    capm_data: DataFrame containing portfolio data
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create mesh grid for surface plot
    beta_mesh, gamma_mesh = np.meshgrid(beta_range, gamma_range)
    
    # Plot the surface
    surf = ax.plot_surface(beta_mesh, gamma_mesh, sml_surface, 
                          cmap='viridis', alpha=0.6)
    
    # Plot the actual portfolio points
    ax.scatter(capm_data['Beta'], capm_data['Gamma'], capm_data['Mean_Return'],
              color='red', s=50, label='Portfolios')
    
    ax.set_xlabel('Beta')
    ax.set_ylabel('Gamma')
    ax.set_zlabel('Expected Return')
    ax.set_title('Three-Moment CAPM Security Market Surface')
    
    plt.colorbar(surf)
    plt.legend()
    plt.show()

def main():
    # Calculate Three-Moment CAPM results
    three_m_results = ThreeMomentCAPM(industrydf, marketdf)
    print("\nThree-Moment CAPM Results:")
    print(three_m_results)
    
    # Generate data for 3D plot
    beta_range = np.linspace(0, 2, 50)
    gamma_range = np.linspace(-1, 1, 50)
    sml_surface = np.zeros((50, 50))
    
    # Calculate expected returns for surface
    for i, beta in enumerate(beta_range):
        for j, gamma in enumerate(gamma_range):
            sml_surface[j,i] = RISK_FREE_RATE + beta * (marketdf.mean().values[0] - RISK_FREE_RATE) + gamma * 0.1
            
    # Add mean returns to results for plotting
    three_m_results['Mean_Return'] = industrydf.mean()
    three_m_results.loc['Market', 'Mean_Return'] = marketdf.mean().values[0]
    
    # Plot the 3D surface
    plot_3M_SML(beta_range, gamma_range, sml_surface, three_m_results)

if __name__ == "__main__":
    main()
