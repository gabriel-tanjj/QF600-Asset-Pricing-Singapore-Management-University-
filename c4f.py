import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Constants from ff3.py
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Read data files
rf = pd.read_excel("excel_data/Risk_Factors.xlsx").drop(columns="Date") 
industry_p = pd.read_excel("excel_data/Industry_Portfolios.xlsx").drop(columns="Date")
market_p = pd.read_excel("excel_data/Market_Portfolio.xlsx").drop(columns="Date")

def carhart_four_factor(data_portfolio: pd.DataFrame, data_market: pd.DataFrame, risk_factors: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Carhart Four-Factor model coefficients for each portfolio.
    Extends the Fama-French Three-Factor model by adding momentum factor.
    
    Parameters:
    data_portfolio (pd.DataFrame): Portfolio returns
    data_market (pd.DataFrame): Market returns  
    risk_factors (pd.DataFrame): Risk factors including 'Rm-Rf', 'SMB', 'HML', 'UMD', 'Rf'
    
    Returns:
    pd.DataFrame: DataFrame containing Alpha, Beta (Rm-Rf), Beta (SMB), Beta (HML), Beta (UMD) and R-squared for each portfolio
    """
    # Make a copy of the portfolio data and align the index with risk factors
    data_portfolio = data_portfolio.copy()
    data_portfolio.index = risk_factors.index

    # Extract the four factors: Market excess return (Rm-Rf), SMB (size), HML (value), UMD (momentum)
    data_factors = risk_factors[['Rm-Rf', 'SMB', 'HML', 'UMD']].values
    
    # Calculate excess returns by subtracting risk-free rate
    excess_returns = data_portfolio.subtract(risk_factors['Rf'], axis=0)

    # Fit linear regression model for each portfolio against the four factors
    models = [LinearRegression().fit(data_factors, excess_returns[col]) for col in excess_returns.columns]
    
    # Create DataFrame to store regression coefficients and R-squared values
    c4f_table = pd.DataFrame({
        'Alpha': [model.intercept_ for model in models],
        'Beta (Rm-Rf)': [model.coef_[0] for model in models],
        'Beta (SMB)': [model.coef_[1] for model in models], 
        'Beta (HML)': [model.coef_[2] for model in models],
        'Beta (UMD)': [model.coef_[3] for model in models],
        'R-squared': [model.score(data_factors, excess_returns[col]) for model, col in zip(models, excess_returns.columns)]
    }, index=excess_returns.columns)

    return c4f_table

def calculate_performance_metrics(excess_returns: pd.DataFrame, risk_factors: pd.DataFrame, capm_betas: pd.Series, c4f_alphas: pd.Series) -> pd.DataFrame:
    """
    Calculate performance metrics for each portfolio.
    
    Parameters:
    excess_returns (pd.DataFrame): Excess returns of portfolios
    risk_factors (pd.DataFrame): Risk factors including 'Rm-Rf'
    capm_betas (pd.Series): CAPM beta values
    c4f_alphas (pd.Series): Carhart four-factor alpha values
    
    Returns:
    pd.DataFrame: DataFrame containing Sharpe Ratio, Sortino Ratio, Treynor Ratio, Jensen's Alpha, and C4F Alpha
    """
    # Calculate mean excess return and standard deviation
    mean_excess_returns = excess_returns.mean()
    std_devs = excess_returns.std(ddof=1)
    
    # Calculate downside risk (negative returns only)
    downside_returns = excess_returns.where(excess_returns < 0)
    semi_variances = downside_returns.pow(2).mean()
    semi_std_devs = np.sqrt(semi_variances)
    
    # Compute performance ratios
    sharpe_ratios = mean_excess_returns / std_devs
    sortino_ratios = mean_excess_returns / semi_std_devs
    treynor_ratios = mean_excess_returns / capm_betas
    
    # Calculate Jensen's Alpha using CAPM
    jensens_alphas = {}
    for industry in excess_returns.columns:
        Y = excess_returns[industry].dropna()
        X = sm.add_constant(risk_factors['Rm-Rf'].loc[Y.index])
        capm_model = sm.OLS(Y, X).fit()
        jensens_alphas[industry] = capm_model.params['const']
    
    # Compile metrics into DataFrame
    performance_metrics = pd.DataFrame({
        'Sharpe Ratio': sharpe_ratios,
        'Sortino Ratio': sortino_ratios,
        'Treynor Ratio': treynor_ratios,
        "Jensen's Alpha": pd.Series(jensens_alphas),
        "C4F Alpha": c4f_alphas
    })

    return performance_metrics.astype(float)

def plot_performance_metrics(performance_metrics: pd.DataFrame, metrics_to_plot: list = None):
    """
    Plot specified performance metrics as bar charts.
    
    Parameters:
    performance_metrics (pd.DataFrame): DataFrame containing performance metrics
    metrics_to_plot (list, optional): List of metric names to plot
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['Sharpe Ratio', 'Sortino Ratio', 'Treynor Ratio']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        performance_metrics[metric].plot(kind='bar', color="skyblue", edgecolor='black')
        plt.title(f'{metric} for Industry Portfolios')
        plt.ylabel(metric)
        plt.xlabel('Industry Portfolios')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

def main():
    # Calculate Carhart Four-Factor model coefficients
    c4f_table = carhart_four_factor(data_portfolio=industry_p, data_market=market_p, risk_factors=rf)
    
    # Calculate CAPM alpha and beta
    industry_excess_returns = industry_p.subtract(rf['Rf'], axis=0)
    market_excess_returns = market_p.subtract(rf['Rf'], axis=0)
    
    # Calculate CAPM betas
    capm_betas = {}
    for industry in industry_excess_returns.columns:
        Y = industry_excess_returns[industry].dropna()
        X = sm.add_constant(market_excess_returns.loc[Y.index])
        capm_model = sm.OLS(Y, X).fit()
        capm_betas[industry] = capm_model.params['Market']

    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(
        excess_returns=industry_excess_returns,
        risk_factors=rf,
        capm_betas=pd.Series(capm_betas),
        c4f_alphas=c4f_table["Alpha"]
    )

    # Display results
    print("\nCarhart Four-Factor Model Coefficients:")
    print(c4f_table)
    print("\nPerformance Metrics:")
    print(performance_metrics)

    # Plot performance metrics
    metrics_to_plot = ['Sharpe Ratio', 'Sortino Ratio', 'Treynor Ratio']
    plot_performance_metrics(performance_metrics, metrics_to_plot)

if __name__ == "__main__":
    main()
