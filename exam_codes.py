import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm

def clean_data(file_path: str) -> pd.DataFrame:
    """
    Cleans financial data from an Excel file by performing the following steps:
    1. Loads data from the specified Excel file.
    2. Identifies and drops any column containing 'Date' in its name.
    3. Drops rows with any missing values.
    4. Calculates and prints the mean and standard deviation for each numerical column.
    5. (Optional) Returns additional statistics or plots as needed.

    Parameters:
    - file_path (str): The path to the Excel file containing the data.

    Returns:
    - pd.DataFrame: The cleaned DataFrame with 'Date' columns removed and no missing values.

    Raises:
    - FileNotFoundError: If the Excel file does not exist at the provided path.
    - ValueError: If the file cannot be parsed as an Excel file.
    """
    try:
        # Load the Excel file into a DataFrame
        df = pd.read_excel(file_path)
        print(f"Successfully loaded data from {file_path}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        raise
    except ValueError as ve:
        print(f"Error: {ve}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading the file: {e}")
        raise

    # Identify columns that contain 'Date' (case-insensitive)
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        df.drop(columns=date_cols, inplace=True)
        print(f"Dropped Date column(s): {date_cols}")
    else:
        print("No 'Date' column found to drop.")

    # Display initial shape
    print(f"Data shape before cleaning: {df.shape}")

    # Handle missing values by dropping rows with any NaNs
    df_before_drop = df.shape[0]
    df.dropna(inplace=True)
    df_after_drop = df.shape[0]
    rows_dropped = df_before_drop - df_after_drop
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} row(s) containing missing values.")
    else:
        print("No missing values found.")

    # Display final shape
    print(f"Data shape after cleaning: {df.shape}")

    # Calculate and display basic statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        means = df[numerical_cols].mean()
        std_devs = df[numerical_cols].std()
        print("\nMean of numerical columns:")
        print(means)
        print("\nStandard Deviation of numerical columns:")
        print(std_devs)
    else:
        print("No numerical columns found to calculate statistics.")

    return df

def load_industry_portfolios(file_path: str) -> pd.DataFrame:
    """
    Loads industry portfolio data from an Excel file.

    Parameters:
    - file_path (str): The path to the Industry_Portfolios.xlsx file.

    Returns:
    - pd.DataFrame: DataFrame containing industry portfolio data with 'Date' column dropped.
    
    Raises:
    - FileNotFoundError: If the specified file path does not exist.
    """
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print("check path")
        raise

    df.drop("Date", axis=1, inplace=True)
    return df

def compute_financial_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes financial metrics such as Alpha, Beta, R-squared, and Correlation for each portfolio.

    Parameters:
    - df (pd.DataFrame): DataFrame containing portfolio returns with 'Date' as index.

    Returns:
    - pd.DataFrame: DataFrame with computed metrics for each portfolio.
    """
    
    R = df.mean().values.reshape(-1, 1)
    V = df.cov().values
    e = np.ones((df.shape[1], 1))
    
    alpha = R.T @ np.linalg.inv(V) @ e
    zeta = R.T @ np.linalg.inv(V) @ R
    delta = e.T @ np.linalg.inv(V) @ e
    
    metrics = pd.DataFrame({
        'Alpha': alpha.flatten(),
        'Zeta': zeta.flatten(),
        'Delta': delta.flatten()
    }, index=df.columns)
    
    return metrics

def MarketModel(data_portfolio: pd.DataFrame, rf: float, data_market: pd.Series) -> pd.DataFrame:
    """
    Applies the Market Model (CAPM) to compute Alpha and Beta for each portfolio.

    Parameters:
    - data_portfolio (pd.DataFrame): DataFrame containing portfolio returns.
    - rf (float): Risk-free rate.
    - data_market (pd.Series): Series containing market returns.

    Returns:
    - pd.DataFrame: DataFrame with Alpha and Beta for each portfolio, including 'Market'.
    """
    data_portfolio = data_portfolio.copy()
    data_market = data_market.copy()
    data_market = data_market.reindex(data_portfolio.index)
    excess_market_returns = data_market.squeeze() - rf 
    excess_portfolio_returns = data_portfolio - rf   
    
    alphas = []
    betas = []
    
    portfolios = excess_portfolio_returns.columns.tolist()
    for portfolio in portfolios:
        y = excess_portfolio_returns[portfolio]
        x = excess_market_returns.values.reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        alphas.append(model.intercept_)
        betas.append(model.coef_[0])
    
    results = pd.DataFrame({
        'Alpha': alphas,
        'Beta': betas
    }, index=portfolios)
    
    results.loc['Market'] = [0.0, 1.0]
    results = results.round(6)
    
    return results

def plot_SML(beta_range: np.ndarray, sml_line: np.ndarray, capm_data: pd.DataFrame) -> None:
    """
    Plots the Security Market Line (SML).

    Parameters:
    - beta_range (np.ndarray): Array of beta values.
    - sml_line (np.ndarray): Expected returns corresponding to the beta_range.
    - capm_data (pd.DataFrame): DataFrame containing portfolio Betas and expected returns.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(beta_range, sml_line, label='SML', color='red')
    plt.scatter(capm_data['Beta'], capm_data['Expected Return'], color='blue')
    plt.xlabel('Beta')
    plt.ylabel('Expected Return')
    plt.title('Security Market Line (SML)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def FF3Factor(data_portfolio: pd.DataFrame, data_market: pd.DataFrame, risk_factors: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the Fama-French 3-Factor Model to compute Alpha and Betas for each portfolio.

    Parameters:
    - data_portfolio (pd.DataFrame): DataFrame containing portfolio returns.
    - data_market (pd.DataFrame): DataFrame containing market returns.
    - risk_factors (pd.DataFrame): DataFrame containing 'Rm-Rf', 'SMB', and 'HML' factors.

    Returns:
    - pd.DataFrame: DataFrame with Alpha, Beta (Rm-Rf), Beta (SMB), Beta (HML), and R-squared for each portfolio.
    """
    data_portfolio = data_portfolio.copy()
    data_portfolio.index = risk_factors.index
    data_factors = risk_factors[['Rm-Rf', 'SMB', 'HML']]
    array_rf = risk_factors['Rf']
    y = data_portfolio.subtract(array_rf, axis=0)
    x = data_factors.values
    c = len(y.columns)
    models = []
    r_sq = []
    alphas = []
    betas_rm_rf = []
    betas_smb = []
    betas_hml = []

    for i in range(c):
        model = LinearRegression().fit(x, y.iloc[:, i])
        models.append(model)
        r_sq.append(model.score(x, y.iloc[:, i]))
        alphas.append(model.intercept_)
        betas_rm_rf.append(model.coef_[0])
        betas_smb.append(model.coef_[1])
        betas_hml.append(model.coef_[2])
        
    FF_table = pd.DataFrame({
        'Alpha': alphas,
        'Beta (Rm-Rf)': betas_rm_rf,
        'Beta (SMB)': betas_smb,
        'Beta (HML)': betas_hml,
        'R-squared': r_sq
    }, index=y.columns)

    return FF_table

def calculate_performance_metrics(
    excess_returns: pd.DataFrame, 
    risk_factors: pd.DataFrame, 
    capm_betas: pd.Series, 
    ff_alphas: pd.Series
) -> pd.DataFrame:
    """
    Calculates Sharpe Ratio, Sortino Ratio, Treynor Ratio, Jensen's Alpha, and Fama-French Alpha for each portfolio.

    Parameters:
    - excess_returns (pd.DataFrame): DataFrame of excess returns for each portfolio.
    - risk_factors (pd.DataFrame): DataFrame containing risk factors.
    - capm_betas (pd.Series): Series containing CAPM Betas for each portfolio.
    - ff_alphas (pd.Series): Series containing Fama-French Alphas for each portfolio.

    Returns:
    - pd.DataFrame: DataFrame with performance metrics for each portfolio.
    """
    performance_metrics = pd.DataFrame(columns=[
        'Sharpe Ratio', 'Sortino Ratio', 'Treynor Ratio',
        "Jensen's Alpha", "FF Three-Factor Alpha"
    ], index=excess_returns.columns)

    mean_excess_returns = excess_returns.mean()
    std_devs = excess_returns.std(ddof=1)
    semi_variances = {}
    for industry in excess_returns.columns:
        downside_returns = excess_returns[industry][excess_returns[industry] < 0]
        if len(downside_returns) > 0:
            semi_variance = (downside_returns ** 2).sum() / len(excess_returns)
            semi_variances[industry] = semi_variance
        else:
            semi_variances[industry] = np.nan

    semi_std_devs = {industry: np.sqrt(semi_variances[industry]) for industry in semi_variances}
    sharpe_ratios = mean_excess_returns / std_devs
    sortino_ratios = mean_excess_returns / pd.Series(semi_std_devs)
    treynor_ratios = mean_excess_returns / capm_betas
    jensens_alphas = {}
    for industry in excess_returns.columns:
        Y = excess_returns[industry]
        X = sm.add_constant(risk_factors['Rm-Rf'])
        capm_model = sm.OLS(Y, X).fit()
        jensens_alphas[industry] = capm_model.params['const']
    
    FF_three_factor_alphas = ff_alphas  # Assuming ff_alphas is provided

    performance_metrics['Sharpe Ratio'] = sharpe_ratios
    performance_metrics['Sortino Ratio'] = sortino_ratios
    performance_metrics['Treynor Ratio'] = treynor_ratios
    performance_metrics["Jensen's Alpha"] = pd.Series(jensens_alphas)
    performance_metrics["FF Three-Factor Alpha"] = FF_three_factor_alphas

    return performance_metrics

def simulate_portfolios(
    market_returns: pd.DataFrame, 
    industry_returns: pd.DataFrame, 
    num_simulations: int = 1000000
) -> None:
    """
    Simulates portfolio weights under uniform and reciprocal weighting schemes and plots the Minimum-Variance Frontier.

    Parameters:
    - market_returns (pd.DataFrame): DataFrame containing market returns with 'Market' column.
    - industry_returns (pd.DataFrame): DataFrame containing industry portfolio returns.
    - num_simulations (int): Number of portfolio simulations to run.

    Returns:
    - None
    """
    industry_returns = industry_returns.loc[market_returns.index]
    market_returns['Market'] = market_returns['Market'] / 100.0
    industry_returns = industry_returns / 100.0
    deviations = industry_returns.subtract(market_returns['Market'], axis=0)

    Ri = deviations.mean(axis=0)
    Ri_array = Ri.values.reshape(-1, 1)
    V = deviations.cov()
    V_array = V.values
    V_inv = np.linalg.inv(V_array)
    ones = np.ones((10, 1))

    # Compute scalar values A, B, C, D
    delta = float(ones.T @ V_inv @ ones) 
    alpha = float(ones.T @ V_inv @ Ri_array)
    zeta = float(Ri_array.T @ V_inv @ Ri_array)
    D = delta * zeta - alpha ** 2

    mu_values = np.arange(0, 0.0010, 0.000005)
    sigma_p = []
    expected_return_deviation = []

    for mu in mu_values:
        numerator = (zeta - alpha * mu) * ones + (delta * mu - alpha) * Ri_array
        w = V_inv @ numerator / D
        E_delta_Rp = float(w.T @ Ri_array)
        sigma_squared = float(w.T @ V_array @ w)
        sigma = np.sqrt(sigma_squared)
        expected_return_deviation.append(E_delta_Rp)
        sigma_p.append(sigma)

    sigma_p_percent = [sigma * 100 for sigma in sigma_p]
    expected_return_deviation_percent = [erd * 100 for erd in expected_return_deviation]

    plt.figure(figsize=(10, 6))
    plt.plot(sigma_p_percent, expected_return_deviation_percent, label='Minimum-Variance Frontier')
    plt.xlabel('Portfolio Standard Deviation [%]')
    plt.ylabel('Portfolio Mean Return [%]')
    plt.title('Minimum-Variance Frontier')
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_volatility_ratio() -> None:
    """
    Simulates pricing kernels for varying risk aversion parameters (gamma),
    computes the volatility ratio (sigma_M / mu_M), plots the ratio, and identifies
    the smallest gamma where the ratio exceeds 0.4.

    Returns:
    - None
    """
    N = 10000  # Number of simulations
    # Simulate epsilon ~ N(0,1)
    epsilon = np.random.randn(N)
    # Simulate nu with given probabilities
    U = np.random.rand(N)
    nu = np.where(U < 0.017, np.log(0.65), 0)
    ln_g = 0.02 + 0.02 * epsilon + nu
    g = np.exp(ln_g)

    gamma_values = np.arange(1, 4.01, 0.001)
    volatility_ratios = []

    for gamma in gamma_values:
        M = 0.99 * g ** (-gamma)
        mu_M = np.mean(M)
        sigma_M = np.std(M)
        volatility_ratio = sigma_M / mu_M
        volatility_ratios.append(volatility_ratio)

    # Plot the volatility ratio against gamma
    plt.figure(figsize=(12, 8))
    plt.plot(gamma_values, volatility_ratios, label='Volatility Ratio (σM / μM)')
    plt.xlabel('Risk Aversion Parameter (γ)')
    plt.ylabel('Volatility Ratio (σM / μM)')
    plt.title('Volatility Ratio of Pricing Kernel vs Risk Aversion Parameter')
    plt.axhline(y=0.4, color='red', linestyle='--', label='Threshold 0.4')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Find the smallest gamma where volatility ratio exceeds 0.4 or change accordingly
    for gamma, vr in zip(gamma_values, volatility_ratios):
        if vr > 0.4:
            print(f"The smallest value of γ where σM/μM > 0.4 is {gamma:.3f}")
            break
        
def compute_equilibrium(
    gamma: float, 
    delta: float = 0.99, 
    lambd: float = 2, 
    Rf: float = np.exp(0.0198) / 0.99, 
    num_draws: int = 10000
) -> pd.DataFrame:
    """
    Computes the Price-Dividend Ratio (P/D) and Equity Premium for varying b0 using a bisection search method.

    Parameters:
    - gamma (float): Risk aversion parameter.
    - delta (float): Discount factor.
    - lambd (float): Loss aversion parameter.
    - Rf (float): Risk-free rate.
    - num_draws (int): Number of simulations for consumption growth.

    Returns:
    - pd.DataFrame: DataFrame containing b0, P/D ratios, Equity Premiums, and Equilibrium x values.
    """
    # Simulate consumption growth
    epsilon = np.random.randn(num_draws)
    U = np.random.rand(num_draws)
    nu = np.where(U < 0.017, np.log(0.65), 0)
    ln_g = 0.02 + 0.02 * epsilon + nu
    g_tilde = np.exp(ln_g)

    # Define utility function v(R)
    def v(R):
        return R - Rf if R >= Rf else lambd * (R - Rf)
    
    # Function to compute e(x)
    def e_x(x, b0, g_tilde):
        R_tilde = x * g_tilde
        v_vals = np.array([v(R) for R in R_tilde])
        Ev = np.mean(v_vals)
        return delta * b0 * Ev + delta * x - 1

    # Bisection search to find x for each b0
    b0_values = np.arange(0, 10.1, 0.1)
    pd_ratios = []
    equity_premiums = []
    equilibrium_x_values = []

    for b0 in b0_values:
        x_minus = 1.0
        x_plus = 1.1

        # Compute e(x_minus) and e(x_plus)
        e_minus = e_x(x_minus, b0, g_tilde)
        e_plus = e_x(x_plus, b0, g_tilde)

        # Check if a solution exists within the interval
        if e_minus >= 0 or e_plus <= 0:
            print(f"No solution for b0 = {b0}")
            pd_ratios.append(np.nan)
            equity_premiums.append(np.nan)
            equilibrium_x_values.append(np.nan)
            continue

        # Bisection method
        tol = 1e-5
        max_iter = 1000
        iter_count = 0
        while True:
            x_0 = 0.5 * (x_minus + x_plus)
            e_0 = e_x(x_0, b0, g_tilde)
            
            if abs(e_0) < tol or iter_count >= max_iter:
                break
            elif e_0 < 0:
                x_minus = x_0
            else:
                x_plus = x_0
            iter_count +=1

        # Calculate price-dividend ratio P/D = 1 / (x - 1)
        pd_ratio = 1.0 / (x_0 - 1.0)
        pd_ratios.append(pd_ratio)

        # Calculate expected market return E(R_m)
        E_Rm = x_0 * np.exp(0.0202)
        equity_premium = E_Rm - Rf
        equity_premiums.append(equity_premium)

        equilibrium_x_values.append(x_0)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'b0': b0_values,
        'Price-Dividend Ratio (P/D)': pd_ratios,
        'Equity Premium': equity_premiums,
        'Equilibrium x': equilibrium_x_values
    })

    # Plot P/D Ratio vs b0
    plt.figure()
    plt.plot(results_df['b0'], results_df['Price-Dividend Ratio (P/D)'])
    plt.xlabel('b0')
    plt.ylabel('Price-Dividend Ratio (P/D)')
    plt.title('Price-Dividend Ratio vs b0')
    plt.grid(True)
    plt.show()

    # Plot Equity Premium vs b0
    plt.figure()
    plt.plot(results_df['b0'], results_df['Equity Premium'])
    plt.xlabel('b0')
    plt.ylabel('Equity Premium')
    plt.title('Equity Premium vs b0')
    plt.grid(True)
    plt.show()

    return results_df

def plot_equilibrium_results(results_df: pd.DataFrame) -> None:
    """
    Plots the Price-Dividend Ratio and Equity Premium against b0.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing b0, P/D ratios, and Equity Premiums.

    Returns:
    - None
    """
    plt.figure()
    plt.plot(results_df['b0'], results_df['Price-Dividend Ratio (P/D)'], label='P/D Ratio')
    plt.xlabel('b0')
    plt.ylabel('Price-Dividend Ratio (P/D)')
    plt.title('Price-Dividend Ratio vs b0')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(results_df['b0'], results_df['Equity Premium'], label='Equity Premium', color='green')
    plt.xlabel('b0')
    plt.ylabel('Equity Premium')
    plt.title('Equity Premium vs b0')
    plt.grid(True)
    plt.legend()
    plt.show()

