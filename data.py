import pandas as pd
import warnings

def load_data(file_path):
    """
    Loads data from different sheets in an Excel file.

    Parameters:
    - file_path (str): The path to the Excel file containing the data sheets.

    Returns:
    - tuple: Three DataFrames containing Industry Portfolios, Market Portfolio, and Risk Factors data.
             Industry Portfolios: Multiple columns (one per industry) with daily returns
             Market Portfolio: Single column with daily market returns
             Risk Factors: Contains Rf, MRP (Rm-Rf), SMB, and HML factors
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
    
    try:
        # Load each sheet and drop the Date column
        industry_portfolios = pd.read_excel(file_path, sheet_name="Industry_Portfolios")
        industry_portfolios = industry_portfolios.set_index("Date").drop(columns="Date", errors='ignore')
        print(f"Industry Portfolios shape: {industry_portfolios.shape[0]} days, {industry_portfolios.shape[1]} industries")
        
        market_portfolio = pd.read_excel(file_path, sheet_name="Market_Portfolio")
        market_portfolio = market_portfolio.set_index("Date").drop(columns="Date", errors='ignore')
        print(f"Market Portfolio shape: {market_portfolio.shape[0]} days, {market_portfolio.shape[1]} column")
        
        risk_factors = pd.read_excel(file_path, sheet_name="Risk_Factors") 
        risk_factors = risk_factors.set_index("Date").drop(columns="Date", errors='ignore')
        print("Risk Factors loaded:")
        print(f"- {risk_factors.shape[0]} days")
        print("- Columns:", ", ".join(risk_factors.columns))
        
        return industry_portfolios, market_portfolio, risk_factors
        
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        raise
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        raise
