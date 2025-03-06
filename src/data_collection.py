"""
Data Collection Module

This module provides functions to fetch:
1. Historical stock price data using yfinance
2. Fama-French factor data from Kenneth French's data library
3. Federal Reserve interest rate decisions
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
import zipfile
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define data directories
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

#Fetch stock data
def fetch_stock_data(tickers, start_date, end_date, interval='1d'):
    """
    Fetch historical stock price data for a list of tickers
    """
    logger.info(f"Fetching stock data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    try:
        # Download data for all tickers
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval)
        print("Data columns:", data.columns)
        if isinstance(data.columns, pd.MultiIndex):
            print("Column levels:", data.columns.levels)
            print("Level 0 values:", list(data.columns.levels[0]))
            print("Level 1 values:", list(data.columns.levels[1]))

        # Check the structure of the returned data
        if isinstance(data.columns, pd.MultiIndex):
            # For multiple tickers, the result has a MultiIndex
            if 'Close' in data.columns.levels[0]:
                # Just use 'Close' directly
                prices = data['Close']
            else:
                # Fallback if even 'Close' is not available
                logger.warning("'Close' column not available in downloaded data")
                raise ValueError("Required price data not available")
        else:
            # For a single ticker, the result has a single-level index
            if 'Close' in data.columns:
                prices = data['Close'].to_frame(name=tickers[0])
            else:
                # Fallback if 'Close' is not available
                logger.warning("'Close' column not available in downloaded data")
                raise ValueError("Required price data not available")
        
        # Save raw data
        raw_file_path = os.path.join(RAW_DATA_DIR, f"stock_prices_{start_date}_to_{end_date}.csv")
        prices.to_csv(raw_file_path)
        logger.info(f"Raw stock data saved to {raw_file_path}")
        
        return prices
    
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        # Print additional diagnostic information
        if 'data' in locals():
            logger.error(f"Data columns structure: {data.columns}")
        else:
            logger.error("No data downloaded")
        raise

def fetch_fama_french_factors(frequency='daily'):
    """
    Fetch Fama-French 3-factor data from Kenneth French's data library
    
    Parameters:
    -----------
    frequency : str, optional
        Data frequency, either 'daily', 'weekly', or 'monthly'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the Fama-French factors
    """
    logger.info(f"Fetching Fama-French factors with {frequency} frequency")
    
    try:
        # URL for the Fama-French 3-factor data
        if frequency == 'daily':
            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
        elif frequency == 'weekly':
            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_weekly_CSV.zip"
        elif frequency == 'monthly':
            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
        else:
            raise ValueError("Frequency must be 'daily', 'weekly', or 'monthly'")
        
        # Create directories if they don't exist
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # Download the zip file
        response = requests.get(url)
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        
        # Extract the CSV file
        file_list = zip_file.namelist()
        csv_file_name = [f for f in file_list if f.endswith('.CSV')][0]
        
        # Save the raw zip file
        raw_zip_path = os.path.join(RAW_DATA_DIR, f"FF_factors_{frequency}.zip")
        with open(raw_zip_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Raw FF factors zip saved to {raw_zip_path}")
        
        # Manual approach to extract only the data rows
        with zip_file.open(csv_file_name) as f:
            # Read all lines as text
            lines = [line.decode('utf-8').strip() for line in f.readlines()]
        
        # Find and extract only the data rows
        data_rows = []
        column_names = ['date', 'mkt_excess', 'smb', 'hml', 'rf']
        
        # Process each line
        for line in lines:
            # Skip empty lines
            if not line:
                continue
                
            # Check if this looks like a data row (starts with a number)
            parts = line.split(',')
            if len(parts) >= 4:  # We expect at least 4 columns of data
                # Try to validate this is a data row by checking if first part is numeric
                first_part = parts[0].strip()
                
                # For any frequency, the date should be all digits
                if first_part.isdigit():
                    # This appears to be a data row
                    data_rows.append(parts[:5])  # Keep only the first 5 columns
        
        # Create a DataFrame from the extracted data
        if not data_rows:
            raise ValueError("No valid data rows found in the Fama-French file")
            
        # Convert to DataFrame
        ff_data = pd.DataFrame(data_rows, columns=column_names)
        
        # Ensure the date column is properly formatted
        if frequency == 'daily' or frequency == 'weekly':
            # For daily/weekly, dates are in YYYYMMDD format
            ff_data['date'] = pd.to_datetime(ff_data['date'], format='%Y%m%d')
        elif frequency == 'monthly':
            # For monthly, dates are in YYYYMM format, convert to end of month
            ff_data['date'] = pd.to_datetime(ff_data['date'] + '01', format='%Y%m%d')
            ff_data['date'] = ff_data['date'] + pd.offsets.MonthEnd(0)
        
        # Convert numeric columns
        numeric_cols = ['mkt_excess', 'smb', 'hml', 'rf']
        for col in numeric_cols:
            ff_data[col] = pd.to_numeric(ff_data[col], errors='coerce') / 100.0
        
        # Set date as index
        ff_data.set_index('date', inplace=True)
        
        # Add market return column
        ff_data['mkt'] = ff_data['mkt_excess'] + ff_data['rf']
        
        # Drop any rows with NaN values (which might occur if data conversion failed)
        ff_data = ff_data.dropna()
        
        # Save processed data
        processed_file_path = os.path.join(PROCESSED_DATA_DIR, f"ff_factors_{frequency}.csv")
        ff_data.to_csv(processed_file_path)
        logger.info(f"Processed FF factors saved to {processed_file_path}")
        print(f"Successfully processed Fama-French factors with shape: {ff_data.shape}")
        
        return ff_data
        
    except Exception as e:
        logger.error(f"Error fetching Fama-French factors: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
def fetch_fed_rate_decisions(start_year=2000):
    """
    Scrape Federal Reserve interest rate decisions from the Federal Reserve website
    
    Parameters:
    -----------
    start_year : int, optional
        Year to start collecting data from, default is 2000
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the Fed rate decisions
    """
    logger.info(f"Fetching Fed rate decisions from {start_year} to present")
    
    try:
        # URL for the Fed rate decisions
        url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        
        # Fetch the webpage content
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Initialize lists to store data
        dates = []
        decisions = []
        current_year = datetime.now().year
        
        # This is a simplified approach - for a complete solution, you would need
        # to scrape historical pages or use an API that provides this data
        
        # Save the raw HTML for reference
        raw_fed_path = os.path.join(RAW_DATA_DIR, "fed_rate_decisions_page.html")
        with open(raw_fed_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Raw Fed decisions HTML saved to {raw_fed_path}")
        
        # For demonstration purposes, we'll create a simulated dataset
        # In a real project, you would properly extract and parse this data
        
        # Simulated Fed rate decisions
        simulated_data = []
        current_rate = 0.25  # Starting rate
        
        # Generate quarterly meetings with occasional rate changes
        current_date = datetime(start_year, 1, 30)
        end_date = datetime.now()
        
        while current_date < end_date:
            # Determine rate change (simplified model)
            if current_date < datetime(2004, 6, 30):
                # Rising rate period
                change = np.random.choice([0, 0.25], p=[0.7, 0.3])
            elif current_date < datetime(2006, 8, 30):
                # Rising rate period
                change = np.random.choice([0, 0.25], p=[0.5, 0.5])
            elif current_date < datetime(2008, 9, 15):
                # Stable period
                change = 0
            elif current_date < datetime(2009, 12, 31):
                # Financial crisis cuts
                change = np.random.choice([0, -0.25, -0.5], p=[0.2, 0.5, 0.3])
            elif current_date < datetime(2015, 12, 15):
                # Zero lower bound period
                change = 0
            elif current_date < datetime(2019, 7, 31):
                # Rising rate period
                change = np.random.choice([0, 0.25], p=[0.6, 0.4])
            elif current_date < datetime(2020, 3, 15):
                # Pre-pandemic cuts
                change = np.random.choice([0, -0.25], p=[0.6, 0.4])
            elif current_date < datetime(2020, 3, 31):
                # Pandemic emergency cuts
                change = -0.75
            elif current_date < datetime(2022, 3, 15):
                # Zero lower bound period
                change = 0
            elif current_date < datetime(2022, 12, 31):
                # Inflation fighting hikes
                change = np.random.choice([0, 0.25, 0.5, 0.75], p=[0.1, 0.3, 0.4, 0.2])
            elif current_date < datetime(2023, 6, 30):
                # Continued hikes
                change = np.random.choice([0, 0.25], p=[0.3, 0.7])
            else:
                # Pause and potential cuts
                change = np.random.choice([0, -0.25], p=[0.8, 0.2])
            
            # Update current rate
            current_rate = max(0, current_rate + change)
            
            # Determine decision type
            if change > 0:
                decision = "Hike"
            elif change < 0:
                decision = "Cut"
            else:
                decision = "Hold"
            
            # Add to simulated data
            simulated_data.append({
                'date': current_date,
                'rate': round(current_rate, 2),
                'change': change,
                'decision': decision
            })
            
            # Move to next meeting (approximately 6-8 weeks)
            weeks_to_next = np.random.randint(6, 9)
            current_date += timedelta(weeks=weeks_to_next)
        
        # Create DataFrame
        fed_data = pd.DataFrame(simulated_data)
        fed_data.set_index('date', inplace=True)
        
        # Save processed data
        processed_file_path = os.path.join(PROCESSED_DATA_DIR, "fed_rate_decisions.csv")
        fed_data.to_csv(processed_file_path)
        logger.info(f"Processed Fed rate decisions saved to {processed_file_path}")
        
        return fed_data
    
    except Exception as e:
        logger.error(f"Error fetching Fed rate decisions: {str(e)}")
        raise


def fetch_fed_minutes():
    """
    Fetch Federal Reserve meeting minutes for sentiment analysis
    
    This is a placeholder function. In a real implementation, you would scrape
    the Fed minutes from their website and perhaps use NLP to analyze sentiment.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the Fed minutes with timestamps and text
    """
    logger.info("Fetching Fed meeting minutes")
    
    # This would be implemented with web scraping or API calls
    # For now, we'll just return a note
    logger.warning("Fed minutes fetch not implemented - would require web scraping and NLP")
    
    # Placeholder return
    return pd.DataFrame(columns=['date', 'minutes_text', 'sentiment_score'])


if __name__ == "__main__":
    # Example usage
    start_date = '2010-01-01'
    end_date = '2023-12-31'
    tickers = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
    
    # Fetch stock data
    stock_data = fetch_stock_data(tickers, start_date, end_date)
    print(f"Downloaded stock data shape: {stock_data.shape}")
    
    # Fetch Fama-French factors
    ff_factors = fetch_fama_french_factors(frequency='daily')
    print(f"Downloaded FF factors shape: {ff_factors.shape}")
    
    # Fetch Fed rate decisions
    fed_decisions = fetch_fed_rate_decisions()
    print(f"Generated Fed decisions shape: {fed_decisions.shape}")
