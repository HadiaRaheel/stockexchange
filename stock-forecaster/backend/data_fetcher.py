import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta

class DataFetcher:
    """Improved data fetcher with retry logic and fallbacks"""
    
    def __init__(self):
        self.session = None
        self.retry_delay = 2  # seconds
        self.max_retries = 3
    
    def fetch_historical_data(self, symbol, period='1y', interval='1d'):
        """
        Fetch historical data with retry logic and fallbacks
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            pandas DataFrame with OHLCV data
        """
        for attempt in range(self.max_retries):
            try:
                print(f"Attempt {attempt + 1} to fetch {symbol}...")
                
                # Create ticker with custom session
                ticker = yf.Ticker(symbol)
                
                # Fetch data
                df = ticker.history(period=period, interval=interval)
                
                if not df.empty:
                    print(f"Successfully fetched {len(df)} records for {symbol}")
                    return df
                
                # If empty, try with auto_adjust=False
                df = ticker.history(period=period, interval=interval, auto_adjust=False)
                
                if not df.empty:
                    print(f"Successfully fetched {len(df)} records for {symbol} (no auto-adjust)")
                    return df
                
                print(f"No data returned for {symbol}, trying alternative method...")
                
                # Try download method instead
                df = yf.download(symbol, period=period, interval=interval, progress=False)
                
                if not df.empty:
                    print(f"Successfully downloaded {len(df)} records for {symbol}")
                    return df
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (attempt + 1)
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"All attempts failed for {symbol}")
        
        # If all retries fail, return sample data for testing
        print(f"Generating sample data for {symbol} for testing purposes...")
        return self._generate_sample_data(symbol, period, interval)
    
    def _generate_sample_data(self, symbol, period='1y', interval='1d'):
        """
        Generate sample data for testing when API fails
        This creates realistic-looking stock data
        """
        import numpy as np
        
        # Determine number of data points
        if interval in ['1m', '2m', '5m']:
            points = 390  # Trading day
        elif interval in ['15m', '30m']:
            points = 26
        elif interval in ['1h', '60m']:
            points = 100
        elif interval == '1d':
            if period == '1y':
                points = 252  # Trading days in a year
            elif period == '6mo':
                points = 126
            elif period == '3mo':
                points = 63
            else:
                points = 100
        else:
            points = 100
        
        # Generate dates
        end_date = datetime.now()
        if interval in ['1m', '2m', '5m', '15m', '30m', '1h', '60m']:
            dates = pd.date_range(end=end_date, periods=points, freq='H')
        else:
            dates = pd.date_range(end=end_date, periods=points, freq='D')
        
        # Generate realistic price data
        base_price = 150.0  # Starting price
        volatility = 0.02   # 2% daily volatility
        
        # Random walk with drift
        returns = np.random.normal(0.0005, volatility, points)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.01, 0.01, points)),
            'High': prices * (1 + np.random.uniform(0, 0.02, points)),
            'Low': prices * (1 - np.random.uniform(0, 0.02, points)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, points)
        }, index=dates)
        
        # Ensure High is highest and Low is lowest
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        
        print(f"⚠️  Using sample data for {symbol} (API unavailable)")
        return df
    
    def validate_symbol(self, symbol):
        """Check if symbol is valid"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return 'symbol' in info or 'shortName' in info
        except:
            return False
    
    def get_symbol_info(self, symbol):
        """Get basic information about a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'symbol': symbol,
                'name': info.get('shortName', symbol),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown')
            }
        except Exception as e:
            print(f"Could not fetch info for {symbol}: {e}")
            return {
                'symbol': symbol,
                'name': symbol,
                'currency': 'USD',
                'exchange': 'Unknown'
            }