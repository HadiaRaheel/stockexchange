import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

class ARIMAForecaster:
    """Traditional ARIMA forecasting model"""
    
    def __init__(self, order=(5, 1, 2)):
        """
        Initialize ARIMA model
        Args:
            order: tuple (p, d, q) for ARIMA parameters
        """
        self.order = order
        self.model = None
    
    def find_best_order(self, data, max_p=5, max_d=2, max_q=5):
        """
        Find best ARIMA order using AIC
        """
        best_aic = np.inf
        best_order = None
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        results = model.fit()
                        
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order if best_order else (1, 1, 1)
    
    def predict(self, historical_data, horizon):
        """
        Generate forecast
        Args:
            historical_data: numpy array of historical prices
            horizon: number of periods to forecast
        Returns:
            numpy array of predictions
        """
        try:
            # Fit model
            model = ARIMA(historical_data, order=self.order)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=horizon)
            
            return np.array(forecast)
        
        except Exception as e:
            print(f"ARIMA Error: {e}")
            # Fallback to simple moving average
            return self._moving_average_fallback(historical_data, horizon)
    
    def _moving_average_fallback(self, data, horizon):
        """Simple moving average as fallback"""
        window = min(20, len(data) // 4)
        last_values = data[-window:]
        mean = np.mean(last_values)
        trend = (data[-1] - data[-window]) / window
        
        predictions = []
        for i in range(1, horizon + 1):
            pred = mean + trend * i
            predictions.append(pred)
        
        return np.array(predictions)
    
    def evaluate(self, train_data, test_data):
        """
        Evaluate model performance
        """
        horizon = len(test_data)
        predictions = self.predict(train_data, horizon)
        
        # Metrics
        rmse = np.sqrt(np.mean((test_data - predictions) ** 2))
        mae = np.mean(np.abs(test_data - predictions))
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'predictions': predictions
        }

class ExponentialSmoothingForecaster:
    """Exponential Smoothing for trend and seasonality"""
    
    def __init__(self):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        self.ExponentialSmoothing = ExponentialSmoothing
    
    def predict(self, historical_data, horizon):
        """Generate forecast using Holt-Winters"""
        try:
            model = self.ExponentialSmoothing(
                historical_data,
                trend='add',
                seasonal=None
            )
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=horizon)
            
            return np.array(forecast)
        
        except Exception as e:
            print(f"Exponential Smoothing Error: {e}")
            return np.full(horizon, historical_data[-1])