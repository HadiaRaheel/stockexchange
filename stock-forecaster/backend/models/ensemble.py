import numpy as np
from models.arima_model import ARIMAForecaster, ExponentialSmoothingForecaster
from models.lstm_model import LSTMForecaster

class EnsembleForecaster:
    """Ensemble model combining multiple forecasting methods"""
    
    def __init__(self, weights=None):
        """
        Initialize ensemble model
        Args:
            weights: dict of model weights (default: equal weighting)
        """
        self.arima = ARIMAForecaster()
        self.lstm = LSTMForecaster()
        self.exp_smooth = ExponentialSmoothingForecaster()
        
        # Default weights
        if weights is None:
            self.weights = {
                'arima': 0.3,
                'lstm': 0.5,
                'exp_smooth': 0.2
            }
        else:
            self.weights = weights
    
    def predict(self, historical_data, horizon):
        """
        Generate ensemble forecast
        Args:
            historical_data: numpy array of historical prices
            horizon: number of periods to forecast
        Returns:
            numpy array of weighted predictions
        """
        predictions = {}
        
        # Get predictions from each model
        try:
            predictions['arima'] = self.arima.predict(historical_data, horizon)
        except Exception as e:
            print(f"ARIMA failed: {e}")
            predictions['arima'] = np.full(horizon, historical_data[-1])
        
        try:
            predictions['lstm'] = self.lstm.predict(historical_data, horizon)
        except Exception as e:
            print(f"LSTM failed: {e}")
            predictions['lstm'] = np.full(horizon, historical_data[-1])
        
        try:
            predictions['exp_smooth'] = self.exp_smooth.predict(historical_data, horizon)
        except Exception as e:
            print(f"Exp Smooth failed: {e}")
            predictions['exp_smooth'] = np.full(horizon, historical_data[-1])
        
        # Weighted average
        ensemble_pred = np.zeros(horizon)
        
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 0)
            ensemble_pred += weight * pred[:horizon]
        
        return ensemble_pred
    
    def optimize_weights(self, train_data, validation_data):
        """
        Optimize ensemble weights based on validation performance
        Args:
            train_data: training data
            validation_data: validation data for weight optimization
        """
        from scipy.optimize import minimize
        
        horizon = len(validation_data)
        
        # Get predictions from each model
        predictions = {
            'arima': self.arima.predict(train_data, horizon),
            'lstm': self.lstm.predict(train_data, horizon),
            'exp_smooth': self.exp_smooth.predict(train_data, horizon)
        }
        
        def objective(weights):
            """Minimize RMSE with given weights"""
            ensemble = np.zeros(horizon)
            for i, model_name in enumerate(['arima', 'lstm', 'exp_smooth']):
                ensemble += weights[i] * predictions[model_name][:horizon]
            
            rmse = np.sqrt(np.mean((validation_data - ensemble) ** 2))
            return rmse
        
        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1), (0, 1), (0, 1)]
        
        # Initial weights
        initial_weights = [0.33, 0.33, 0.34]
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimized_weights = result.x
            self.weights = {
                'arima': optimized_weights[0],
                'lstm': optimized_weights[1],
                'exp_smooth': optimized_weights[2]
            }
            print(f"Optimized weights: {self.weights}")
        
        return self.weights
    
    def evaluate(self, train_data, test_data):
        """Evaluate ensemble performance"""
        horizon = len(test_data)
        predictions = self.predict(train_data, horizon)
        
        rmse = np.sqrt(np.mean((test_data - predictions) ** 2))
        mae = np.mean(np.abs(test_data - predictions))
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'predictions': predictions,
            'weights': self.weights
        }