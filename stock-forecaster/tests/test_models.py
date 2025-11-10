import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.arima_model import ARIMAForecaster
from models.lstm_model import LSTMForecaster
from models.ensemble import EnsembleForecaster

class TestARIMAForecaster:
    """Test cases for ARIMA model"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.model = ARIMAForecaster()
        self.sample_data = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109] * 10)
    
    def test_model_initialization(self):
        """Test model initializes correctly"""
        assert self.model is not None
        assert self.model.order == (5, 1, 2)
    
    def test_prediction_shape(self):
        """Test prediction returns correct shape"""
        horizon = 10
        predictions = self.model.predict(self.sample_data, horizon)
        
        assert predictions.shape[0] == horizon
        assert isinstance(predictions, np.ndarray)
    
    def test_prediction_values_reasonable(self):
        """Test predictions are in reasonable range"""
        predictions = self.model.predict(self.sample_data, 5)
        
        # Predictions should be within 50% of last value
        last_value = self.sample_data[-1]
        assert all(0.5 * last_value < p < 1.5 * last_value for p in predictions)
    
    def test_evaluation_metrics(self):
        """Test evaluation returns proper metrics"""
        train = self.sample_data[:80]
        test = self.sample_data[80:]
        
        results = self.model.evaluate(train, test)
        
        assert 'rmse' in results
        assert 'mae' in results
        assert 'mape' in results
        assert results['rmse'] >= 0
        assert results['mae'] >= 0
        assert results['mape'] >= 0


class TestLSTMForecaster:
    """Test cases for LSTM model"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.model = LSTMForecaster(sequence_length=10)
        self.sample_data = np.array([100 + i * 0.5 for i in range(100)])
    
    def test_model_initialization(self):
        """Test model initializes correctly"""
        assert self.model is not None
        assert self.model.sequence_length == 10
    
    def test_prediction_shape(self):
        """Test prediction returns correct shape"""
        horizon = 5
        predictions = self.model.predict(self.sample_data, horizon)
        
        assert predictions.shape[0] == horizon
        assert isinstance(predictions, np.ndarray)
    
    def test_data_preparation(self):
        """Test data preparation for LSTM"""
        X, y = self.model.prepare_data(self.sample_data, 10)
        
        assert X.shape[1] == 10
        assert len(X) == len(y)
        assert len(X) == len(self.sample_data) - 10
    
    def test_fallback_prediction(self):
        """Test fallback works with insufficient data"""
        short_data = np.array([100, 101, 102])
        predictions = self.model.predict(short_data, 5)
        
        assert len(predictions) == 5
        assert all(not np.isnan(p) for p in predictions)


class TestEnsembleForecaster:
    """Test cases for Ensemble model"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.model = EnsembleForecaster()
        self.sample_data = np.array([100 + i * 0.5 for i in range(100)])
    
    def test_model_initialization(self):
        """Test ensemble initializes with correct weights"""
        assert self.model is not None
        assert 'arima' in self.model.weights
        assert 'lstm' in self.model.weights
        assert 'exp_smooth' in self.model.weights
        
        # Weights should sum to 1
        total_weight = sum(self.model.weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_prediction_shape(self):
        """Test ensemble prediction returns correct shape"""
        horizon = 10
        predictions = self.model.predict(self.sample_data, horizon)
        
        assert predictions.shape[0] == horizon
        assert isinstance(predictions, np.ndarray)
    
    def test_custom_weights(self):
        """Test ensemble with custom weights"""
        custom_weights = {'arima': 0.5, 'lstm': 0.3, 'exp_smooth': 0.2}
        model = EnsembleForecaster(weights=custom_weights)
        
        assert model.weights == custom_weights
    
    def test_evaluation_metrics(self):
        """Test ensemble evaluation"""
        train = self.sample_data[:80]
        test = self.sample_data[80:]
        
        results = self.model.evaluate(train, test)
        
        assert 'rmse' in results
        assert 'mae' in results
        assert 'mape' in results
        assert 'weights' in results
        assert results['rmse'] >= 0


class TestDataValidation:
    """Test cases for data validation"""
    
    def test_empty_data_handling(self):
        """Test models handle empty data gracefully"""
        model = ARIMAForecaster()
        empty_data = np.array([])
        
        with pytest.raises((ValueError, Exception)):
            model.predict(empty_data, 5)
    
    def test_single_value_data(self):
        """Test models handle single value data"""
        model = LSTMForecaster()
        single_data = np.array([100])
        
        predictions = model.predict(single_data, 3)
        assert len(predictions) == 3
    
    def test_negative_horizon(self):
        """Test models handle negative horizon"""
        model = ARIMAForecaster()
        data = np.array([100, 101, 102, 103])
        
        # Should handle gracefully or raise error
        try:
            predictions = model.predict(data, -5)
            assert len(predictions) == 0 or len(predictions) == 5
        except ValueError:
            pass  # Expected behavior


# Integration Tests
class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_end_to_end_forecast(self):
        """Test complete forecasting pipeline"""
        # Generate sample data
        np.random.seed(42)
        data = 100 + np.cumsum(np.random.randn(200) * 0.5)
        
        # Split data
        train = data[:180]
        test = data[180:]
        horizon = len(test)
        
        # Test each model
        models = {
            'arima': ARIMAForecaster(),
            'lstm': LSTMForecaster(sequence_length=20),
            'ensemble': EnsembleForecaster()
        }
        
        for name, model in models.items():
            predictions = model.predict(train, horizon)
            
            assert len(predictions) == horizon, f"{name} failed shape test"
            assert not np.any(np.isnan(predictions)), f"{name} produced NaN"
            assert not np.any(np.isinf(predictions)), f"{name} produced Inf"
    
    def test_model_comparison(self):
        """Test comparing multiple models"""
        data = np.array([100 + i * 0.3 for i in range(150)])
        train = data[:130]
        test = data[130:]
        
        models = {
            'ARIMA': ARIMAForecaster(),
            'LSTM': LSTMForecaster(sequence_length=20),
            'Ensemble': EnsembleForecaster()
        }
        
        results = {}
        for name, model in models.items():
            result = model.evaluate(train, test)
            results[name] = result
        
        # All models should have valid metrics
        for name, result in results.items():
            assert result['rmse'] > 0, f"{name} has invalid RMSE"
            assert result['mae'] > 0, f"{name} has invalid MAE"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])