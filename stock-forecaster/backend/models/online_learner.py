"""
Online Learning Module for Real-time Model Updates
Place this file in: models/online_learner.py
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class OnlineLSTM:
    """
    LSTM with online learning capabilities
    Updates model incrementally as new data arrives
    """
    
    def __init__(self, sequence_length=30, hidden_size=50, buffer_size=500):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.buffer_size = buffer_size
        
        # Circular buffer for recent data
        self.data_buffer = deque(maxlen=buffer_size)
        
        # Model components
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.scaler = None
        
        # Learning parameters
        self.learning_rate = 0.0001
        self.batch_size = 16
        self.update_frequency = 10  # Update every N new data points
        self.new_data_count = 0
    
    def initialize_model(self):
        """Initialize the LSTM model"""
        from models.lstm_model import LSTMNetwork
        from sklearn.preprocessing import MinMaxScaler
        
        self.model = LSTMNetwork(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=2,
            output_size=1
        )
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def add_observation(self, price_value):
        """
        Add new observation and trigger update if needed
        """
        self.data_buffer.append(price_value)
        self.new_data_count += 1
        
        # Check if we should update
        if (self.new_data_count >= self.update_frequency and 
            len(self.data_buffer) >= self.sequence_length + 1):
            self.incremental_update()
            self.new_data_count = 0
    
    def incremental_update(self):
        """
        Perform incremental model update with buffered data
        """
        try:
            if self.model is None:
                self.initialize_model()
            
            # Get data from buffer
            data = np.array(list(self.data_buffer))
            
            # Scale data
            if len(data) < self.buffer_size // 2:
                # Not enough data to retrain scaler
                scaled_data = self.scaler.transform(data.reshape(-1, 1))
            else:
                # Retrain scaler with buffered data
                scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
            
            # Prepare sequences
            X, y = self._prepare_sequences(scaled_data.flatten())
            
            if len(X) == 0:
                return
            
            # Convert to tensors
            X = torch.FloatTensor(X).view(-1, self.sequence_length, 1)
            y = torch.FloatTensor(y).view(-1, 1)
            
            # Mini-batch update
            self.model.train()
            
            # Single epoch with current batch
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            logger.info(f"Online LSTM update: loss={loss.item():.6f}, buffer_size={len(self.data_buffer)}")
        
        except Exception as e:
            logger.error(f"Online update failed: {e}")
    
    def _prepare_sequences(self, data):
        """Prepare sequences from data"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def predict(self, historical_data, horizon):
        """
        Make prediction using current model
        """
        if self.model is None:
            self.initialize_model()
        
        try:
            # Add historical data to buffer
            for price in historical_data[-self.buffer_size:]:
                if len(self.data_buffer) < self.buffer_size:
                    self.data_buffer.append(price)
            
            # Scale data
            if len(self.data_buffer) >= self.buffer_size // 2:
                scaled_data = self.scaler.fit_transform(
                    np.array(list(self.data_buffer)).reshape(-1, 1)
                )
            else:
                scaled_data = self.scaler.transform(
                    np.array(list(self.data_buffer)).reshape(-1, 1)
                )
            
            # Generate predictions
            self.model.eval()
            predictions = []
            
            current_seq = scaled_data[-self.sequence_length:].flatten()
            
            with torch.no_grad():
                for _ in range(horizon):
                    x_input = torch.FloatTensor(current_seq).view(1, self.sequence_length, 1)
                    pred = self.model(x_input)
                    pred_value = pred.item()
                    
                    predictions.append(pred_value)
                    current_seq = np.append(current_seq[1:], pred_value)
            
            # Inverse transform
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            
            return predictions.flatten()
        
        except Exception as e:
            logger.error(f"Online prediction failed: {e}")
            return np.full(horizon, historical_data[-1])


class RollingWindowRegressor:
    """
    Simple rolling window regression for adaptive predictions
    Uses weighted recent observations
    """
    
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.data_buffer = deque(maxlen=window_size)
        self.weights = None
        self._compute_weights()
    
    def _compute_weights(self):
        """Compute exponential weights favoring recent data"""
        self.weights = np.exp(np.linspace(-2, 0, self.window_size))
        self.weights /= self.weights.sum()
    
    def add_observation(self, value):
        """Add new observation to buffer"""
        self.data_buffer.append(value)
    
    def predict(self, horizon):
        """
        Predict using weighted regression on recent data
        """
        if len(self.data_buffer) < 10:
            return np.full(horizon, list(self.data_buffer)[-1])
        
        data = np.array(list(self.data_buffer))
        
        # Use weights based on available data
        n = len(data)
        weights = self.weights[-n:] if n < self.window_size else self.weights
        
        # Weighted linear regression
        x = np.arange(n)
        
        # Weighted mean
        x_mean = np.average(x, weights=weights)
        y_mean = np.average(data, weights=weights)
        
        # Weighted slope
        numerator = np.sum(weights * (x - x_mean) * (data - y_mean))
        denominator = np.sum(weights * (x - x_mean) ** 2)
        
        if denominator > 0:
            slope = numerator / denominator
        else:
            slope = 0
        
        intercept = y_mean - slope * x_mean
        
        # Generate predictions
        future_x = np.arange(n, n + horizon)
        predictions = slope * future_x + intercept
        
        return predictions


class AdaptiveEnsemble:
    """
    Ensemble that adaptively weights models based on recent performance
    """
    
    def __init__(self, models_dict):
        """
        Args:
            models_dict: Dictionary of {model_name: model_instance}
        """
        self.models = models_dict
        self.weights = {name: 1.0 / len(models_dict) for name in models_dict.keys()}
        
        # Performance tracking
        self.recent_errors = {name: deque(maxlen=20) for name in models_dict.keys()}
        
        self.learning_rate = 0.1
    
    def update_weights(self, predictions_dict, actual_value):
        """
        Update model weights based on prediction errors
        """
        errors = {}
        
        for model_name, prediction in predictions_dict.items():
            error = abs(prediction - actual_value)
            errors[model_name] = error
            self.recent_errors[model_name].append(error)
        
        # Compute average recent errors
        avg_errors = {}
        for model_name in self.models.keys():
            if len(self.recent_errors[model_name]) > 0:
                avg_errors[model_name] = np.mean(list(self.recent_errors[model_name]))
            else:
                avg_errors[model_name] = 1.0
        
        # Convert to weights (inverse of error)
        inverse_errors = {k: 1.0 / (v + 1e-8) for k, v in avg_errors.items()}
        total = sum(inverse_errors.values())
        
        # Update weights with learning rate (smooth transition)
        new_weights = {k: v / total for k, v in inverse_errors.items()}
        
        for model_name in self.weights.keys():
            self.weights[model_name] = (
                (1 - self.learning_rate) * self.weights[model_name] +
                self.learning_rate * new_weights[model_name]
            )
        
        # Normalize
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        logger.info(f"Updated ensemble weights: {self.weights}")
    
    def predict(self, historical_data, horizon):
        """
        Generate weighted ensemble prediction
        """
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict(historical_data, horizon)
                predictions[model_name] = pred
            except Exception as e:
                logger.error(f"{model_name} prediction failed: {e}")
                predictions[model_name] = np.full(horizon, historical_data[-1])
        
        # Weighted average
        ensemble_pred = np.zeros(horizon)
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 0)
            ensemble_pred += weight * pred[:horizon]
        
        return ensemble_pred
    
    def get_weights(self):
        """Return current model weights"""
        return self.weights.copy()