import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import random


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LSTMNetwork(nn.Module):
    """LSTM Neural Network for time series forecasting"""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class LSTMForecaster:
    """LSTM-based forecasting model with caching for consistency"""
    
    def __init__(self, sequence_length=30, hidden_size=50, num_layers=2):
        """
        Initialize LSTM model
        Args:
            sequence_length: number of time steps to look back
            hidden_size: number of LSTM units
            num_layers: number of LSTM layers
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self._cached_model = None
        self._cached_scaler = None
        self._last_train_data_hash = None
    
    def _get_data_hash(self, data):
        """Create a hash of the data to detect changes"""
        return hash(data.tobytes())
    
    def prepare_data(self, data, sequence_length):
        """Prepare data for LSTM training"""
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(self, historical_data, epochs=50, batch_size=32):
        """Train LSTM model"""
        set_seed(42)  # Set seed for reproducibility
        
        # Scale data
        scaled_data = self.scaler.fit_transform(historical_data.reshape(-1, 1))
        
        # Prepare sequences
        X, y = self.prepare_data(scaled_data.flatten(), self.sequence_length)
        
        if len(X) == 0:
            raise ValueError("Not enough data for training")
        
        # Convert to tensors
        X = torch.FloatTensor(X).view(-1, self.sequence_length, 1)
        y = torch.FloatTensor(y).view(-1, 1)
        
        # Initialize model
        self.model = LSTMNetwork(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1
        )
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X)
            loss = criterion(outputs, y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    def predict(self, historical_data, horizon):
        """
        Generate forecast with caching for consistency
        Args:
            historical_data: numpy array of historical prices
            horizon: number of periods to forecast
        Returns:
            numpy array of predictions
        """
        try:
            set_seed(42)  # Always set seed for reproducibility
            
            # Ensure we have enough data
            if len(historical_data) < self.sequence_length:
                print(f"Insufficient data for LSTM ({len(historical_data)} < {self.sequence_length}), using fallback")
                return self._fallback_prediction(historical_data, horizon)
            
            # Check if we can use cached model
            data_hash = self._get_data_hash(historical_data)
            use_cache = (self._last_train_data_hash == data_hash and 
                        self._cached_model is not None and 
                        self._cached_scaler is not None)
            
            if use_cache:
                print("  Using cached LSTM model")
                self.model = self._cached_model
                self.scaler = self._cached_scaler
            else:
                print("  Training new LSTM model")
                # Scale data
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = self.scaler.fit_transform(historical_data.reshape(-1, 1))
                
                # Train model
                X, y = self.prepare_data(scaled_data.flatten(), self.sequence_length)
                
                if len(X) == 0:
                    return self._fallback_prediction(historical_data, horizon)
                
                X = torch.FloatTensor(X).view(-1, self.sequence_length, 1)
                y = torch.FloatTensor(y).view(-1, 1)
                
                # Initialize model
                self.model = LSTMNetwork(1, self.hidden_size, self.num_layers, 1)
                
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
                
                self.model.train()
                for epoch in range(100):  # Increased epochs for better convergence
                    outputs = self.model(X)
                    loss = criterion(outputs, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if epoch % 20 == 0:
                        print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")
                
                # Cache the trained model
                self._cached_model = self.model
                self._cached_scaler = self.scaler
                self._last_train_data_hash = data_hash
            
            # Scale data for prediction
            scaled_data = self.scaler.transform(historical_data.reshape(-1, 1))
            
            # Generate predictions
            self.model.eval()
            predictions = []
            
            # Use last sequence as starting point
            current_sequence = scaled_data[-self.sequence_length:].flatten()
            
            with torch.no_grad():
                for i in range(horizon):
                    # Prepare input
                    x_input = torch.FloatTensor(current_sequence).view(1, self.sequence_length, 1)
                    
                    # Predict next value
                    pred = self.model(x_input)
                    pred_value = pred.item()
                    
                    predictions.append(pred_value)
                    
                    # Update sequence
                    current_sequence = np.append(current_sequence[1:], pred_value)
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            predictions_smooth = predictions.flatten()
            
            print(f"LSTM prediction range: ${predictions_smooth.min():.2f} - ${predictions_smooth.max():.2f}")
            
            return predictions_smooth
        
        except Exception as e:
            print(f"LSTM Error: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_prediction(historical_data, horizon)
    
    def _fallback_prediction(self, data, horizon):
        """Fallback to simple linear extrapolation"""
        set_seed(42)  # Ensure fallback is also deterministic
        
        if len(data) < 2:
            return np.full(horizon, data[-1])
        
        # Simple linear trend with realistic constraints
        recent_data = data[-min(20, len(data)):]
        trend = (recent_data[-1] - recent_data[0]) / len(recent_data)
        
        # Limit trend to avoid unrealistic jumps (max 2% change per step)
        max_change = recent_data[-1] * 0.02
        trend = np.clip(trend, -max_change, max_change)
        
        predictions = []
        last_price = data[-1]
        
        for i in range(1, horizon + 1):
            # Add some mean reversion
            pred = last_price + (trend * i * 0.8)
            
            # Constrain to reasonable range
            pred = np.clip(pred, last_price * 0.8, last_price * 1.2)
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def evaluate(self, train_data, test_data):
        """Evaluate model performance"""
        set_seed(42)
        
        horizon = len(test_data)
        predictions = self.predict(train_data, horizon)
        
        # Ensure same length
        min_len = min(len(test_data), len(predictions))
        test_data = test_data[:min_len]
        predictions = predictions[:min_len]
        
        rmse = np.sqrt(np.mean((test_data - predictions) ** 2))
        mae = np.mean(np.abs(test_data - predictions))
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'predictions': predictions
        }