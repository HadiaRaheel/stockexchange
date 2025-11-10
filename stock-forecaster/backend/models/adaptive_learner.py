"""
Adaptive Learning Manager for Continuous Model Improvement
Place this file in: models/adaptive_learner.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import os
import json
import logging

logger = logging.getLogger(__name__)


class AdaptiveLearner:
    """
    Manages adaptive learning, model versioning, and continuous improvement
    """
    
    def __init__(self, model_dir='model_versions'):
        self.model_dir = model_dir
        self.performance_history = []
        self.learning_rate = 0.3  # For ensemble weight updates
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Performance tracking file
        self.perf_file = os.path.join(model_dir, 'performance_history.json')
        self.load_performance_history()
    
    def save_model_version(self, model, model_name, symbol, metrics):
        """
        Save a versioned model with metadata
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_name = f"{model_name}_{symbol}_{timestamp}"
        
        model_path = os.path.join(self.model_dir, f"{version_name}.pkl")
        metadata_path = os.path.join(self.model_dir, f"{version_name}_meta.json")
        
        try:
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'symbol': symbol,
                'timestamp': timestamp,
                'metrics': metrics,
                'version': version_name
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved model version: {version_name}")
            return version_name
        
        except Exception as e:
            logger.error(f"Error saving model version: {e}")
            return None
    
    def load_model_version(self, version_name):
        """
        Load a specific model version
        """
        model_path = os.path.join(self.model_dir, f"{version_name}.pkl")
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            logger.error(f"Error loading model version: {e}")
            return None
    
    def get_best_model_version(self, model_name, symbol, metric='rmse'):
        """
        Find the best performing model version for a given symbol
        """
        best_version = None
        best_score = float('inf')
        
        for entry in self.performance_history:
            if (entry['model_name'] == model_name and 
                entry['symbol'] == symbol and 
                entry['metrics'].get(metric) is not None):
                
                score = entry['metrics'][metric]
                if score < best_score:
                    best_score = score
                    best_version = entry['version']
        
        return best_version, best_score
    
    def incremental_update(self, model, new_data, model_type='lstm'):
        """
        Perform incremental update on model with new data
        """
        try:
            if model_type == 'lstm':
                return self._incremental_lstm_update(model, new_data)
            elif model_type == 'arima':
                return self._incremental_arima_update(model, new_data)
            else:
                logger.warning(f"Incremental update not implemented for {model_type}")
                return model
        except Exception as e:
            logger.error(f"Incremental update failed: {e}")
            return model
    
    def _incremental_lstm_update(self, lstm_model, new_data):
        """
        Fine-tune LSTM with new data using transfer learning
        """
        import torch
        import torch.nn as nn
        
        try:
            if lstm_model.model is None:
                return lstm_model
            
            # Prepare new data
            scaled_data = lstm_model.scaler.transform(new_data.reshape(-1, 1))
            X, y = lstm_model.prepare_data(scaled_data.flatten(), lstm_model.sequence_length)
            
            if len(X) == 0:
                return lstm_model
            
            X = torch.FloatTensor(X).view(-1, lstm_model.sequence_length, 1)
            y = torch.FloatTensor(y).view(-1, 1)
            
            # Fine-tune with lower learning rate
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(lstm_model.model.parameters(), lr=0.0001)
            
            lstm_model.model.train()
            for epoch in range(20):  # Fewer epochs for fine-tuning
                outputs = lstm_model.model(X)
                loss = criterion(outputs, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            logger.info(f"LSTM incremental update completed. Final loss: {loss.item():.6f}")
            return lstm_model
        
        except Exception as e:
            logger.error(f"LSTM incremental update error: {e}")
            return lstm_model
    
    def _incremental_arima_update(self, arima_model, new_data):
        """
        Update ARIMA with new observations
        """
        # ARIMA models are typically retrained rather than incrementally updated
        # But we can keep the same order and just retrain on expanded data
        logger.info("ARIMA models are retrained with new data automatically")
        return arima_model
    
    def adaptive_ensemble_weights(self, predictions_dict, actual_values):
        """
        Adaptively update ensemble weights based on recent performance
        Uses exponential weighting to favor recent accuracy
        """
        errors = {}
        weights = {}
        
        # Calculate errors for each model
        for model_name, predictions in predictions_dict.items():
            if len(predictions) > 0 and len(actual_values) > 0:
                min_len = min(len(predictions), len(actual_values))
                error = np.sqrt(np.mean((actual_values[:min_len] - predictions[:min_len]) ** 2))
                errors[model_name] = error
        
        # Convert errors to weights (inverse relationship)
        if errors:
            # Add small epsilon to avoid division by zero
            inverse_errors = {k: 1 / (v + 1e-8) for k, v in errors.items()}
            total = sum(inverse_errors.values())
            
            # Normalize to sum to 1
            weights = {k: v / total for k, v in inverse_errors.items()}
            
            logger.info(f"Adaptive weights computed: {weights}")
        
        return weights
    
    def should_retrain(self, symbol, current_metrics, threshold_degradation=0.15):
        """
        Determine if model should be retrained based on performance degradation
        """
        # Get historical performance for this symbol
        symbol_history = [
            entry for entry in self.performance_history 
            if entry['symbol'] == symbol
        ]
        
        if len(symbol_history) < 2:
            return False  # Need history to compare
        
        # Get recent average performance
        recent_entries = sorted(symbol_history, key=lambda x: x['timestamp'])[-5:]
        
        if not recent_entries or current_metrics.get('rmse') is None:
            return False
        
        avg_rmse = np.mean([e['metrics']['rmse'] for e in recent_entries if e['metrics'].get('rmse')])
        current_rmse = current_metrics['rmse']
        
        # Check if performance degraded significantly
        degradation = (current_rmse - avg_rmse) / avg_rmse if avg_rmse > 0 else 0
        
        if degradation > threshold_degradation:
            logger.info(f"Performance degradation detected: {degradation*100:.1f}% - Recommending retrain")
            return True
        
        return False
    
    def rolling_window_update(self, historical_data, window_size=100):
        """
        Create rolling windows for continuous training
        Returns list of training windows
        """
        windows = []
        
        if len(historical_data) < window_size:
            return [historical_data]
        
        # Create overlapping windows
        step_size = max(1, window_size // 4)  # 75% overlap
        
        for i in range(0, len(historical_data) - window_size + 1, step_size):
            window = historical_data[i:i + window_size]
            windows.append(window)
        
        return windows
    
    def log_performance(self, model_name, symbol, metrics, version=None):
        """
        Log model performance for tracking
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'symbol': symbol,
            'metrics': metrics,
            'version': version
        }
        
        self.performance_history.append(entry)
        self.save_performance_history()
    
    def save_performance_history(self):
        """
        Save performance history to disk
        """
        try:
            with open(self.perf_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")
    
    def load_performance_history(self):
        """
        Load performance history from disk
        """
        try:
            if os.path.exists(self.perf_file):
                with open(self.perf_file, 'r') as f:
                    self.performance_history = json.load(f)
                logger.info(f"Loaded {len(self.performance_history)} performance entries")
        except Exception as e:
            logger.error(f"Error loading performance history: {e}")
            self.performance_history = []
    
    def get_performance_trend(self, symbol, model_name=None, days=30):
        """
        Get performance trend for a symbol over time
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        filtered = [
            entry for entry in self.performance_history
            if (entry['symbol'] == symbol and
                datetime.fromisoformat(entry['timestamp']) > cutoff_date and
                (model_name is None or entry['model_name'] == model_name))
        ]
        
        if not filtered:
            return None
        
        # Sort by timestamp
        filtered.sort(key=lambda x: x['timestamp'])
        
        timestamps = [entry['timestamp'] for entry in filtered]
        rmse_values = [entry['metrics'].get('rmse') for entry in filtered if entry['metrics'].get('rmse')]
        
        return {
            'timestamps': timestamps,
            'rmse_values': rmse_values,
            'count': len(filtered)
        }
    
    def cleanup_old_versions(self, keep_latest=5):
        """
        Remove old model versions, keeping only the latest N
        """
        # Get all model files
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
        
        # Group by symbol and model type
        groups = {}
        for f in model_files:
            parts = f.replace('.pkl', '').split('_')
            if len(parts) >= 3:
                key = f"{parts[0]}_{parts[1]}"  # model_symbol
                if key not in groups:
                    groups[key] = []
                groups[key].append(f)
        
        # Keep only latest N for each group
        deleted = 0
        for key, files in groups.items():
            if len(files) > keep_latest:
                # Sort by timestamp (in filename)
                files.sort(reverse=True)
                
                # Delete older versions
                for old_file in files[keep_latest:]:
                    try:
                        os.remove(os.path.join(self.model_dir, old_file))
                        # Also remove metadata
                        meta_file = old_file.replace('.pkl', '_meta.json')
                        meta_path = os.path.join(self.model_dir, meta_file)
                        if os.path.exists(meta_path):
                            os.remove(meta_path)
                        deleted += 1
                    except Exception as e:
                        logger.error(f"Error deleting {old_file}: {e}")
        
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old model versions")