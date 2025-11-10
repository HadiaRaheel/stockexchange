# """
# Adaptive Learning Manager for Continuous Model Improvement
# Place this file in: models/adaptive_learner.py
# """

# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta
# import pickle
# import os
# import json
# import logging

# logger = logging.getLogger(__name__)


# class AdaptiveLearner:
#     """
#     Manages adaptive learning, model versioning, and continuous improvement
#     """
    
#     def __init__(self, model_dir='model_versions'):
#         self.model_dir = model_dir
#         self.performance_history = []
#         self.learning_rate = 0.3  # For ensemble weight updates
        
#         # Create model directory if it doesn't exist
#         os.makedirs(model_dir, exist_ok=True)
        
#         # Performance tracking file
#         self.perf_file = os.path.join(model_dir, 'performance_history.json')
#         self.load_performance_history()
    
#     def save_model_version(self, model, model_name, symbol, metrics):
#         """
#         Save a versioned model with metadata
#         """
#         now = datetime.now()
#         timestamp = now.strftime("%Y%m%d_%H%M%S")  # store string
#         iso_timestamp = now.isoformat()
#         version_name = f"{model_name}_{symbol}_{timestamp}"
        
#         # model_path = os.path.join(self.model_dir, f"{version_name}.pkl")
#         # metadata_path = os.path.join(self.model_dir, f"{version_name}_meta.json")
        
#         try:
#             # Save model
#             # with open(model_path, 'wb') as f:
#             #     pickle.dump(model, f)
            
#             # Save metadata
#             metadata = {
#                 'model_name': model_name,
#                 'symbol': symbol,
#                 'timestamp': iso_timestamp,
#                 'metrics': metrics,
#                 'version': version_name
#             }
#             path = os.path.join(self.model_dir, f"{version_name}_meta.json")
#             with open(path, 'w') as f:
#                 json.dump(metadata, f, indent=2)
    
            
#             # with open(metadata_path, 'w') as f:
#             #     json.dump(metadata, f, indent=2)
            
#             # logger.info(f"Saved model version: {version_name}")
#             # return version_name
        
#         except Exception as e:
#             logger.error(f"Error saving model version: {e}")
#             return None
    
#     def load_model_version(self, version_name):
#         """
#         Load a specific model version
#         """
#         model_path = os.path.join(self.model_dir, f"{version_name}.pkl")
        
#         try:
#             with open(model_path, 'rb') as f:
#                 model = pickle.load(f)
#             return model
#         except Exception as e:
#             logger.error(f"Error loading model version: {e}")
#             return None
    
#     def get_best_model_version(self, model_name, symbol, metric='rmse'):
#         """
#         Find the best performing model version for a given symbol
#         """
#         best_version = None
#         best_score = float('inf')
        
#         for entry in self.performance_history:
#             if (entry['model_name'] == model_name and 
#                 entry['symbol'] == symbol and 
#                 entry['metrics'].get(metric) is not None):
                
#                 score = entry['metrics'][metric]
#                 if score < best_score:
#                     best_score = score
#                     best_version = entry['version']
        
#         return best_version, best_score
    
#     def incremental_update(self, model, new_data, model_type='lstm'):
#         """
#         Perform incremental update on model with new data
#         """
#         try:
#             if model_type == 'lstm':
#                 return self._incremental_lstm_update(model, new_data)
#             elif model_type == 'arima':
#                 return self._incremental_arima_update(model, new_data)
#             else:
#                 logger.warning(f"Incremental update not implemented for {model_type}")
#                 return model
#         except Exception as e:
#             logger.error(f"Incremental update failed: {e}")
#             return model
    
#     def _incremental_lstm_update(self, lstm_model, new_data):
#         """
#         Fine-tune LSTM with new data using transfer learning
#         """
#         import torch
#         import torch.nn as nn
        
#         try:
#             if lstm_model.model is None:
#                 return lstm_model
            
#             # Prepare new data
#             scaled_data = lstm_model.scaler.transform(new_data.reshape(-1, 1))
#             X, y = lstm_model.prepare_data(scaled_data.flatten(), lstm_model.sequence_length)
            
#             if len(X) == 0:
#                 return lstm_model
            
#             X = torch.FloatTensor(X).view(-1, lstm_model.sequence_length, 1)
#             y = torch.FloatTensor(y).view(-1, 1)
            
#             # Fine-tune with lower learning rate
#             criterion = nn.MSELoss()
#             optimizer = torch.optim.Adam(lstm_model.model.parameters(), lr=0.0001)
            
#             lstm_model.model.train()
#             for epoch in range(20):  # Fewer epochs for fine-tuning
#                 outputs = lstm_model.model(X)
#                 loss = criterion(outputs, y)
                
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
            
#             logger.info(f"LSTM incremental update completed. Final loss: {loss.item():.6f}")
#             return lstm_model
        
#         except Exception as e:
#             logger.error(f"LSTM incremental update error: {e}")
#             return lstm_model
    
#     def _incremental_arima_update(self, arima_model, new_data):
#         """
#         Update ARIMA with new observations
#         """
#         # ARIMA models are typically retrained rather than incrementally updated
#         # But we can keep the same order and just retrain on expanded data
#         logger.info("ARIMA models are retrained with new data automatically")
#         return arima_model
    
#     def adaptive_ensemble_weights(self, predictions_dict, actual_values):
#         """
#         Adaptively update ensemble weights based on recent performance
#         Uses exponential weighting to favor recent accuracy
#         """
#         errors = {}
#         weights = {}
        
#         # Calculate errors for each model
#         for model_name, predictions in predictions_dict.items():
#             if len(predictions) > 0 and len(actual_values) > 0:
#                 min_len = min(len(predictions), len(actual_values))
#                 error = np.sqrt(np.mean((actual_values[:min_len] - predictions[:min_len]) ** 2))
#                 errors[model_name] = error
        
#         # Convert errors to weights (inverse relationship)
#         if errors:
#             # Add small epsilon to avoid division by zero
#             inverse_errors = {k: 1 / (v + 1e-8) for k, v in errors.items()}
#             total = sum(inverse_errors.values())
            
#             # Normalize to sum to 1
#             weights = {k: v / total for k, v in inverse_errors.items()}
            
#             logger.info(f"Adaptive weights computed: {weights}")
        
#         return weights
    
#     def should_retrain(self, symbol, current_metrics, threshold_degradation=0.15):
#         """
#         Determine if model should be retrained based on performance degradation
#         """
#         # Get historical performance for this symbol
#         symbol_history = [
#             entry for entry in self.performance_history 
#             if entry['symbol'] == symbol
#         ]
        
#         if len(symbol_history) < 2:
#             return False  # Need history to compare
        
#         # Get recent average performance
#         recent_entries = sorted(symbol_history, key=lambda x: x['timestamp'])[-5:]
        
#         if not recent_entries or current_metrics.get('rmse') is None:
#             return False
        
#         avg_rmse = np.mean([e['metrics']['rmse'] for e in recent_entries if e['metrics'].get('rmse')])
#         current_rmse = current_metrics['rmse']
        
#         # Check if performance degraded significantly
#         degradation = (current_rmse - avg_rmse) / avg_rmse if avg_rmse > 0 else 0
        
#         if degradation > threshold_degradation:
#             logger.info(f"Performance degradation detected: {degradation*100:.1f}% - Recommending retrain")
#             return True
        
#         return False
    
#     def rolling_window_update(self, historical_data, window_size=100):
#         """
#         Create rolling windows for continuous training
#         Returns list of training windows
#         """
#         windows = []
        
#         if len(historical_data) < window_size:
#             return [historical_data]
        
#         # Create overlapping windows
#         step_size = max(1, window_size // 4)  # 75% overlap
        
#         for i in range(0, len(historical_data) - window_size + 1, step_size):
#             window = historical_data[i:i + window_size]
#             windows.append(window)
        
#         return windows
    
#     def log_performance(self, model_name, symbol, metrics, version=None):
#         """
#         Log model performance for tracking
#         """
#         entry = {
#             'timestamp': datetime.now().isoformat(),
#             'model_name': model_name,
#             'symbol': symbol,
#             'metrics': metrics,
#             'version': version
#         }
        
#         self.performance_history.append(entry)
#         self.save_performance_history()
    
#     def save_performance_history(self):
#         """
#         Save performance history to disk
#         """
#         try:
#             with open(self.perf_file, 'w') as f:
#                 json.dump(self.performance_history, f, indent=2)
#         except Exception as e:
#             logger.error(f"Error saving performance history: {e}")
    
#     def load_performance_history(self):
#         """
#         Load performance history from disk
#         """
#         try:
#             if os.path.exists(self.perf_file):
#                 with open(self.perf_file, 'r') as f:
#                     self.performance_history = json.load(f)
#                 logger.info(f"Loaded {len(self.performance_history)} performance entries")
#         except Exception as e:
#             logger.error(f"Error loading performance history: {e}")
#             self.performance_history = []
    
#     def get_performance_trend(self, symbol, model_name=None, days=30):
#         import os, json
#         import datetime

#         files = [f for f in os.listdir(self.model_dir) if f.endswith('_meta.json') and symbol in f]
#         files.sort(reverse=True)  # latest first

#         timestamps = []
#         rmse_values = []
#         count = 0

#         for f in files:
#             with open(os.path.join(self.model_dir, f), 'r') as file:
#                 meta = json.load(file)
            
#             if model_name and meta['model_name'].lower() != model_name.lower():
#                 continue
            
#             ts = meta.get('timestamp')
#             try:
#                 # parse ISO timestamp
#                 ts_dt = datetime.datetime.fromisoformat(ts)
#                 timestamps.append(ts_dt.isoformat())
#             except:
#                 continue
            
#             metrics = meta.get('metrics', {})
#             rmse = metrics.get('rmse', None)
#             rmse_values.append(rmse if rmse is not None else 0)
#             count += 1
            
#             if count >= days:
#                 break
        
#         if count == 0:
#             return None
        
#         return {
#             'timestamps': timestamps[::-1],  # oldest first
#             'rmse_values': rmse_values[::-1],
#             'count': count
#         }
#         # """
#         # Get performance trend for a symbol over time
#         # """
#         # cutoff_date = datetime.now() - timedelta(days=days)
        
#         # filtered = [
#         #     entry for entry in self.performance_history
#         #     if (entry['symbol'] == symbol and
#         #         datetime.fromisoformat(entry['timestamp']) > cutoff_date and
#         #         (model_name is None or entry['model_name'] == model_name))
#         # ]
        
#         # if not filtered:
#         #     return None
        
#         # # Sort by timestamp
#         # filtered.sort(key=lambda x: x['timestamp'])
        
#         # timestamps = [entry['timestamp'] for entry in filtered]
#         # rmse_values = [entry['metrics'].get('rmse') for entry in filtered if entry['metrics'].get('rmse')]
        
#         # return {
#         #     'timestamps': timestamps,
#         #     'rmse_values': rmse_values,
#         #     'count': len(filtered)
#         # }


        
    
#     def cleanup_old_versions(self, keep_latest=5):
#         """
#         Remove old model versions, keeping only the latest N
#         """
#         # Get all model files
#         model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
        
#         # Group by symbol and model type
#         groups = {}
#         for f in model_files:
#             parts = f.replace('.pkl', '').split('_')
#             if len(parts) >= 3:
#                 key = f"{parts[0]}_{parts[1]}"  # model_symbol
#                 if key not in groups:
#                     groups[key] = []
#                 groups[key].append(f)
        
#         # Keep only latest N for each group
#         deleted = 0
#         for key, files in groups.items():
#             if len(files) > keep_latest:
#                 # Sort by timestamp (in filename)
#                 files.sort(reverse=True)
                
#                 # Delete older versions
#                 for old_file in files[keep_latest:]:
#                     try:
#                         os.remove(os.path.join(self.model_dir, old_file))
#                         # Also remove metadata
#                         meta_file = old_file.replace('.pkl', '_meta.json')
#                         meta_path = os.path.join(self.model_dir, meta_file)
#                         if os.path.exists(meta_path):
#                             os.remove(meta_path)
#                         deleted += 1
#                     except Exception as e:
#                         logger.error(f"Error deleting {old_file}: {e}")
        
#         if deleted > 0:
#             logger.info(f"Cleaned up {deleted} old model versions")



"""
Enhanced Adaptive Learning Manager with Robust Metrics Handling
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
        Enhanced to handle None metrics properly
        """
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        iso_timestamp = now.isoformat()
        version_name = f"{model_name}_{symbol}_{timestamp}"
        
        try:
            # Validate and clean metrics
            clean_metrics = self._validate_metrics(metrics)
            
            # Save model (pickle)
            model_path = os.path.join(self.model_dir, f"{version_name}.pkl")
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved model pickle: {version_name}.pkl")
            except Exception as pickle_error:
                logger.warning(f"Could not pickle model: {pickle_error}")
                # Continue even if pickle fails - metadata is more important
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'symbol': symbol,
                'timestamp': iso_timestamp,
                'metrics': clean_metrics,
                'version': version_name,
                'training_date': now.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            meta_path = os.path.join(self.model_dir, f"{version_name}_meta.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Also log to performance history
            self.log_performance(model_name, symbol, clean_metrics, version_name)
            
            logger.info(f"✓ Saved model version: {version_name}")
            logger.info(f"  Metrics: RMSE=${clean_metrics.get('rmse', 'N/A')}, MAE=${clean_metrics.get('mae', 'N/A')}")
            
            return version_name
        
        except Exception as e:
            logger.error(f"Error saving model version: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _validate_metrics(self, metrics):
        """
        Validate and clean metrics dictionary
        Ensures all metrics are valid numbers or None
        """
        if metrics is None:
            return {'rmse': None, 'mae': None, 'mape': None}
        
        clean = {}
        for key in ['rmse', 'mae', 'mape']:
            value = metrics.get(key)
            
            # Check if value is valid
            if value is None:
                clean[key] = None
            elif isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                clean[key] = float(value)
            else:
                clean[key] = None
                logger.warning(f"Invalid {key} value: {value}, setting to None")
        
        return clean
    
    def load_model_version(self, version_name):
        """
        Load a specific model version
        """
        model_path = os.path.join(self.model_dir, f"{version_name}.pkl")
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded model version: {version_name}")
            return model
        except Exception as e:
            logger.error(f"Error loading model version: {e}")
            return None
    
    def get_best_model_version(self, model_name, symbol, metric='rmse'):
        """
        Find the best performing model version for a given symbol
        Only considers versions with valid metrics
        """
        best_version = None
        best_score = float('inf')
        candidates_found = 0
        
        for entry in self.performance_history:
            if (entry['model_name'] == model_name and 
                entry['symbol'] == symbol):
                
                score = entry['metrics'].get(metric)
                
                # Only consider if metric is valid
                if score is not None and isinstance(score, (int, float)) and not np.isnan(score):
                    candidates_found += 1
                    if score < best_score:
                        best_score = score
                        best_version = entry['version']
        
        if best_version:
            logger.info(f"Best {model_name} version for {symbol}: {best_version} ({metric}=${best_score:.2f})")
            logger.info(f"  Evaluated {candidates_found} candidate versions")
        else:
            logger.info(f"No valid versions found for {model_name} {symbol}")
        
        return best_version, best_score if best_version else None
    
    def incremental_update(self, model, new_data, model_type='lstm'):
        """
        Perform incremental update on model with new data
        Enhanced with better error handling and logging
        """
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"INCREMENTAL UPDATE: {model_type.upper()}")
            logger.info(f"New data points: {len(new_data)}")
            logger.info(f"{'='*50}")
            
            if model_type == 'lstm':
                return self._incremental_lstm_update(model, new_data)
            elif model_type == 'arima':
                return self._incremental_arima_update(model, new_data)
            else:
                logger.warning(f"Incremental update not implemented for {model_type}")
                return model
        except Exception as e:
            logger.error(f"Incremental update failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return model
    
    def _incremental_lstm_update(self, lstm_model, new_data):
        """
        Fine-tune LSTM with new data using transfer learning
        Enhanced with validation and metrics
        """
        import torch
        import torch.nn as nn
        
        try:
            if lstm_model.model is None:
                logger.warning("LSTM model not initialized, skipping update")
                return lstm_model
            
            # Validate new data
            if len(new_data) < lstm_model.sequence_length + 10:
                logger.warning(f"Insufficient new data for update: {len(new_data)} points")
                return lstm_model
            
            # Prepare new data
            scaled_data = lstm_model.scaler.transform(new_data.reshape(-1, 1))
            X, y = lstm_model.prepare_data(scaled_data.flatten(), lstm_model.sequence_length)
            
            if len(X) == 0:
                logger.warning("No sequences created from new data")
                return lstm_model
            
            X = torch.FloatTensor(X).view(-1, lstm_model.sequence_length, 1)
            y = torch.FloatTensor(y).view(-1, 1)
            
            # Fine-tune with lower learning rate
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(lstm_model.model.parameters(), lr=0.0001)
            
            lstm_model.model.train()
            initial_loss = None
            
            for epoch in range(20):  # Fewer epochs for fine-tuning
                outputs = lstm_model.model(X)
                loss = criterion(outputs, y)
                
                if epoch == 0:
                    initial_loss = loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if epoch % 5 == 0:
                    logger.info(f"  Epoch {epoch}: Loss = {loss.item():.6f}")
            
            final_loss = loss.item()
            improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss else 0
            
            logger.info(f"✓ LSTM incremental update completed")
            logger.info(f"  Initial loss: {initial_loss:.6f}")
            logger.info(f"  Final loss: {final_loss:.6f}")
            logger.info(f"  Improvement: {improvement:.2f}%")
            logger.info(f"  Sequences trained: {len(X)}")
            
            return lstm_model
        
        except Exception as e:
            logger.error(f"LSTM incremental update error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return lstm_model
    
    def _incremental_arima_update(self, arima_model, new_data):
        """
        Update ARIMA with new observations
        Note: ARIMA models are typically retrained rather than incrementally updated
        """
        logger.info("ARIMA models will be retrained with new data on next prediction")
        return arima_model
    
    def adaptive_ensemble_weights(self, predictions_dict, actual_values):
        """
        Adaptively update ensemble weights based on recent performance
        Uses exponential weighting to favor recent accuracy
        Enhanced with validation
        """
        errors = {}
        weights = {}
        
        logger.info("\nComputing adaptive ensemble weights...")
        
        # Calculate errors for each model
        for model_name, predictions in predictions_dict.items():
            if len(predictions) > 0 and len(actual_values) > 0:
                min_len = min(len(predictions), len(actual_values))
                error = np.sqrt(np.mean((actual_values[:min_len] - predictions[:min_len]) ** 2))
                errors[model_name] = error
                logger.info(f"  {model_name}: RMSE = ${error:.2f}")
        
        # Convert errors to weights (inverse relationship)
        if errors:
            # Add small epsilon to avoid division by zero
            inverse_errors = {k: 1 / (v + 1e-8) for k, v in errors.items()}
            total = sum(inverse_errors.values())
            
            # Normalize to sum to 1
            weights = {k: v / total for k, v in inverse_errors.items()}
            
            logger.info(f"\n✓ Adaptive weights computed:")
            for name, weight in weights.items():
                logger.info(f"  {name}: {weight*100:.1f}%")
        
        return weights
    
    def should_retrain(self, symbol, current_metrics, threshold_degradation=0.15):
        """
        Determine if model should be retrained based on performance degradation
        Enhanced with more sophisticated logic
        """
        # Validate current metrics
        if current_metrics.get('rmse') is None:
            logger.info("Cannot assess retrain need: current metrics unavailable")
            return False
        
        # Get historical performance for this symbol
        symbol_history = [
            entry for entry in self.performance_history 
            if entry['symbol'] == symbol and entry['metrics'].get('rmse') is not None
        ]
        
        if len(symbol_history) < 3:  # Need at least 3 data points
            logger.info(f"Insufficient history for {symbol}: {len(symbol_history)} entries")
            return False
        
        # Get recent average performance (last 5 valid entries)
        recent_entries = sorted(
            symbol_history, 
            key=lambda x: x['timestamp']
        )[-5:]
        
        avg_rmse = np.mean([e['metrics']['rmse'] for e in recent_entries])
        current_rmse = current_metrics['rmse']
        
        # Check if performance degraded significantly
        degradation = (current_rmse - avg_rmse) / avg_rmse if avg_rmse > 0 else 0
        
        logger.info(f"\nRetrain Assessment for {symbol}:")
        logger.info(f"  Historical avg RMSE: ${avg_rmse:.2f}")
        logger.info(f"  Current RMSE: ${current_rmse:.2f}")
        logger.info(f"  Degradation: {degradation*100:.1f}%")
        logger.info(f"  Threshold: {threshold_degradation*100:.1f}%")
        
        if degradation > threshold_degradation:
            logger.warning(f"⚠ Performance degradation detected - RETRAIN RECOMMENDED")
            return True
        else:
            logger.info(f"✓ Performance acceptable - no retrain needed")
            return False
    
    def rolling_window_update(self, historical_data, window_size=100):
        """
        Create rolling windows for continuous training
        Returns list of training windows for walk-forward validation
        """
        windows = []
        
        if len(historical_data) < window_size:
            logger.warning(f"Data too short for rolling windows: {len(historical_data)} < {window_size}")
            return [historical_data]
        
        # Create overlapping windows
        step_size = max(1, window_size // 4)  # 75% overlap
        
        for i in range(0, len(historical_data) - window_size + 1, step_size):
            window = historical_data[i:i + window_size]
            windows.append(window)
        
        logger.info(f"Created {len(windows)} rolling windows (size={window_size}, step={step_size})")
        return windows
    
    def log_performance(self, model_name, symbol, metrics, version=None):
        """
        Log model performance for tracking
        Enhanced with validation
        """
        # Validate metrics before logging
        clean_metrics = self._validate_metrics(metrics)
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'symbol': symbol,
            'metrics': clean_metrics,
            'version': version
        }
        
        self.performance_history.append(entry)
        self.save_performance_history()
        
        logger.info(f"Logged performance: {model_name} {symbol} - RMSE=${clean_metrics.get('rmse', 'N/A')}")
    
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
        Enhanced to use both metadata files and performance history
        """
        import os
        import json
        import datetime
        
        # Try to load from metadata files first (most reliable)
        files = [f for f in os.listdir(self.model_dir) if f.endswith('_meta.json') and symbol in f]
        files.sort(reverse=True)  # Latest first
        
        timestamps = []
        rmse_values = []
        mae_values = []
        count = 0
        
        for f in files:
            if count >= days:
                break
                
            try:
                with open(os.path.join(self.model_dir, f), 'r') as file:
                    meta = json.load(file)
                
                # Filter by model name if specified
                if model_name and meta['model_name'].lower() != model_name.lower():
                    continue
                
                ts = meta.get('timestamp')
                if ts:
                    try:
                        ts_dt = datetime.datetime.fromisoformat(ts)
                        timestamps.append(ts_dt.isoformat())
                    except:
                        continue
                    
                    metrics = meta.get('metrics', {})
                    rmse = metrics.get('rmse')
                    mae = metrics.get('mae')
                    
                    # Only include if we have valid metrics
                    if rmse is not None:
                        rmse_values.append(float(rmse))
                        mae_values.append(float(mae) if mae is not None else None)
                        count += 1
            except Exception as e:
                logger.warning(f"Could not load metadata from {f}: {e}")
                continue
        
        if count == 0:
            logger.info(f"No performance trend data found for {symbol}")
            return None
        
        # Reverse to get chronological order (oldest first)
        return {
            'timestamps': timestamps[::-1],
            'rmse_values': rmse_values[::-1],
            'mae_values': mae_values[::-1],
            'count': count
        }
    
    def cleanup_old_versions(self, keep_latest=5):
        """
        Remove old model versions, keeping only the latest N
        Enhanced with better organization by symbol and model type
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
                        # Delete pickle
                        pkl_path = os.path.join(self.model_dir, old_file)
                        if os.path.exists(pkl_path):
                            os.remove(pkl_path)
                        
                        # Delete metadata
                        meta_file = old_file.replace('.pkl', '_meta.json')
                        meta_path = os.path.join(self.model_dir, meta_file)
                        if os.path.exists(meta_path):
                            os.remove(meta_path)
                        
                        deleted += 1
                        logger.info(f"Deleted old version: {old_file}")
                    except Exception as e:
                        logger.error(f"Error deleting {old_file}: {e}")
        
        if deleted > 0:
            logger.info(f"✓ Cleaned up {deleted} old model versions (kept latest {keep_latest})")
        else:
            logger.info(f"No cleanup needed (all groups have ≤{keep_latest} versions)")
        
        return deleted