"""
Continuous Evaluation and Monitoring System
Place this file in: models/continuous_monitor.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import json
import os
import logging

logger = logging.getLogger(__name__)


class ContinuousMonitor:
    """
    Monitors model performance continuously as new ground-truth data arrives
    """
    
    def __init__(self, storage_dir='monitoring_data'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Storage for predictions and actual values
        self.prediction_store_file = os.path.join(storage_dir, 'predictions.json')
        self.metrics_store_file = os.path.join(storage_dir, 'continuous_metrics.json')
        
        # In-memory stores
        self.prediction_store = self._load_prediction_store()
        self.metrics_history = self._load_metrics_history()
        
        # Real-time metrics buffer
        self.recent_errors = deque(maxlen=100)
        
    def _load_prediction_store(self):
        """Load stored predictions from disk"""
        try:
            if os.path.exists(self.prediction_store_file):
                with open(self.prediction_store_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading prediction store: {e}")
            return {}
    
    def _save_prediction_store(self):
        """Save predictions to disk"""
        try:
            with open(self.prediction_store_file, 'w') as f:
                json.dump(self.prediction_store, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving prediction store: {e}")
    
    def _load_metrics_history(self):
        """Load metrics history from disk"""
        try:
            if os.path.exists(self.metrics_store_file):
                with open(self.metrics_store_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading metrics history: {e}")
            return []
    
    def _save_metrics_history(self):
        """Save metrics history to disk"""
        try:
            with open(self.metrics_store_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics history: {e}")
    
    def store_prediction(self, symbol, model_name, prediction_time, 
                        target_time, predicted_value):
        """
        Store a prediction for future evaluation
        
        Args:
            symbol: Stock symbol
            model_name: Name of the model
            prediction_time: When the prediction was made
            target_time: The time being predicted
            predicted_value: The predicted price
        """
        key = f"{symbol}_{target_time}"
        
        if key not in self.prediction_store:
            self.prediction_store[key] = []
        
        self.prediction_store[key].append({
            'symbol': symbol,
            'model': model_name,
            'prediction_time': prediction_time,
            'target_time': target_time,
            'predicted_value': float(predicted_value),
            'actual_value': None,
            'evaluated': False
        })
        
        self._save_prediction_store()
        logger.info(f"Stored prediction for {symbol} at {target_time}")
    
    def evaluate_predictions(self, symbol, actual_data):
        """
        Evaluate stored predictions against actual data
        
        Args:
            symbol: Stock symbol
            actual_data: DataFrame with timestamp index and 'Close' column
        
        Returns:
            dict with evaluation results
        """
        evaluated_count = 0
        results = []
        
        for key, predictions in self.prediction_store.items():
            if not key.startswith(symbol):
                continue
            
            for pred in predictions:
                if pred['evaluated']:
                    continue
                
                target_time = pd.Timestamp(pred['target_time'])
                
                # Find matching actual value
                if target_time in actual_data.index:
                    actual_value = actual_data.loc[target_time, 'Close']
                    
                    # Calculate error
                    error = abs(pred['predicted_value'] - actual_value)
                    pct_error = (error / actual_value) * 100 if actual_value != 0 else 0
                    
                    # Update prediction
                    pred['actual_value'] = float(actual_value)
                    pred['evaluated'] = True
                    pred['error'] = float(error)
                    pred['pct_error'] = float(pct_error)
                    pred['evaluation_time'] = datetime.now().isoformat()
                    
                    # Store in recent errors
                    self.recent_errors.append({
                        'symbol': symbol,
                        'model': pred['model'],
                        'error': error,
                        'pct_error': pct_error,
                        'time': pred['evaluation_time']
                    })
                    
                    results.append(pred)
                    evaluated_count += 1
                    
                    logger.info(
                        f"Evaluated prediction for {symbol}: "
                        f"Predicted={pred['predicted_value']:.2f}, "
                        f"Actual={actual_value:.2f}, "
                        f"Error={error:.2f} ({pct_error:.2f}%)"
                    )
        
        if evaluated_count > 0:
            self._save_prediction_store()
            self._update_continuous_metrics(symbol, results)
        
        return {
            'evaluated_count': evaluated_count,
            'results': results
        }
    
    def _update_continuous_metrics(self, symbol, evaluated_predictions):
        """Update continuous metrics based on new evaluations"""
        if not evaluated_predictions:
            return
        
        # Group by model
        by_model = {}
        for pred in evaluated_predictions:
            model = pred['model']
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(pred)
        
        # Calculate metrics for each model
        timestamp = datetime.now().isoformat()
        
        for model, preds in by_model.items():
            errors = [p['error'] for p in preds]
            pct_errors = [p['pct_error'] for p in preds]
            
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean([e**2 for e in errors]))
            mape = np.mean(pct_errors)
            
            metric_entry = {
                'timestamp': timestamp,
                'symbol': symbol,
                'model': model,
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'sample_size': len(preds)
            }
            
            self.metrics_history.append(metric_entry)
            
            logger.info(
                f"Continuous metrics for {model} on {symbol}: "
                f"MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%"
            )
        
        self._save_metrics_history()
    
    def get_continuous_metrics(self, symbol=None, model=None, days=30):
        """
        Get continuous metrics history
        
        Args:
            symbol: Filter by symbol (optional)
            model: Filter by model (optional)
            days: Number of days to look back
        
        Returns:
            list of metric entries
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        filtered = []
        for entry in self.metrics_history:
            # Time filter
            entry_time = datetime.fromisoformat(entry['timestamp'])
            if entry_time < cutoff:
                continue
            
            # Symbol filter
            if symbol and entry['symbol'] != symbol:
                continue
            
            # Model filter
            if model and entry['model'] != model:
                continue
            
            filtered.append(entry)
        
        return filtered
    
    def get_recent_errors(self, limit=50):
        """Get recent prediction errors"""
        return list(self.recent_errors)[-limit:]
    
    def get_model_comparison(self, symbol, days=30):
        """
        Compare model performance over time
        
        Returns:
            dict with comparison data for each model
        """
        metrics = self.get_continuous_metrics(symbol=symbol, days=days)
        
        if not metrics:
            return None
        
        # Group by model
        by_model = {}
        for entry in metrics:
            model = entry['model']
            if model not in by_model:
                by_model[model] = {
                    'timestamps': [],
                    'mae': [],
                    'rmse': [],
                    'mape': []
                }
            
            by_model[model]['timestamps'].append(entry['timestamp'])
            by_model[model]['mae'].append(entry['mae'])
            by_model[model]['rmse'].append(entry['rmse'])
            by_model[model]['mape'].append(entry['mape'])
        
        # Calculate averages
        comparison = {}
        for model, data in by_model.items():
            comparison[model] = {
                'avg_mae': float(np.mean(data['mae'])),
                'avg_rmse': float(np.mean(data['rmse'])),
                'avg_mape': float(np.mean(data['mape'])),
                'data_points': len(data['mae']),
                'trend_data': data
            }
        
        return comparison
    
    def calculate_error_overlay(self, predictions, actuals, timestamps):
        """
        Calculate error overlay data for candlestick chart
        
        Args:
            predictions: array of predicted values
            actuals: array of actual values
            timestamps: array of timestamps
        
        Returns:
            dict with error overlay data
        """
        errors = np.array(predictions) - np.array(actuals)
        abs_errors = np.abs(errors)
        pct_errors = (errors / np.array(actuals)) * 100
        
        return {
            'timestamps': [str(t) for t in timestamps],
            'predictions': [float(p) for p in predictions],
            'actuals': [float(a) for a in actuals],
            'errors': [float(e) for e in errors],
            'abs_errors': [float(e) for e in abs_errors],
            'pct_errors': [float(e) for e in pct_errors],
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'mae': float(np.mean(abs_errors)),
            'mape': float(np.mean(np.abs(pct_errors)))
        }
    
    def get_prediction_accuracy_over_time(self, symbol, model=None):
        """
        Get prediction accuracy trend over time
        
        Returns:
            dict with accuracy trend data
        """
        # Get all evaluated predictions
        evaluated = []
        for key, predictions in self.prediction_store.items():
            if not key.startswith(symbol):
                continue
            
            for pred in predictions:
                if pred['evaluated']:
                    if model is None or pred['model'] == model:
                        evaluated.append(pred)
        
        if not evaluated:
            return None
        
        # Sort by evaluation time
        evaluated.sort(key=lambda x: x['evaluation_time'])
        
        # Calculate rolling accuracy
        window_size = 10
        rolling_mae = []
        rolling_mape = []
        timestamps = []
        
        for i in range(len(evaluated)):
            if i < window_size - 1:
                continue
            
            window = evaluated[i-window_size+1:i+1]
            mae = np.mean([p['error'] for p in window])
            mape = np.mean([p['pct_error'] for p in window])
            
            rolling_mae.append(float(mae))
            rolling_mape.append(float(mape))
            timestamps.append(window[-1]['evaluation_time'])
        
        return {
            'timestamps': timestamps,
            'rolling_mae': rolling_mae,
            'rolling_mape': rolling_mape,
            'window_size': window_size,
            'total_predictions': len(evaluated)
        }
    
    def detect_performance_degradation(self, symbol, model, threshold=0.20):
        """
        Detect if model performance has degraded significantly
        
        Args:
            symbol: Stock symbol
            model: Model name
            threshold: Degradation threshold (default 20%)
        
        Returns:
            dict with degradation analysis
        """
        metrics = self.get_continuous_metrics(symbol=symbol, model=model, days=30)
        
        if len(metrics) < 10:
            return {
                'degraded': False,
                'reason': 'Insufficient data for analysis'
            }
        
        # Compare recent vs historical performance
        sorted_metrics = sorted(metrics, key=lambda x: x['timestamp'])
        
        recent_size = max(5, len(sorted_metrics) // 4)
        recent = sorted_metrics[-recent_size:]
        historical = sorted_metrics[:-recent_size]
        
        recent_rmse = np.mean([m['rmse'] for m in recent])
        historical_rmse = np.mean([m['rmse'] for m in historical])
        
        if historical_rmse == 0:
            return {
                'degraded': False,
                'reason': 'Cannot calculate degradation'
            }
        
        degradation = (recent_rmse - historical_rmse) / historical_rmse
        
        return {
            'degraded': degradation > threshold,
            'degradation_pct': float(degradation * 100),
            'recent_rmse': float(recent_rmse),
            'historical_rmse': float(historical_rmse),
            'threshold': threshold * 100,
            'recommendation': 'Retrain model' if degradation > threshold else 'Performance acceptable'
        }
    
    def cleanup_old_data(self, days=90):
        """Remove predictions and metrics older than specified days"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Clean predictions
        removed_predictions = 0
        for key in list(self.prediction_store.keys()):
            self.prediction_store[key] = [
                p for p in self.prediction_store[key]
                if p['prediction_time'] > cutoff
            ]
            if not self.prediction_store[key]:
                del self.prediction_store[key]
                removed_predictions += 1
        
        # Clean metrics
        original_count = len(self.metrics_history)
        self.metrics_history = [
            m for m in self.metrics_history
            if m['timestamp'] > cutoff
        ]
        removed_metrics = original_count - len(self.metrics_history)
        
        self._save_prediction_store()
        self._save_metrics_history()
        
        logger.info(
            f"Cleaned up old data: "
            f"{removed_predictions} prediction groups, "
            f"{removed_metrics} metric entries"
        )
        
        return {
            'removed_predictions': removed_predictions,
            'removed_metrics': removed_metrics
        }