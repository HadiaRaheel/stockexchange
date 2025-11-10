from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback
import logging

# Import database (using SQLite)
from database import SQLDatabase as Database

# Import data fetcher
from data_fetcher import DataFetcher

# Import models
from models.arima_model import ARIMAForecaster 
from models.lstm_model import LSTMForecaster
from models.ensemble import EnsembleForecaster

from models.adaptive_learner import AdaptiveLearner
from models.online_learner import OnlineLSTM, AdaptiveEnsemble


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Flask app
app = Flask(__name__)
CORS(app)

# Initialize components
db = Database()
data_fetcher = DataFetcher()

# Initialize models
arima_model = ARIMAForecaster()
lstm_model = LSTMForecaster()
ensemble_model = EnsembleForecaster()

adaptive_learner = AdaptiveLearner()
online_lstm = OnlineLSTM()

# Minimum data points required
MIN_DATA_POINTS = 50


def validate_data(data, min_points=MIN_DATA_POINTS):
    """Validate data quality and quantity"""
    if data is None or len(data) < min_points:
        return False, f"Insufficient data: need at least {min_points} points, got {len(data) if data is not None else 0}"
    
    if np.any(np.isnan(data)):
        return False, "Data contains NaN values"
    
    if np.any(np.isinf(data)):
        return False, "Data contains infinite values"
    
    return True, "Data is valid"


# def calculate_metrics(actual, predicted):
#     """
#     Calculate performance metrics safely
#     Returns dict with rmse, mae, mape or None if calculation fails
#     """
#     try:
#         # Ensure arrays are same length
#         min_len = min(len(actual), len(predicted))
#         actual = np.array(actual[:min_len])
#         predicted = np.array(predicted[:min_len])
        
#         # Check for valid data
#         if len(actual) == 0:
#             return None
        
#         # Calculate RMSE
#         rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        
#         # Calculate MAE
#         mae = float(np.mean(np.abs(actual - predicted)))
        
#         # Calculate MAPE (handle division by zero)
#         mask = np.abs(actual) > 1e-8
#         if np.any(mask):
#             mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
#         else:
#             mape = None
        
#         return {
#             'rmse': rmse,
#             'mae': mae,
#             'mape': mape
#         }
#     except Exception as e:
#         logger.error(f"Error calculating metrics: {e}")
#         return None

def calculate_metrics(actual, predicted):
    """
    Calculate performance metrics safely with better error handling
    Returns dict with rmse, mae, mape or None if calculation fails
    """
    try:
        # Convert to numpy arrays
        actual = np.array(actual, dtype=np.float64)
        predicted = np.array(predicted, dtype=np.float64)
        
        # Ensure arrays are same length
        if len(actual) != len(predicted):
            logger.warning(f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}")
            min_len = min(len(actual), len(predicted))
            actual = actual[:min_len]
            predicted = predicted[:min_len]
        
        # Check for valid data
        if len(actual) == 0:
            logger.error("Empty arrays passed to calculate_metrics")
            return None
        
        # Check for NaN or Inf
        if np.any(np.isnan(actual)) or np.any(np.isnan(predicted)):
            logger.error("NaN values detected in metrics calculation")
            return None
        
        if np.any(np.isinf(actual)) or np.any(np.isinf(predicted)):
            logger.error("Inf values detected in metrics calculation")
            return None
        
        # Calculate RMSE
        mse = np.mean((actual - predicted) ** 2)
        rmse = float(np.sqrt(mse))
        
        # Calculate MAE
        mae = float(np.mean(np.abs(actual - predicted)))
        
        # Calculate MAPE (handle division by zero)
        mask = np.abs(actual) > 1e-8
        if np.any(mask):
            mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
        else:
            mape = 0.0
        
        # Sanity check on values
        if rmse > 1e6 or mae > 1e6 or mape > 1e6:
            logger.warning(f"Suspiciously large metrics: RMSE={rmse}, MAE={mae}, MAPE={mape}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        logger.error(traceback.format_exc())
        return None


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/stocks', methods=['GET'])
def get_available_stocks():
    """Return list of available stocks"""
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'BTC-USD', 'ETH-USD', 'EURUSD=X']
    return jsonify({'stocks': stocks})


@app.route('/api/historical/<symbol>', methods=['GET'])
def get_historical_data(symbol):
    """Fetch historical data for a symbol"""
    try:
        # Validate symbol
        if not symbol or len(symbol) > 20:
            return jsonify({'success': False, 'error': 'Invalid symbol'}), 400
        
        # Get data from database first
        cached_data = db.get_historical_data(symbol, limit=200)
        
        if not cached_data or len(cached_data) == 0:
            logger.info(f"Fetching fresh data for {symbol}...")
            
            # Use data fetcher
            df = data_fetcher.fetch_historical_data(symbol, period='1y', interval='1d')
            
            if df is None or df.empty:
                return jsonify({
                    'success': False, 
                    'error': f'No data found for {symbol}. Please check the symbol and try again.'
                }), 404
            
            # Store in database
            data_list = []
            for idx, row in df.iterrows():
                try:
                    data_point = {
                        'symbol': symbol,
                        'timestamp': idx.isoformat(),
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': int(row['Volume']) if not pd.isna(row.get('Volume')) else 0
                    }
                    data_list.append(data_point)
                except Exception as e:
                    logger.warning(f"Skipping malformed row: {e}")
                    continue
            
            if len(data_list) == 0:
                return jsonify({
                    'success': False, 
                    'error': f'No valid data could be retrieved for {symbol}'
                }), 404
            
            db.store_historical_data(data_list)
            cached_data = data_list
            logger.info(f"Stored {len(data_list)} data points for {symbol}")
        else:
            logger.info(f"Using {len(cached_data)} cached data points for {symbol}")
        
        # Format for candlestick chart (last 100 points)
        candlestick_data = [{
            'x': item['timestamp'],
            'open': item['open'],
            'high': item['high'],
            'low': item['low'],
            'close': item['close']
        } for item in cached_data[-100:]]
        
        return jsonify({
            'success': True,
            'data': candlestick_data
        })
    
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False, 
            'error': f'Server error while fetching data: {str(e)}'
        }), 500

# # NEW ROUTE 1: Incremental model update
# @app.route('/api/update-model', methods=['POST'])
# def update_model_incremental():
#     """
#     Incrementally update model with new data
#     """
#     try:
#         data = request.json
        
#         symbol = data.get('symbol')
#         model_type = data.get('model', 'lstm')
        
#         if not symbol:
#             return jsonify({'success': False, 'error': 'Symbol is required'}), 400
        
#         logger.info(f"Performing incremental update for {symbol} ({model_type})")
        
#         # Fetch latest data
#         df = data_fetcher.fetch_historical_data(symbol, period='1mo', interval='1d')
        
#         if df is None or df.empty:
#             return jsonify({
#                 'success': False,
#                 'error': 'No data available for update'
#             }), 404
        
#         new_data = df['Close'].values
        
#         # Perform incremental update
#         if model_type == 'lstm':
#             updated_model = adaptive_learner.incremental_update(
#                 lstm_model, 
#                 new_data, 
#                 model_type='lstm'
#             )
#             validation_size = min(len(new_data), MIN_DATA_POINTS)
#             train_data = new_data[:-validation_size]
#             validation_data = new_data[-validation_size:]
            
#             try:
#                 validation_pred = lstm_model.predict(train_data, validation_size)
#                 metrics = calculate_metrics(validation_data, validation_pred)
#             except Exception as e:
#                 logger.warning(f"Metrics calculation failed: {e}")
#                 metrics = None
#             if metrics is None:
#                 metrics = {'rmse': None, 'mae': None, 'mape': None}
#             else:
#                 for key in ['rmse', 'mae', 'mape']:
#                     if key not in metrics:
#                         metrics[key] = None
#                         # Save updated version
#             version = adaptive_learner.save_model_version(
#                 updated_model,
#                 'LSTM',
#                 symbol,
#                 metrics 
#             )
            
#             return jsonify({
#                 'success': True,
#                 'message': f'Model updated incrementally',
#                 'version': version,
#                 'data_points': len(new_data)
#             })
        
#         elif model_type == 'ensemble':
#             # Update ensemble weights adaptively
#             return jsonify({
#                 'success': True,
#                 'message': 'Ensemble weights will be updated on next prediction'
#             })
        
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': f'Incremental update not supported for {model_type}'
#             }), 400
    
#     except Exception as e:
#         logger.error(f"Update error: {e}")
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 500

@app.route('/api/update-model', methods=['POST'])
def update_model_incremental():
    """
    Incrementally update model with new data
    """
    try:
        data = request.json
        
        symbol = data.get('symbol')
        model_type = data.get('model', 'lstm')
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'}), 400
        
        logger.info(f"Performing incremental update for {symbol} ({model_type})")
        
        # Fetch latest data
        df = data_fetcher.fetch_historical_data(symbol, period='6mo', interval='1d')  # Increased period
        
        if df is None or df.empty:
            return jsonify({
                'success': False,
                'error': 'No data available for update'
            }), 404
        
        new_data = df['Close'].values
        
        # Ensure we have enough data
        if len(new_data) < MIN_DATA_POINTS * 2:
            return jsonify({
                'success': False,
                'error': f'Insufficient data: need at least {MIN_DATA_POINTS * 2} points, got {len(new_data)}'
            }), 400
        
        # Perform incremental update
        if model_type == 'lstm':
            # Calculate proper validation size (20% of data, minimum 30 points)
            validation_size = max(30, min(len(new_data) // 5, 60))
            train_data = new_data[:-validation_size]
            validation_data = new_data[-validation_size:]
            
            logger.info(f"Train size: {len(train_data)}, Validation size: {len(validation_data)}")
            
            # Validate we have enough training data
            if len(train_data) < MIN_DATA_POINTS:
                return jsonify({
                    'success': False,
                    'error': f'Insufficient training data after split'
                }), 400
            
            # Perform incremental update
            updated_model = adaptive_learner.incremental_update(
                lstm_model, 
                new_data, 
                model_type='lstm'
            )
            
            # Calculate metrics with error handling
            metrics = None
            try:
                logger.info("Generating validation predictions...")
                validation_pred = updated_model.predict(train_data, validation_size)
                
                # Ensure predictions are valid
                if validation_pred is None or len(validation_pred) == 0:
                    logger.warning("Model returned empty predictions")
                elif len(validation_pred) != len(validation_data):
                    logger.warning(f"Prediction length mismatch: {len(validation_pred)} vs {len(validation_data)}")
                    # Truncate to shorter length
                    min_len = min(len(validation_pred), len(validation_data))
                    validation_pred = validation_pred[:min_len]
                    validation_data = validation_data[:min_len]
                    metrics = calculate_metrics(validation_data, validation_pred)
                else:
                    metrics = calculate_metrics(validation_data, validation_pred)
                
                if metrics:
                    logger.info(f"Calculated metrics - RMSE: ${metrics.get('rmse', 0):.2f}, MAE: ${metrics.get('mae', 0):.2f}")
                else:
                    logger.warning("Metrics calculation returned None")
                    
            except Exception as e:
                logger.error(f"Metrics calculation failed: {e}")
                logger.error(traceback.format_exc())
            
            # Only save model version if we have valid metrics
            if metrics and metrics.get('rmse') is not None:
                # Ensure all required keys are present
                if 'mae' not in metrics:
                    metrics['mae'] = metrics['rmse']  # Use RMSE as fallback
                if 'mape' not in metrics or metrics['mape'] is None:
                    metrics['mape'] = 0.0  # Default value
                
                version = adaptive_learner.save_model_version(
                    updated_model,
                    'LSTM',
                    symbol,
                    metrics 
                )
                
                # Log performance
                adaptive_learner.log_performance(
                    'LSTM',
                    symbol,
                    metrics,
                    version=version
                )
                
                return jsonify({
                    'success': True,
                    'message': f'Model updated incrementally',
                    'version': version,
                    'data_points': len(new_data),
                    'metrics': metrics
                })
            else:
                # Don't save if metrics are invalid
                return jsonify({
                    'success': False,
                    'error': 'Model update completed but metrics calculation failed',
                    'message': 'Try again with more data or check data quality'
                }), 400
        
        elif model_type == 'ensemble':
            # Update ensemble weights adaptively
            return jsonify({
                'success': True,
                'message': 'Ensemble weights will be updated on next prediction'
            })
        
        else:
            return jsonify({
                'success': False,
                'error': f'Incremental update not supported for {model_type}'
            }), 400
    
    except Exception as e:
        logger.error(f"Update error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500









# NEW ROUTE 2: Get performance history
@app.route('/api/performance-history/<symbol>', methods=['GET'])
def get_performance_history(symbol):
    """
    Get performance history for a symbol
    """
    try:
        days = int(request.args.get('days', 30))
        model_name = request.args.get('model', None)
        
        trend = adaptive_learner.get_performance_trend(
            symbol, 
            model_name=model_name,
            days=days
        )
        
        if trend is None:
            return jsonify({
                'success': True,
                'data': [],
                'message': 'No performance history available'
            })
        
        return jsonify({
            'success': True,
            'data': {
                'timestamps': trend['timestamps'],
                'rmse_values': trend['rmse_values'],
                'count': trend['count']
            }
        })
    
    except Exception as e:
        logger.error(f"Performance history error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# NEW ROUTE 3: Check if retrain needed
@app.route('/api/check-retrain', methods=['POST'])
def check_retrain_needed():
    """
    Check if model should be retrained based on performance
    """
    try:
        data = request.json
        symbol = data.get('symbol')
        current_metrics = data.get('metrics', {})
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol required'}), 400
        
        should_retrain = adaptive_learner.should_retrain(
            symbol,
            current_metrics
        )
        
        return jsonify({
            'success': True,
            'should_retrain': should_retrain,
            'reason': 'Performance degradation detected' if should_retrain else 'Performance acceptable'
        })
    
    except Exception as e:
        logger.error(f"Check retrain error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# NEW ROUTE 4: Online learning prediction
@app.route('/api/online-forecast', methods=['POST'])
def online_forecast():
    """
    Generate forecast using online learning model
    """
    try:
        data = request.json
        symbol = data.get('symbol')
        horizon = int(data.get('horizon', 24))
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol required'}), 400
        
        # Get historical data
        df = data_fetcher.fetch_historical_data(symbol, period='3mo', interval='1h')
        
        if df is None or df.empty:
            return jsonify({
                'success': False,
                'error': 'No data available'
            }), 404
        
        prices = df['Close'].values
        
        # Use online LSTM
        forecast = online_lstm.predict(prices, horizon)
        
        # Create timestamps
        last_timestamp = df.index[-1]
        forecast_timestamps = []
        
        for i in range(1, horizon + 1):
            timestamp = last_timestamp + timedelta(hours=i)
            forecast_timestamps.append(timestamp.isoformat())
        
        forecast_points = [{
            'x': forecast_timestamps[i],
            'y': float(forecast[i])
        } for i in range(len(forecast))]
        
        return jsonify({
            'success': True,
            'forecast': forecast_points,
            'model': 'Online-LSTM',
            'buffer_size': len(online_lstm.data_buffer)
        })
    
    except Exception as e:
        logger.error(f"Online forecast error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# NEW ROUTE 5: Adaptive ensemble forecast
@app.route('/api/adaptive-ensemble', methods=['POST'])
def adaptive_ensemble_forecast():
    """
    Generate forecast using adaptive ensemble with dynamic weights
    """
    try:
        data = request.json
        symbol = data.get('symbol')
        horizon = int(data.get('horizon', 24))
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol required'}), 400
        
        logger.info(f"Generating adaptive ensemble forecast for {symbol}")
        
        # Get data
        interval = '1h' if horizon <= 24 else '1d'
        df = data_fetcher.fetch_historical_data(symbol, period='1y', interval=interval)
        
        if df is None or df.empty:
            return jsonify({
                'success': False,
                'error': 'No data available'
            }), 404
        
        prices = df['Close'].values
        
        # Validate
        is_valid, message = validate_data(prices)
        if not is_valid:
            return jsonify({'success': False, 'error': message}), 400
        
        # Create adaptive ensemble
        models_dict = {
            'arima': arima_model,
            'lstm': lstm_model
        }
        
        adaptive_ens = AdaptiveEnsemble(models_dict)
        
        # If we have recent actual values, update weights
        if len(prices) > horizon:
            # Use last horizon points to update weights
            validation_data = prices[-horizon:]
            train_data = prices[:-horizon]
            
            predictions_dict = {}
            for name, model in models_dict.items():
                try:
                    pred = model.predict(train_data, horizon)
                    predictions_dict[name] = pred[0] if len(pred) > 0 else train_data[-1]
                except:
                    predictions_dict[name] = train_data[-1]
            
            # Update weights based on first prediction
            adaptive_ens.update_weights(predictions_dict, validation_data[0])
        
        # Generate forecast
        forecast = adaptive_ens.predict(prices, horizon)
        
        # Create response
        last_timestamp = df.index[-1]
        forecast_timestamps = []
        
        for i in range(1, horizon + 1):
            if horizon <= 24:
                timestamp = last_timestamp + timedelta(hours=i)
            else:
                timestamp = last_timestamp + timedelta(days=i)
            forecast_timestamps.append(timestamp.isoformat())
        
        forecast_points = [{
            'x': forecast_timestamps[i],
            'y': float(forecast[i])
        } for i in range(len(forecast))]
        
        return jsonify({
            'success': True,
            'forecast': forecast_points,
            'model': 'Adaptive-Ensemble',
            'weights': adaptive_ens.get_weights()
        })
    
    except Exception as e:
        logger.error(f"Adaptive ensemble error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# NEW ROUTE 6: Model versions management
@app.route('/api/model-versions/<symbol>', methods=['GET'])
def get_model_versions(symbol):
    """
    Get all saved versions for a symbol
    """
    try:
        import os
        import json
        
        model_dir = adaptive_learner.model_dir
        versions = []
        
        for file in os.listdir(model_dir):
            if file.endswith('_meta.json') and symbol in file:
                with open(os.path.join(model_dir, file), 'r') as f:
                    metadata = json.load(f)
                    versions.append(metadata)
        
        # Sort by timestamp
        versions.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'success': True,
            'versions': versions,
            'count': len(versions)
        })
    
    except Exception as e:
        logger.error(f"Model versions error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# NEW ROUTE 7: Cleanup old models
@app.route('/api/cleanup-models', methods=['POST'])
def cleanup_old_models():
    """
    Clean up old model versions
    """
    try:
        keep_latest = int(request.json.get('keep_latest', 5))
        
        adaptive_learner.cleanup_old_versions(keep_latest=keep_latest)
        
        return jsonify({
            'success': True,
            'message': f'Cleaned up old models, keeping latest {keep_latest} versions'
        })
    
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/forecast', methods=['POST'])
def generate_forecast():
    """Generate forecast for given symbol and horizon"""
    try:
        data = request.json
        
        # Validate input
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        symbol = data.get('symbol')
        horizon = data.get('horizon')
        model_type = data.get('model', 'ensemble')
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'}), 400
        
        try:
            horizon = int(horizon)
            if horizon <= 0 or horizon > 168:  # Max 1 week
                return jsonify({'success': False, 'error': 'Horizon must be between 1 and 168 hours'}), 400
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'Invalid horizon value'}), 400
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Generating {model_type.upper()} forecast")
        logger.info(f"Symbol: {symbol} | Horizon: {horizon}h")
        logger.info(f"{'='*50}")
        
        # Get historical data
        interval = '1h' if horizon <= 24 else '1d'
        df = data_fetcher.fetch_historical_data(symbol, period='1y', interval=interval)
        
        if df is None or df.empty:
            return jsonify({
                'success': False, 
                'error': f'No historical data available for {symbol}'
            }), 404
        
        # Prepare data
        prices = df['Close'].values
        
        # Validate data
        is_valid, message = validate_data(prices)
        if not is_valid:
            return jsonify({'success': False, 'error': message}), 400
        
        logger.info(f"Using {len(prices)} historical prices")
        logger.info(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        logger.info(f"Last price: ${prices[-1]:.2f}")
        
        # Generate forecasts based on model type
        logger.info(f"\n  Training {model_type.upper()} model...")
        
        try:
            if model_type == 'arima':
                forecast = arima_model.predict(prices, horizon)
                model_name = 'ARIMA'
            elif model_type == 'lstm':
                forecast = lstm_model.predict(prices, horizon)
                model_name = 'LSTM'
            else:  # ensemble
                forecast = ensemble_model.predict(prices, horizon)
                model_name = 'Ensemble'
        except Exception as model_error:
            logger.error(f"Model {model_type} failed: {model_error}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False, 
                'error': f'Model training failed: {str(model_error)}'
            }), 500
        
        # Validate forecast
        if forecast is None or len(forecast) == 0:
            return jsonify({
                'success': False, 
                'error': 'Model failed to generate valid predictions'
            }), 500
        
        # Ensure forecast has correct length
        if len(forecast) != horizon:
            logger.warning(f"Forecast length mismatch: expected {horizon}, got {len(forecast)}")
            if len(forecast) < horizon:
                # Pad with last value
                forecast = np.concatenate([forecast, np.full(horizon - len(forecast), forecast[-1])])
            else:
                # Truncate
                forecast = forecast[:horizon]
        
        logger.info(f"Forecast generated: {len(forecast)} points")
        logger.info(f"Predicted range: ${forecast.min():.2f} - ${forecast.max():.2f}")
        
        # Calculate metrics using walk-forward validation approach
        # Use the last 'horizon' points from historical data as validation
        if len(prices) > horizon:
            # Split: use all but last 'horizon' points for training, last 'horizon' for validation
            validation_size = min(horizon, len(prices) // 5)  # Max 20% for validation
            train_prices = prices[:-validation_size]
            validation_prices = prices[-validation_size:]
            
            # Generate predictions on validation set
            try:
                if model_type == 'arima':
                    validation_pred = arima_model.predict(train_prices, validation_size)
                elif model_type == 'lstm':
                    validation_pred = lstm_model.predict(train_prices, validation_size)
                else:
                    validation_pred = ensemble_model.predict(train_prices, validation_size)
                
                metrics = calculate_metrics(validation_prices, validation_pred)
            except Exception as e:
                logger.warning(f"Could not calculate validation metrics: {e}")
                metrics = None
        else:
            metrics = None
        
        if metrics:
            logger.info(f"\n Validation Metrics:")
            logger.info(f"  RMSE: ${metrics['rmse']:.2f}")
            logger.info(f"  MAE: ${metrics['mae']:.2f}")
            if metrics['mape']:
                logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        else:
            logger.info("\n Metrics: Not available (insufficient validation data)")
            metrics = {'rmse': None, 'mae': None, 'mape': None}
        
        logger.info(f"{'='*50}\n")
        
        # Create forecast timestamps
        last_timestamp = df.index[-1]
        forecast_timestamps = []
        
        for i in range(1, horizon + 1):
            if horizon <= 24:
                timestamp = last_timestamp + timedelta(hours=i)
            else:
                timestamp = last_timestamp + timedelta(days=i)
            forecast_timestamps.append(timestamp.isoformat())
        
        # Store forecast in database
        forecast_data = {
            'symbol': symbol,
            'model': model_name,
            'horizon': horizon,
            'timestamp': datetime.now().isoformat(),
            'predictions': [float(x) for x in forecast],
            'metrics': metrics
        }
        
        try:
            db.store_forecast(forecast_data)
        except Exception as db_error:
            logger.warning(f"Could not store forecast in database: {db_error}")
        
        # Format response
        forecast_points = [{
            'x': forecast_timestamps[i],
            'y': float(forecast[i])
        } for i in range(len(forecast))]
        
        return jsonify({
            'success': True,
            'forecast': forecast_points,
            'metrics': metrics,
            'model': model_name
        })
    
    except Exception as e:
        logger.error(f"\nForecast error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False, 
            'error': f'An unexpected error occurred: {str(e)}'
        }), 500


@app.route('/api/compare-models', methods=['POST'])
def compare_models():
    """Compare performance of different models"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        symbol = data.get('symbol')
        horizon = data.get('horizon')
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'}), 400
        
        try:
            horizon = int(horizon)
            if horizon <= 0 or horizon > 168:
                return jsonify({'success': False, 'error': 'Horizon must be between 1 and 168 hours'}), 400
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'Invalid horizon value'}), 400
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Comparing all models for {symbol}")
        logger.info(f"Horizon: {horizon}h")
        logger.info(f"{'='*50}\n")
        
        # Get historical data
        interval = '1h' if horizon <= 24 else '1d'
        df = data_fetcher.fetch_historical_data(symbol, period='1y', interval=interval)
        
        if df is None or df.empty:
            return jsonify({
                'success': False, 
                'error': f'No historical data available for {symbol}'
            }), 404
        
        prices = df['Close'].values
        
        # Validate data
        is_valid, message = validate_data(prices)
        if not is_valid:
            return jsonify({'success': False, 'error': message}), 400
        
        # Split data for comparison (use same approach as individual forecast)
        test_size = min(horizon, len(prices) // 5)
        train_size = len(prices) - test_size
        
        if train_size < MIN_DATA_POINTS:
            return jsonify({
                'success': False, 
                'error': f'Insufficient data for comparison (need {MIN_DATA_POINTS + test_size} points minimum)'
            }), 400
        
        train = prices[:train_size]
        test = prices[train_size:]
        
        logger.info(f"Training set: {len(train)} points")
        logger.info(f"Test set: {len(test)} points\n")
        
        results = {}
        
        # Test each model
        models = [
            ('ARIMA', arima_model),
            ('LSTM', lstm_model),
            ('Ensemble', ensemble_model)
        ]
        
        for name, model in models:
            logger.info(f"  Testing {name}...")
            try:
                pred = model.predict(train, len(test))
                
                # Ensure same length
                pred = pred[:len(test)]
                
                metrics = calculate_metrics(test, pred)
                
                if metrics:
                    results[name] = metrics
                    logger.info(f"  ✓ {name} RMSE: ${metrics['rmse']:.2f}")
                else:
                    results[name] = {'rmse': 999, 'mae': 999, 'mape': 999}
                    logger.warning(f" {name} metrics calculation failed")
                    
            except Exception as e:
                logger.error(f" {name} failed: {e}")
                results[name] = {'rmse': 999, 'mae': 999, 'mape': 999}
        
        logger.info(f"\n✓ Comparison complete!")
        logger.info(f"{'='*50}\n")
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        logger.error(f"\nComparison error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False, 
            'error': f'An unexpected error occurred: {str(e)}'
        }), 500

@app.route('/api/forecast-enhanced', methods=['POST'])
def generate_forecast_enhanced():
    """
    Enhanced forecast with adaptive learning features
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        symbol = data.get('symbol')
        horizon = int(data.get('horizon'))
        model_type = data.get('model', 'ensemble')
        use_adaptive = data.get('adaptive', False)
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'}), 400
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Enhanced {model_type.upper()} forecast (Adaptive: {use_adaptive})")
        logger.info(f"Symbol: {symbol} | Horizon: {horizon}h")
        logger.info(f"{'='*50}")
        
        # Get historical data
        interval = '1h' if horizon <= 24 else '1d'
        df = data_fetcher.fetch_historical_data(symbol, period='1y', interval=interval)
        
        if df is None or df.empty:
            return jsonify({
                'success': False,
                'error': f'No historical data available for {symbol}'
            }), 404
        
        prices = df['Close'].values
        
        # Validate data
        is_valid, message = validate_data(prices)
        if not is_valid:
            return jsonify({'success': False, 'error': message}), 400
        
        logger.info(f"Using {len(prices)} historical prices")
        
        # Check if we should use best saved version
        best_version, best_score = adaptive_learner.get_best_model_version(
            model_type.upper(),
            symbol
        )
        
        if best_version and use_adaptive:
            logger.info(f"Using best model version: {best_version} (RMSE: {best_score:.2f})")
        
        # Generate forecast
        try:
            if model_type == 'arima':
                forecast = arima_model.predict(prices, horizon)
                model_name = 'ARIMA'
            elif model_type == 'lstm':
                forecast = lstm_model.predict(prices, horizon)
                model_name = 'LSTM'
            else:  # ensemble
                forecast = ensemble_model.predict(prices, horizon)
                model_name = 'Ensemble'
        except Exception as model_error:
            logger.error(f"Model {model_type} failed: {model_error}")
            return jsonify({
                'success': False,
                'error': f'Model training failed: {str(model_error)}'
            }), 500
        
        # Validate forecast
        if forecast is None or len(forecast) == 0:
            return jsonify({
                'success': False,
                'error': 'Model failed to generate valid predictions'
            }), 500
        
        # Calculate metrics
        if len(prices) > horizon:
            validation_size = min(horizon, len(prices) // 5)
            train_prices = prices[:-validation_size]
            validation_prices = prices[-validation_size:]
            
            try:
                if model_type == 'arima':
                    validation_pred = arima_model.predict(train_prices, validation_size)
                elif model_type == 'lstm':
                    validation_pred = lstm_model.predict(train_prices, validation_size)
                else:
                    validation_pred = ensemble_model.predict(train_prices, validation_size)
                
                metrics = calculate_metrics(validation_prices, validation_pred)
                
                # Log performance
                if use_adaptive and metrics:
                    adaptive_learner.log_performance(
                        model_name,
                        symbol,
                        metrics
                    )
                    
                    # Check if retrain needed
                    if adaptive_learner.should_retrain(symbol, metrics):
                        logger.warning("Performance degradation detected - retrain recommended")
            except Exception as e:
                logger.warning(f"Could not calculate validation metrics: {e}")
                metrics = None
        else:
            metrics = None
        
        if metrics:
            logger.info(f"\nValidation Metrics:")
            logger.info(f"  RMSE: ${metrics['rmse']:.2f}")
            logger.info(f"  MAE: ${metrics['mae']:.2f}")
            if metrics['mape']:
                logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        else:
            metrics = {'rmse': None, 'mae': None, 'mape': None}
        
        # Create timestamps
        last_timestamp = df.index[-1]
        forecast_timestamps = []
        
        for i in range(1, horizon + 1):
            if horizon <= 24:
                timestamp = last_timestamp + timedelta(hours=i)
            else:
                timestamp = last_timestamp + timedelta(days=i)
            forecast_timestamps.append(timestamp.isoformat())
        
        # Save model version if adaptive learning enabled
        if use_adaptive and metrics:
            version = adaptive_learner.save_model_version(
                lstm_model if model_type == 'lstm' else ensemble_model,
                model_name,
                symbol,
                metrics
            )
            logger.info(f"Saved model version: {version}")
        
        # Format response
        forecast_points = [{
            'x': forecast_timestamps[i],
            'y': float(forecast[i])
        } for i in range(len(forecast))]
        
        return jsonify({
            'success': True,
            'forecast': forecast_points,
            'metrics': metrics,
            'model': model_name,
            'adaptive_learning': use_adaptive,
            'best_version': best_version if use_adaptive else None
        })
    
    except Exception as e:
        logger.error(f"\nForecast error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'An unexpected error occurred: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("="*50)
    logger.info("Stock Forecasting Application Starting...")
    logger.info("="*50)
    logger.info(f"Template folder: {app.template_folder}")
    logger.info(f"Server: http://localhost:5000")
    logger.info("="*50)
    app.run(debug=True, port=5000)