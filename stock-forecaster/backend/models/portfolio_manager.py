"""
Portfolio Management System
Place this file in: models/portfolio_manager.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import logging

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Simulated portfolio for trading stocks/cryptos
    """
    
    def __init__(self, initial_cash=100000.0, portfolio_id='default'):
        self.portfolio_id = portfolio_id
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # {symbol: {'quantity': int, 'avg_cost': float}}
        self.transaction_history = []
        self.portfolio_history = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def get_position(self, symbol):
        """Get current position for a symbol"""
        return self.positions.get(symbol, {'quantity': 0, 'avg_cost': 0.0})
    
    def buy(self, symbol, quantity, price, timestamp=None):
        """
        Execute buy order
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares to buy
            price: Price per share
            timestamp: Transaction timestamp
        
        Returns:
            dict with transaction details or None if failed
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        cost = quantity * price
        
        # Check if we have enough cash
        if cost > self.cash:
            logger.warning(f"Insufficient cash for buy: need ${cost:.2f}, have ${self.cash:.2f}")
            return None
        
        # Update cash
        self.cash -= cost
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'avg_cost': 0.0}
        
        current_pos = self.positions[symbol]
        total_quantity = current_pos['quantity'] + quantity
        
        # Calculate new average cost
        if total_quantity > 0:
            total_cost = (current_pos['quantity'] * current_pos['avg_cost']) + cost
            new_avg_cost = total_cost / total_quantity
        else:
            new_avg_cost = price
        
        self.positions[symbol] = {
            'quantity': total_quantity,
            'avg_cost': new_avg_cost
        }
        
        # Record transaction
        transaction = {
            'timestamp': timestamp,
            'action': 'BUY',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'value': cost,
            'cash_after': self.cash
        }
        
        self.transaction_history.append(transaction)
        self.total_trades += 1
        
        logger.info(f"✅ BUY {quantity} {symbol} @ ${price:.2f} = ${cost:.2f}")
        
        return transaction
    
    def sell(self, symbol, quantity, price, timestamp=None):
        """
        Execute sell order
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares to sell
            price: Price per share
            timestamp: Transaction timestamp
        
        Returns:
            dict with transaction details or None if failed
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Check if we have the position
        if symbol not in self.positions:
            logger.warning(f"Cannot sell {symbol}: no position")
            return None
        
        current_pos = self.positions[symbol]
        
        if current_pos['quantity'] < quantity:
            logger.warning(f"Insufficient shares: have {current_pos['quantity']}, trying to sell {quantity}")
            return None
        
        # Calculate proceeds
        proceeds = quantity * price
        
        # Calculate profit/loss
        cost_basis = quantity * current_pos['avg_cost']
        profit_loss = proceeds - cost_basis
        
        # Update cash
        self.cash += proceeds
        
        # Update position
        new_quantity = current_pos['quantity'] - quantity
        
        if new_quantity > 0:
            self.positions[symbol]['quantity'] = new_quantity
        else:
            # Close position
            del self.positions[symbol]
        
        # Record transaction
        transaction = {
            'timestamp': timestamp,
            'action': 'SELL',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'value': proceeds,
            'profit_loss': profit_loss,
            'cash_after': self.cash
        }
        
        self.transaction_history.append(transaction)
        self.total_trades += 1
        
        # Track win/loss
        if profit_loss > 0:
            self.winning_trades += 1
        elif profit_loss < 0:
            self.losing_trades += 1
        
        logger.info(f"✅ SELL {quantity} {symbol} @ ${price:.2f} = ${proceeds:.2f} (P/L: ${profit_loss:.2f})")
        
        return transaction
    
    def get_portfolio_value(self, current_prices):
        """
        Calculate total portfolio value
        
        Args:
            current_prices: dict of {symbol: current_price}
        
        Returns:
            float: total portfolio value
        """
        holdings_value = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                holdings_value += position['quantity'] * current_prices[symbol]
            else:
                # Use average cost if current price not available
                holdings_value += position['quantity'] * position['avg_cost']
        
        total_value = self.cash + holdings_value
        return total_value
    
    def get_positions_summary(self, current_prices):
        """
        Get summary of all positions
        
        Args:
            current_prices: dict of {symbol: current_price}
        
        Returns:
            list of position details
        """
        positions = []
        
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position['avg_cost'])
            market_value = position['quantity'] * current_price
            cost_basis = position['quantity'] * position['avg_cost']
            unrealized_pl = market_value - cost_basis
            unrealized_pl_pct = (unrealized_pl / cost_basis * 100) if cost_basis > 0 else 0
            
            positions.append({
                'symbol': symbol,
                'quantity': position['quantity'],
                'avg_cost': position['avg_cost'],
                'current_price': current_price,
                'market_value': market_value,
                'cost_basis': cost_basis,
                'unrealized_pl': unrealized_pl,
                'unrealized_pl_pct': unrealized_pl_pct
            })
        
        return positions
    
    def record_snapshot(self, current_prices, timestamp=None):
        """Record portfolio value at a point in time"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        total_value = self.get_portfolio_value(current_prices)
        
        snapshot = {
            'timestamp': timestamp,
            'total_value': total_value,
            'cash': self.cash,
            'holdings_value': total_value - self.cash,
            'positions_count': len(self.positions)
        }
        
        self.portfolio_history.append(snapshot)
        return snapshot
    
    def get_performance_metrics(self, current_prices):
        """
        Calculate portfolio performance metrics
        
        Returns:
            dict with performance metrics
        """
        current_value = self.get_portfolio_value(current_prices)
        
        # Total return
        total_return = current_value - self.initial_cash
        total_return_pct = (total_return / self.initial_cash) * 100
        
        # Win rate
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # Calculate returns over time for volatility and Sharpe
        if len(self.portfolio_history) > 1:
            values = [h['total_value'] for h in self.portfolio_history]
            returns = np.diff(values) / values[:-1]
            
            # Volatility (annualized)
            volatility = np.std(returns) * np.sqrt(252)
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            avg_return = np.mean(returns) * 252
            sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0
        else:
            volatility = 0.0
            sharpe_ratio = 0.0
        
        return {
            'current_value': current_value,
            'initial_value': self.initial_cash,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'cash': self.cash,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def to_dict(self):
        """Convert portfolio to dictionary for serialization"""
        return {
            'portfolio_id': self.portfolio_id,
            'initial_cash': self.initial_cash,
            'cash': self.cash,
            'positions': self.positions,
            'transaction_history': self.transaction_history,
            'portfolio_history': self.portfolio_history,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create portfolio from dictionary"""
        portfolio = cls(
            initial_cash=data['initial_cash'],
            portfolio_id=data['portfolio_id']
        )
        portfolio.cash = data['cash']
        portfolio.positions = data['positions']
        portfolio.transaction_history = data['transaction_history']
        portfolio.portfolio_history = data.get('portfolio_history', [])
        portfolio.total_trades = data.get('total_trades', 0)
        portfolio.winning_trades = data.get('winning_trades', 0)
        portfolio.losing_trades = data.get('losing_trades', 0)
        return portfolio


class TradingStrategy:
    """
    Base class for trading strategies
    """
    
    def __init__(self, name='strategy'):
        self.name = name
    
    def generate_signal(self, historical_data, forecast_data, current_price):
        """
        Generate trading signal
        
        Args:
            historical_data: Historical prices
            forecast_data: Forecasted prices
            current_price: Current market price
        
        Returns:
            str: 'BUY', 'SELL', or 'HOLD'
        """
        raise NotImplementedError


class ForecastBasedStrategy(TradingStrategy):
    """
    Strategy based on forecast predictions
    """
    
    def __init__(self, buy_threshold=0.03, sell_threshold=-0.02):
        super().__init__(name='ForecastBased')
        self.buy_threshold = buy_threshold  # Buy if forecast shows 3% gain
        self.sell_threshold = sell_threshold  # Sell if forecast shows 2% loss
    
    def generate_signal(self, historical_data, forecast_data, current_price):
        """
        Generate signal based on forecast
        """
        if len(forecast_data) == 0:
            return 'HOLD'
        
        # Calculate expected return from forecast
        forecast_price = forecast_data[0]  # Next period forecast
        expected_return = (forecast_price - current_price) / current_price
        
        if expected_return > self.buy_threshold:
            return 'BUY'
        elif expected_return < self.sell_threshold:
            return 'SELL'
        else:
            return 'HOLD'


class MomentumStrategy(TradingStrategy):
    """
    Momentum-based trading strategy
    """
    
    def __init__(self, window=20):
        super().__init__(name='Momentum')
        self.window = window
    
    def generate_signal(self, historical_data, forecast_data, current_price):
        """
        Generate signal based on momentum
        """
        if len(historical_data) < self.window:
            return 'HOLD'
        
        recent = historical_data[-self.window:]
        sma = np.mean(recent)
        
        # Buy if price above moving average and rising
        if current_price > sma * 1.02:
            return 'BUY'
        # Sell if price below moving average
        elif current_price < sma * 0.98:
            return 'SELL'
        else:
            return 'HOLD'


class PortfolioManager:
    """
    Main portfolio management system
    """
    
    def __init__(self, storage_dir='portfolio_data'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.portfolios = {}
        self.strategies = {
            'forecast': ForecastBasedStrategy(),
            'momentum': MomentumStrategy()
        }
        
        self.load_portfolios()
    
    def create_portfolio(self, portfolio_id, initial_cash=100000.0):
        """Create a new portfolio"""
        if portfolio_id in self.portfolios:
            logger.warning(f"Portfolio {portfolio_id} already exists")
            return self.portfolios[portfolio_id]
        
        portfolio = Portfolio(initial_cash=initial_cash, portfolio_id=portfolio_id)
        self.portfolios[portfolio_id] = portfolio
        self.save_portfolio(portfolio_id)
        
        logger.info(f"✅ Created portfolio {portfolio_id} with ${initial_cash:.2f}")
        return portfolio
    
    def get_portfolio(self, portfolio_id):
        """Get portfolio by ID"""
        return self.portfolios.get(portfolio_id)
    
    def execute_trade(self, portfolio_id, action, symbol, quantity, price, timestamp=None):
        """
        Execute a trade
        
        Args:
            portfolio_id: Portfolio ID
            action: 'BUY' or 'SELL'
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            timestamp: Transaction timestamp
        
        Returns:
            dict with transaction details
        """
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            logger.error(f"Portfolio {portfolio_id} not found")
            return None
        
        if action.upper() == 'BUY':
            result = portfolio.buy(symbol, quantity, price, timestamp)
        elif action.upper() == 'SELL':
            result = portfolio.sell(symbol, quantity, price, timestamp)
        else:
            logger.error(f"Invalid action: {action}")
            return None
        
        if result:
            self.save_portfolio(portfolio_id)
        
        return result
    
    def execute_strategy(self, portfolio_id, symbol, historical_data, 
                        forecast_data, current_price, strategy_name='forecast'):
        """
        Execute trading strategy
        
        Args:
            portfolio_id: Portfolio ID
            symbol: Stock symbol
            historical_data: Historical prices
            forecast_data: Forecasted prices
            current_price: Current market price
            strategy_name: Strategy to use
        
        Returns:
            dict with signal and execution details
        """
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            return {'signal': 'HOLD', 'reason': 'Portfolio not found'}
        
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            return {'signal': 'HOLD', 'reason': 'Strategy not found'}
        
        # Generate signal
        signal = strategy.generate_signal(historical_data, forecast_data, current_price)
        
        result = {
            'signal': signal,
            'strategy': strategy_name,
            'symbol': symbol,
            'current_price': current_price
        }
        
        # Execute based on signal
        if signal == 'BUY':
            # Calculate position size (use 10% of cash)
            max_investment = portfolio.cash * 0.1
            quantity = int(max_investment / current_price)
            
            if quantity > 0:
                transaction = portfolio.buy(symbol, quantity, current_price)
                if transaction:
                    result['executed'] = True
                    result['transaction'] = transaction
                    self.save_portfolio(portfolio_id)
                else:
                    result['executed'] = False
                    result['reason'] = 'Insufficient cash'
            else:
                result['executed'] = False
                result['reason'] = 'Insufficient cash for minimum quantity'
        
        elif signal == 'SELL':
            # Sell current position
            position = portfolio.get_position(symbol)
            if position['quantity'] > 0:
                transaction = portfolio.sell(symbol, position['quantity'], current_price)
                if transaction:
                    result['executed'] = True
                    result['transaction'] = transaction
                    self.save_portfolio(portfolio_id)
                else:
                    result['executed'] = False
                    result['reason'] = 'Sell failed'
            else:
                result['executed'] = False
                result['reason'] = 'No position to sell'
        
        else:  # HOLD
            result['executed'] = False
            result['reason'] = 'Hold signal'
        
        return result
    
    def save_portfolio(self, portfolio_id):
        """Save portfolio to disk"""
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            return False
        
        filepath = os.path.join(self.storage_dir, f'{portfolio_id}.json')
        
        try:
            with open(filepath, 'w') as f:
                json.dump(portfolio.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
            return False
    
    def load_portfolios(self):
        """Load all portfolios from disk"""
        if not os.path.exists(self.storage_dir):
            return
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.storage_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        portfolio = Portfolio.from_dict(data)
                        self.portfolios[portfolio.portfolio_id] = portfolio
                        logger.info(f"Loaded portfolio: {portfolio.portfolio_id}")
                except Exception as e:
                    logger.error(f"Error loading portfolio {filename}: {e}")
    
    def get_all_portfolios_summary(self, current_prices):
        """Get summary of all portfolios"""
        summaries = []
        
        for portfolio_id, portfolio in self.portfolios.items():
            metrics = portfolio.get_performance_metrics(current_prices)
            summaries.append({
                'portfolio_id': portfolio_id,
                **metrics
            })
        
        return summaries