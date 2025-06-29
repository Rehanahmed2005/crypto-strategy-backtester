from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fastapi import UploadFile, File
import json
import io

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Pydantic Models
class OHLCVData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class StrategyConfig(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    symbol: str
    
    # Entry conditions
    entry_indicator: str  # "ema", "rsi", "macd"
    entry_operator: str   # ">", "<", ">=", "<=", "="
    entry_value: float
    
    # Exit conditions
    exit_indicator: str
    exit_operator: str
    exit_value: float
    
    # Risk management
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    
    # Execution
    initial_capital: float = 10000.0
    position_size_pct: float = 100.0  # % of capital per trade
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class BacktestResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str
    
    # Performance metrics
    total_return_pct: float
    total_return_usd: float
    cagr_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    volatility_pct: float
    
    # Trading metrics
    total_trades: int
    win_rate_pct: float
    avg_trade_duration_hours: float
    largest_win_pct: float
    largest_loss_pct: float
    
    # Detailed results
    trades: List[Dict[str, Any]]
    equity_curve: List[Dict[str, Any]]
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Trade(BaseModel):
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl_pct: float
    pnl_usd: float
    type: str  # "long" or "short"

# Technical Indicators
def calculate_ema(prices: np.array, period: int) -> np.array:
    """Calculate Exponential Moving Average"""
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    alpha = 2 / (period + 1)
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema

def calculate_rsi(prices: np.array, period: int = 14) -> np.array:
    """Calculate Relative Strength Index"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = np.zeros_like(prices)
    avg_losses = np.zeros_like(prices)
    
    # Initialize first values
    avg_gains[period] = np.mean(gains[:period])
    avg_losses[period] = np.mean(losses[:period])
    
    # Calculate smoothed averages
    for i in range(period + 1, len(prices)):
        avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
        avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
    
    # Calculate RSI
    rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(prices: np.array, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

# Backtesting Engine
class BacktestEngine:
    def __init__(self, data: pd.DataFrame, strategy: StrategyConfig):
        self.data = data.copy()
        self.strategy = strategy
        self.trades = []
        self.equity_curve = []
        self.position = None  # Current position
        
    def run_backtest(self) -> BacktestResult:
        """Run the backtest with the given strategy"""
        # Prepare indicators
        self._prepare_indicators()
        
        # Initialize
        initial_capital = self.strategy.initial_capital
        current_capital = initial_capital
        position_value = 0
        
        # Track equity
        for i in range(len(self.data)):
            row = self.data.iloc[i]
            
            # Check for entry signal
            if self.position is None and self._check_entry_signal(i):
                self.position = self._enter_position(row, current_capital)
                position_value = self.position['quantity'] * row['close']
                current_capital -= position_value
            
            # Check for exit signal
            elif self.position is not None and self._check_exit_signal(i, row):
                trade = self._exit_position(row)
                self.trades.append(trade)
                current_capital += trade['exit_price'] * trade['quantity']
                position_value = 0
                self.position = None
            
            # Update position value
            if self.position is not None:
                position_value = self.position['quantity'] * row['close']
            
            # Record equity
            total_equity = current_capital + position_value
            self.equity_curve.append({
                'timestamp': row['timestamp'].isoformat(),
                'equity': total_equity,
                'pnl_pct': ((total_equity - initial_capital) / initial_capital) * 100
            })
        
        # Close any open position
        if self.position is not None:
            final_row = self.data.iloc[-1]
            trade = self._exit_position(final_row)
            self.trades.append(trade)
        
        # Calculate performance metrics
        return self._calculate_performance_metrics(initial_capital)
    
    def _prepare_indicators(self):
        """Calculate all technical indicators"""
        prices = self.data['close'].values
        
        # EMA
        self.data['ema_20'] = calculate_ema(prices, 20)
        self.data['ema_50'] = calculate_ema(prices, 50)
        
        # RSI
        self.data['rsi'] = calculate_rsi(prices)
        
        # MACD
        macd, signal, histogram = calculate_macd(prices)
        self.data['macd'] = macd
        self.data['macd_signal'] = signal
        self.data['macd_histogram'] = histogram
    
    def _check_entry_signal(self, index: int) -> bool:
        """Check if entry conditions are met"""
        if index < 50:  # Need enough data for indicators
            return False
        
        row = self.data.iloc[index]
        indicator_value = self._get_indicator_value(row, self.strategy.entry_indicator)
        
        return self._evaluate_condition(
            indicator_value, 
            self.strategy.entry_operator, 
            self.strategy.entry_value
        )
    
    def _check_exit_signal(self, index: int, row: pd.Series) -> bool:
        """Check if exit conditions are met"""
        if self.position is None:
            return False
        
        # Check stop loss
        if self.strategy.stop_loss_pct:
            current_price = row['close']
            entry_price = self.position['entry_price']
            loss_pct = ((current_price - entry_price) / entry_price) * 100
            if loss_pct <= -abs(self.strategy.stop_loss_pct):
                return True
        
        # Check take profit
        if self.strategy.take_profit_pct:
            current_price = row['close']
            entry_price = self.position['entry_price']
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            if profit_pct >= self.strategy.take_profit_pct:
                return True
        
        # Check exit indicator
        indicator_value = self._get_indicator_value(row, self.strategy.exit_indicator)
        return self._evaluate_condition(
            indicator_value,
            self.strategy.exit_operator,
            self.strategy.exit_value
        )
    
    def _get_indicator_value(self, row: pd.Series, indicator: str) -> float:
        """Get the value of a technical indicator"""
        if indicator == "ema_20":
            return row['ema_20']
        elif indicator == "ema_50":
            return row['ema_50']
        elif indicator == "rsi":
            return row['rsi']
        elif indicator == "macd":
            return row['macd']
        elif indicator == "price":
            return row['close']
        else:
            raise ValueError(f"Unknown indicator: {indicator}")
    
    def _evaluate_condition(self, left: float, operator: str, right: float) -> bool:
        """Evaluate a condition"""
        if operator == ">":
            return left > right
        elif operator == "<":
            return left < right
        elif operator == ">=":
            return left >= right
        elif operator == "<=":
            return left <= right
        elif operator == "=":
            return abs(left - right) < 0.001  # Float comparison
        else:
            raise ValueError(f"Unknown operator: {operator}")
    
    def _enter_position(self, row: pd.Series, available_capital: float) -> Dict[str, Any]:
        """Enter a new position"""
        position_size = available_capital * (self.strategy.position_size_pct / 100)
        quantity = position_size / row['close']
        
        return {
            'entry_time': row['timestamp'],
            'entry_price': row['close'],
            'quantity': quantity,
            'type': 'long'
        }
    
    def _exit_position(self, row: pd.Series) -> Dict[str, Any]:
        """Exit current position"""
        if self.position is None:
            raise ValueError("No position to exit")
        
        pnl_usd = (row['close'] - self.position['entry_price']) * self.position['quantity']
        pnl_pct = ((row['close'] - self.position['entry_price']) / self.position['entry_price']) * 100
        
        duration = row['timestamp'] - self.position['entry_time']
        
        return {
            'entry_time': self.position['entry_time'].isoformat(),
            'exit_time': row['timestamp'].isoformat(),
            'entry_price': self.position['entry_price'],
            'exit_price': row['close'],
            'quantity': self.position['quantity'],
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'duration_hours': duration.total_seconds() / 3600,
            'type': self.position['type']
        }
    
    def _calculate_performance_metrics(self, initial_capital: float) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        if not self.equity_curve:
            raise ValueError("No equity curve data")
        
        # Extract returns
        equity_values = [point['equity'] for point in self.equity_curve]
        returns = [(equity_values[i] - equity_values[i-1]) / equity_values[i-1] 
                  for i in range(1, len(equity_values))]
        
        # Performance metrics
        final_equity = equity_values[-1]
        total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100
        total_return_usd = final_equity - initial_capital
        
        # CAGR (assuming daily data)
        days = len(equity_values)
        years = days / 365.0
        cagr_pct = ((final_equity / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Risk metrics
        returns_array = np.array(returns)
        std_dev = np.std(returns_array)
        periods_per_year = 365 * 24  # Hourly data

        volatility_pct = std_dev * np.sqrt(periods_per_year) * 100 if std_dev > 0 else 0

        if std_dev > 0:
            sharpe_ratio = (np.mean(returns_array) * periods_per_year) / (std_dev * np.sqrt(periods_per_year))
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        peak = equity_values[0]
        max_drawdown_pct = 0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = ((peak - equity) / peak) * 100
            if drawdown > max_drawdown_pct:
                max_drawdown_pct = drawdown
        
        # Trading metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl_pct'] > 0]
        win_rate_pct = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        avg_duration = np.mean([t['duration_hours'] for t in self.trades]) if self.trades else 0
        largest_win = max([t['pnl_pct'] for t in self.trades]) if self.trades else 0
        largest_loss = min([t['pnl_pct'] for t in self.trades]) if self.trades else 0
        
        return BacktestResult(
            strategy_id=self.strategy.id,
            total_return_pct=total_return_pct,
            total_return_usd=total_return_usd,
            cagr_pct=cagr_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            volatility_pct=volatility_pct,
            total_trades=total_trades,
            win_rate_pct=win_rate_pct,
            avg_trade_duration_hours=avg_duration,
            largest_win_pct=largest_win,
            largest_loss_pct=largest_loss,
            trades=self.trades,
            equity_curve=self.equity_curve
        )

# Sample data generator for testing
def generate_sample_ohlcv_data(symbol: str = "BTC-USDT", days: int = 365) -> List[Dict[str, Any]]:
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducible results
    
    start_date = datetime.now() - timedelta(days=days)
    start_price = 50000.0  # Starting BTC price
    
    data = []
    current_price = start_price
    
    for i in range(days * 24):  # Hourly data
        timestamp = start_date + timedelta(hours=i)
        
        # Random walk with trend
        change_pct = np.random.normal(0.001, 0.02)  # Small upward trend with volatility
        new_price = current_price * (1 + change_pct)
        
        # OHLC logic
        high = new_price * (1 + abs(np.random.normal(0, 0.01)))
        low = new_price * (1 - abs(np.random.normal(0, 0.01)))
        
        # Ensure OHLC constraints
        high = max(high, current_price, new_price)
        low = min(low, current_price, new_price)
        
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'symbol': symbol,
            'timestamp': timestamp,
            'open': current_price,
            'high': high,
            'low': low,
            'close': new_price,
            'volume': volume
        })
        
        current_price = new_price
    
    return data

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Backtesting Platform API"}

@api_router.post("/generate-sample-data")
async def generate_sample_data(symbol: str = "BTC-USDT", days: int = 365):
    """Generate and store sample OHLCV data"""
    try:
        sample_data = generate_sample_ohlcv_data(symbol, days)
        
        # Clear existing data for this symbol
        await db.ohlcv_data.delete_many({"symbol": symbol})
        
        # Insert new data
        ohlcv_objects = [OHLCVData(**item) for item in sample_data]
        ohlcv_dicts = [obj.dict() for obj in ohlcv_objects]
        await db.ohlcv_data.insert_many(ohlcv_dicts)
        
        return {
            "message": f"Generated {len(sample_data)} data points for {symbol}",
            "data_points": len(sample_data),
            "date_range": {
                "start": sample_data[0]['timestamp'].isoformat(),
                "end": sample_data[-1]['timestamp'].isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/data/{symbol}")
async def get_ohlcv_data(symbol: str, limit: int = 1000):
    """Get OHLCV data for a symbol"""
    try:
        data = await db.ohlcv_data.find({"symbol": symbol}).sort("timestamp", 1).limit(limit).to_list(limit)
        
        # Convert MongoDB ObjectId to string to make it JSON serializable
        for item in data:
            if '_id' in item:
                item['_id'] = str(item['_id'])
        
        return {"symbol": symbol, "data": data, "count": len(data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/strategy")
async def create_strategy(strategy: StrategyConfig):
    try:
        strategy_dict = strategy.dict()
        insert_result = await db.strategies.insert_one(strategy_dict)
        strategy_dict["_id"] = str(insert_result.inserted_id)
        return strategy_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/strategies")
async def get_strategies():
    """Get all trading strategies"""
    try:
        strategies = await db.strategies.find().to_list(1000)
        
        # Convert MongoDB ObjectId to string to make it JSON serializable
        for strategy in strategies:
            if '_id' in strategy:
                strategy['_id'] = str(strategy['_id'])
        
        return {"strategies": strategies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/backtest/{strategy_id}")
async def run_backtest(strategy_id: str):
    """Run backtest for a strategy"""
    try:
        # Get strategy
        strategy_data = await db.strategies.find_one({"id": strategy_id})
        if not strategy_data:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        strategy = StrategyConfig(**strategy_data)
        
        # Get OHLCV data
        ohlcv_data = await db.ohlcv_data.find({"symbol": strategy.symbol}).sort("timestamp", 1).to_list(10000)
        if not ohlcv_data:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {strategy.symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Run backtest
        engine = BacktestEngine(df, strategy)
        result = engine.run_backtest()
        
        # Save result
        result_dict = result.dict()
        await db.backtest_results.insert_one(result_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/backtest-results/{strategy_id}")
async def get_backtest_results(strategy_id: str):
    """Get backtest results for a strategy"""
    try:
        results = await db.backtest_results.find({"strategy_id": strategy_id}).sort("created_at", -1).to_list(10)
        
        # Convert MongoDB ObjectId to string to make it JSON serializable
        for result in results:
            if '_id' in result:
                result['_id'] = str(result['_id'])
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()