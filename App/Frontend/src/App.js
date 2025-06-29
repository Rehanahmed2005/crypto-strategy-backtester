import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const App = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [strategies, setStrategies] = useState([]);
  const [backtestResults, setBacktestResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [sampleDataGenerated, setSampleDataGenerated] = useState(false);

  // Strategy form state
  const [newStrategy, setNewStrategy] = useState({
    name: '',
    symbol: 'BTC-USDT',
    entry_indicator: 'rsi',
    entry_operator: '<',
    entry_value: 30,
    exit_indicator: 'rsi',
    exit_operator: '>',
    exit_value: 70,
    stop_loss_pct: 5,
    take_profit_pct: 10,
    initial_capital: 10000,
    position_size_pct: 100
  });

  useEffect(() => {
    loadStrategies();
  }, []);

  const loadStrategies = async () => {
    try {
      const response = await axios.get(`${API}/strategies`);
      setStrategies(response.data.strategies || []);
    } catch (error) {
      console.error('Error loading strategies:', error);
    }
  };

  const generateSampleData = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/generate-sample-data?symbol=BTC-USDT&days=365`);
      console.log('Sample data generated:', response.data);
      setSampleDataGenerated(true);
      alert(`Generated ${response.data.data_points} data points for BTC-USDT`);
    } catch (error) {
      console.error('Error generating sample data:', error);
      alert('Error generating sample data');
    }
    setLoading(false);
  };

  const createStrategy = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await axios.post(`${API}/strategy`, newStrategy);
      console.log('Strategy created:', response.data);
      setStrategies([...strategies, response.data]);
      
      // Reset form
      setNewStrategy({
        name: '',
        symbol: 'BTC-USDT',
        entry_indicator: 'rsi',
        entry_operator: '<',
        entry_value: 30,
        exit_indicator: 'rsi',
        exit_operator: '>',
        exit_value: 70,
        stop_loss_pct: 5,
        take_profit_pct: 10,
        initial_capital: 10000,
        position_size_pct: 100
      });
      
      alert('Strategy created successfully!');
    } catch (error) {
      console.error('Error creating strategy:', error);
      alert('Error creating strategy');
    }
    setLoading(false);
  };

  const runBacktest = async (strategyId) => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/backtest/${strategyId}`);
      setBacktestResults(response.data);
      console.log('Backtest results:', response.data);
      alert('Backtest completed successfully!');
    } catch (error) {
      console.error('Error running backtest:', error);
      alert('Error running backtest: ' + (error.response?.data?.detail || error.message));
    }
    setLoading(false);
  };

  const formatNumber = (num, decimals = 2) => {
    return typeof num === 'number' ? num.toFixed(decimals) : '0.00';
  };

  const formatCurrency = (num) => {
    return `$${formatNumber(num, 2)}`;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <h1 className="text-2xl font-bold text-gray-900">📈 Backtesting Platform</h1>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {['dashboard', 'strategy-builder', 'results'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-4 px-1 border-b-2 font-medium text-sm capitalize ${
                  activeTab === tab
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.replace('-', ' ')}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          
          {/* Dashboard Tab */}
          {activeTab === 'dashboard' && (
            <div className="space-y-6">
              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="px-4 py-5 sm:p-6">
                  <h2 className="text-lg font-medium text-gray-900 mb-4">Quick Start</h2>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
                      <div>
                        <h3 className="font-medium text-blue-900">1. Generate Sample Data</h3>
                        <p className="text-sm text-blue-700">Create sample BTC-USDT price data for testing strategies</p>
                      </div>
                      <button
                        onClick={generateSampleData}
                        disabled={loading || sampleDataGenerated}
                        className={`px-4 py-2 rounded-md text-sm font-medium ${
                          sampleDataGenerated
                            ? 'bg-green-100 text-green-800 cursor-not-allowed'
                            : 'bg-blue-600 text-white hover:bg-blue-700'
                        }`}
                      >
                        {sampleDataGenerated ? 'Data Generated ✓' : (loading ? 'Generating...' : 'Generate Data')}
                      </button>
                    </div>
                    
                    <div className="flex items-center justify-between p-4 bg-green-50 rounded-lg">
                      <div>
                        <h3 className="font-medium text-green-900">2. Create a Strategy</h3>
                        <p className="text-sm text-green-700">Build your trading strategy using technical indicators</p>
                      </div>
                      <button
                        onClick={() => setActiveTab('strategy-builder')}
                        className="px-4 py-2 bg-green-600 text-white rounded-md text-sm font-medium hover:bg-green-700"
                      >
                        Create Strategy
                      </button>
                    </div>
                    
                    <div className="flex items-center justify-between p-4 bg-purple-50 rounded-lg">
                      <div>
                        <h3 className="font-medium text-purple-900">3. Run Backtest</h3>
                        <p className="text-sm text-purple-700">Test your strategy against historical data</p>
                      </div>
                      <button
                        onClick={() => setActiveTab('results')}
                        className="px-4 py-2 bg-purple-600 text-white rounded-md text-sm font-medium hover:bg-purple-700"
                      >
                        View Results
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Strategies Overview */}
              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="px-4 py-5 sm:p-6">
                  <h2 className="text-lg font-medium text-gray-900 mb-4">Your Strategies</h2>
                  {strategies.length === 0 ? (
                    <p className="text-gray-500">No strategies created yet. Create your first strategy to get started!</p>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {strategies.map((strategy) => (
                        <div key={strategy.id} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
                          <h3 className="font-medium text-gray-900">{strategy.name}</h3>
                          <p className="text-sm text-gray-500 mt-1">{strategy.symbol}</p>
                          <div className="mt-2 text-xs text-gray-600">
                            <p>Entry: {strategy.entry_indicator} {strategy.entry_operator} {strategy.entry_value}</p>
                            <p>Exit: {strategy.exit_indicator} {strategy.exit_operator} {strategy.exit_value}</p>
                          </div>
                          <button
                            onClick={() => runBacktest(strategy.id || strategy._id)
}
                            disabled={loading}
                            className="mt-3 w-full px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50"
                          >
                            {loading ? 'Running...' : 'Run Backtest'}
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Strategy Builder Tab */}
          {activeTab === 'strategy-builder' && (
            <div className="bg-white shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-6">Strategy Builder</h2>
                
                <form onSubmit={createStrategy} className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700">Strategy Name</label>
                      <input
                        type="text"
                        value={newStrategy.name}
                        onChange={(e) => setNewStrategy({...newStrategy, name: e.target.value})}
                        className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                        placeholder="My RSI Strategy"
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700">Symbol</label>
                      <select
                        value={newStrategy.symbol}
                        onChange={(e) => setNewStrategy({...newStrategy, symbol: e.target.value})}
                        className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                      >
                        <option value="BTC-USDT">BTC-USDT</option>
                        <option value="ETH-USDT">ETH-USDT</option>
                      </select>
                    </div>
                  </div>

                  {/* Entry Conditions */}
                  <div className="border-t pt-6">
                    <h3 className="text-md font-medium text-gray-900 mb-4">Entry Conditions</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700">Indicator</label>
                        <select
                          value={newStrategy.entry_indicator}
                          onChange={(e) => setNewStrategy({...newStrategy, entry_indicator: e.target.value})}
                          className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                        >
                          <option value="rsi">RSI</option>
                          <option value="ema_20">EMA 20</option>
                          <option value="ema_50">EMA 50</option>
                          <option value="macd">MACD</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700">Operator</label>
                        <select
                          value={newStrategy.entry_operator}
                          onChange={(e) => setNewStrategy({...newStrategy, entry_operator: e.target.value})}
                          className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                        >
                          <option value="<">Less than</option>
                          <option value=">">Greater than</option>
                          <option value="<=">Less than or equal</option>
                          <option value=">=">Greater than or equal</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700">Value</label>
                        <input
                          type="number"
                          step="0.01"
                          value={newStrategy.entry_value}
                          onChange={(e) => setNewStrategy({...newStrategy, entry_value: parseFloat(e.target.value)})}
                          className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Exit Conditions */}
                  <div>
                    <h3 className="text-md font-medium text-gray-900 mb-4">Exit Conditions</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700">Indicator</label>
                        <select
                          value={newStrategy.exit_indicator}
                          onChange={(e) => setNewStrategy({...newStrategy, exit_indicator: e.target.value})}
                          className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                        >
                          <option value="rsi">RSI</option>
                          <option value="ema_20">EMA 20</option>
                          <option value="ema_50">EMA 50</option>
                          <option value="macd">MACD</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700">Operator</label>
                        <select
                          value={newStrategy.exit_operator}
                          onChange={(e) => setNewStrategy({...newStrategy, exit_operator: e.target.value})}
                          className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                        >
                          <option value="<">Less than</option>
                          <option value=">">Greater than</option>
                          <option value="<=">Less than or equal</option>
                          <option value=">=">Greater than or equal</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700">Value</label>
                        <input
                          type="number"
                          step="0.01"
                          value={newStrategy.exit_value}
                          onChange={(e) => setNewStrategy({...newStrategy, exit_value: parseFloat(e.target.value)})}
                          className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Risk Management */}
                  <div className="border-t pt-6">
                    <h3 className="text-md font-medium text-gray-900 mb-4">Risk Management</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700">Stop Loss (%)</label>
                        <input
                          type="number"
                          step="0.1"
                          value={newStrategy.stop_loss_pct}
                          onChange={(e) => setNewStrategy({...newStrategy, stop_loss_pct: parseFloat(e.target.value)})}
                          className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                          placeholder="5"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700">Take Profit (%)</label>
                        <input
                          type="number"
                          step="0.1"
                          value={newStrategy.take_profit_pct}
                          onChange={(e) => setNewStrategy({...newStrategy, take_profit_pct: parseFloat(e.target.value)})}
                          className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                          placeholder="10"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Execution Parameters */}
                  <div className="border-t pt-6">
                    <h3 className="text-md font-medium text-gray-900 mb-4">Execution Parameters</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700">Initial Capital ($)</label>
                        <input
                          type="number"
                          value={newStrategy.initial_capital}
                          onChange={(e) => setNewStrategy({...newStrategy, initial_capital: parseFloat(e.target.value)})}
                          className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700">Position Size (%)</label>
                        <input
                          type="number"
                          min="1"
                          max="100"
                          value={newStrategy.position_size_pct}
                          onChange={(e) => setNewStrategy({...newStrategy, position_size_pct: parseFloat(e.target.value)})}
                          className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                        />
                      </div>
                    </div>
                  </div>

                  <div className="pt-6">
                    <button
                      type="submit"
                      disabled={loading}
                      className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
                    >
                      {loading ? 'Creating Strategy...' : 'Create Strategy'}
                    </button>
                  </div>
                </form>
              </div>
            </div>
          )}

          {/* Results Tab */}
          {activeTab === 'results' && (
            <div className="space-y-6">
              {backtestResults ? (
                <>
                  {/* Performance Overview */}
                  <div className="bg-white overflow-hidden shadow rounded-lg">
                    <div className="px-4 py-5 sm:p-6">
                      <h2 className="text-lg font-medium text-gray-900 mb-6">Performance Overview</h2>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        <div className="bg-blue-50 p-4 rounded-lg">
                          <h3 className="text-sm font-medium text-blue-900">Total Return</h3>
                          <p className="text-2xl font-bold text-blue-600">
                            {formatNumber(backtestResults.total_return_pct)}%
                          </p>
                          <p className="text-sm text-blue-700">
                            {formatCurrency(backtestResults.total_return_usd)}
                          </p>
                        </div>
                        
                        <div className="bg-green-50 p-4 rounded-lg">
                          <h3 className="text-sm font-medium text-green-900">CAGR</h3>
                          <p className="text-2xl font-bold text-green-600">
                            {formatNumber(backtestResults.cagr_pct)}%
                          </p>
                        </div>
                        
                        <div className="bg-purple-50 p-4 rounded-lg">
                          <h3 className="text-sm font-medium text-purple-900">Sharpe Ratio</h3>
                          <p className="text-2xl font-bold text-purple-600">
                            {formatNumber(backtestResults.sharpe_ratio)}
                          </p>
                        </div>
                        
                        <div className="bg-red-50 p-4 rounded-lg">
                          <h3 className="text-sm font-medium text-red-900">Max Drawdown</h3>
                          <p className="text-2xl font-bold text-red-600">
                            -{formatNumber(backtestResults.max_drawdown_pct)}%
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Trading Metrics */}
                  <div className="bg-white overflow-hidden shadow rounded-lg">
                    <div className="px-4 py-5 sm:p-6">
                      <h2 className="text-lg font-medium text-gray-900 mb-6">Trading Metrics</h2>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        <div className="bg-gray-50 p-4 rounded-lg">
                          <h3 className="text-sm font-medium text-gray-900">Total Trades</h3>
                          <p className="text-xl font-bold text-gray-700">{backtestResults.total_trades}</p>
                        </div>
                        
                        <div className="bg-gray-50 p-4 rounded-lg">
                          <h3 className="text-sm font-medium text-gray-900">Win Rate</h3>
                          <p className="text-xl font-bold text-gray-700">
                            {formatNumber(backtestResults.win_rate_pct)}%
                          </p>
                        </div>
                        
                        <div className="bg-gray-50 p-4 rounded-lg">
                          <h3 className="text-sm font-medium text-gray-900">Avg Duration</h3>
                          <p className="text-xl font-bold text-gray-700">
                            {formatNumber(backtestResults.avg_trade_duration_hours)}h
                          </p>
                        </div>
                        
                        <div className="bg-green-50 p-4 rounded-lg">
                          <h3 className="text-sm font-medium text-green-900">Largest Win</h3>
                          <p className="text-xl font-bold text-green-600">
                            {formatNumber(backtestResults.largest_win_pct)}%
                          </p>
                        </div>
                        
                        <div className="bg-red-50 p-4 rounded-lg">
                          <h3 className="text-sm font-medium text-red-900">Largest Loss</h3>
                          <p className="text-xl font-bold text-red-600">
                            {formatNumber(backtestResults.largest_loss_pct)}%
                          </p>
                        </div>
                        
                        <div className="bg-blue-50 p-4 rounded-lg">
                          <h3 className="text-sm font-medium text-blue-900">Volatility</h3>
                          <p className="text-xl font-bold text-blue-600">
                            {formatNumber(backtestResults.volatility_pct)}%
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Recent Trades */}
                  {backtestResults.trades && backtestResults.trades.length > 0 && (
                    <div className="bg-white overflow-hidden shadow rounded-lg">
                      <div className="px-4 py-5 sm:p-6">
                        <h2 className="text-lg font-medium text-gray-900 mb-6">Recent Trades</h2>
                        <div className="overflow-x-auto">
                          <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                              <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                  Entry
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                  Exit
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                  Duration
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                  P&L (%)
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                  P&L ($)
                                </th>
                              </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                              {backtestResults.trades.slice(-10).map((trade, index) => (
                                <tr key={index}>
                                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                    {formatCurrency(trade.entry_price)}
                                  </td>
                                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                    {formatCurrency(trade.exit_price)}
                                  </td>
                                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                    {formatNumber(trade.duration_hours)}h
                                  </td>
                                  <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${
                                    trade.pnl_pct >= 0 ? 'text-green-600' : 'text-red-600'
                                  }`}>
                                    {trade.pnl_pct >= 0 ? '+' : ''}{formatNumber(trade.pnl_pct)}%
                                  </td>
                                  <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${
                                    trade.pnl_usd >= 0 ? 'text-green-600' : 'text-red-600'
                                  }`}>
                                    {trade.pnl_usd >= 0 ? '+' : ''}{formatCurrency(trade.pnl_usd)}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="bg-white overflow-hidden shadow rounded-lg">
                  <div className="px-4 py-5 sm:p-6 text-center">
                    <h2 className="text-lg font-medium text-gray-900 mb-4">No Backtest Results</h2>
                    <p className="text-gray-500 mb-6">
                      Run a backtest to see performance results and analytics.
                    </p>
                    <button
                      onClick={() => setActiveTab('dashboard')}
                      className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700"
                    >
                      Go to Dashboard
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default App;