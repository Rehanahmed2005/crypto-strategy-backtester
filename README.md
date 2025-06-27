# GoQuant | Backtesting Platform

A full-stack backtesting platform for creating and evaluating crypto trading strategies using custom indicators and visual performance metrics.

---

## Project Overview

This platform enables users to define trading strategies using indicators like **RSI**, **MACD**, and **EMA**, simulate them on historical/synthetic data, and view performance metrics like ROI, Sharpe ratio, and win rate.

The goal is simple: **test trading logic before burning your money in live markets**.

---

## Key Features

- **Strategy Builder** – Intuitive interface to define entry/exit logic with indicators  
- **Backtesting Engine** – Simulates strategies over historical/synthetic BTC-USDT data  
- **Analytics Dashboard** – View total return, CAGR, Sharpe, win rate, drawdown  
- **Risk Controls** – Stop-loss, take-profit, position sizing  
- **Trade Log** – Detailed breakdown of each trade with PnL  
- **Error Handling** – NaN-safe calculations and validation

---

## Tech Stack

**Backend:**
- FastAPI (Python)
- NumPy (data handling)
- Uvicorn (ASGI server)

**Frontend:**
- React (18+)
- Tailwind CSS
- Axios (API requests)

**Dev Tools:**
- Git & GitHub
- VS Code
- Python venv + npm

---

## Strategy Workflow
- Generate Sample Data
    Synthetic BTC/USDT data for testing

- Create Strategy
    Choose indicator (RSI, EMA, MACD)
    Define entry/exit conditions (e.g. RSI < 30, EMA > 50)
    Add SL/TP and position size

- Run Backtest
    Engine simulates trades based on strategy
    Results shown with ROI, PnL, Sharpe, and trade log
    
- Analyze Results
    Total Return, CAGR, Drawdown
    Win Rate, Avg Trade Duration, Volatility

## Testing
- Tested on:
    Edge strategies (e.g. RSI > 90 / < 10)
    Zero-trade cases (handled safely)
    Large trade volumes

- Sharpe & volatility calculations patched to avoid NaN

## Notes
    Currently uses synthetic data only
    Strategy logic is threshold-based, not full scripting
    No DB or user accounts — everything is local and stateless

## Security & Validation
- Inputs validated via Pydantic (backend)
- Environment variables handled via .env (never committed)
- CORS enabled only for local development

## Status
- Core functionality complete
- Additional features (charts, DB, strategy chaining) under future roadmap

🙋 Author
Rehan Ahmed
Built with discipline, caffeine, and no StackOverflow copy-paste guilt.
📫 Rehansalmani2005@gmail.com | Rehanahmed4001@gmail,com