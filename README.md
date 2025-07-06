# Order-book-market-making

# Market Maker Simulation

A comprehensive Python simulation that models market microstructure and market making strategies in real-time. This project demonstrates how market makers operate in electronic trading environments, managing inventory risk while profiting from bid-ask spreads.

## Overview

This simulation creates an **independent market** with its own order book dynamics and a **market maker** that responds to market conditions. Unlike simplified trading simulations, this implementation includes realistic market microstructure elements like order flow, liquidity provision, and dynamic pricing.

## Key Features

### Independent Market
- **Autonomous Order Book**: Generates its own buy/sell orders with realistic timing
- **Geometric Brownian Motion**: Price discovery follows established financial models
- **Market Microstructure**: Includes spreads, order depth, and liquidity dynamics
- **Order Flow Simulation**: Both market and limit orders with configurable arrival rates

### Market Maker
- **Dynamic Pricing**: Adjusts quotes based on market conditions and inventory
- **Risk Management**: Inventory limits and position-based quote skewing
- **Volatility Adaptation**: Spreads widen during high volatility periods
- **Real-time P&L**: Tracks realized and unrealized profits/losses

### Interactive Visualization
- **Real-time Charts**: Live price action, order book, and P&L tracking
- **Order Book Display**: Visual representation of market depth with market maker orders highlighted
- **Performance Metrics**: Trading statistics, inventory levels, and profitability
- **Interactive Controls**: Adjust parameters during simulation

## Installation

### Prerequisites
```bash
pip install numpy pandas matplotlib
```

### Running the Simulation
```bash
python market_maker3.py
```

## How It Works

### Market Dynamics
1. **Price Evolution**: The market price follows a geometric Brownian motion model
2. **Order Generation**: Random market participants place buy/sell orders
3. **Order Matching**: Market orders execute against the best available prices
4. **Book Replenishment**: The market maintains consistent liquidity depth

### Market Making Strategy
1. **Quote Calculation**: Determines optimal bid/ask prices based on:
   - Current market spread
   - Inventory position
   - Market volatility
   - Risk parameters

2. **Dynamic Sizing**: Order sizes adjust based on:
   - Current inventory levels
   - Market conditions
   - Risk limits

3. **Risk Management**: 
   - Maximum inventory limits
   - Position-based quote skewing
   - Dynamic spread adjustment

## Controls

| Parameter | Description | Range |
|-----------|-------------|-------|
| **Spread Multiplier** | How much wider than market spread | 1.0x - 3.0x |
| **Market Volatility** | Controls price movement intensity | 5% - 50% |
| **Order Size** | Base size for market maker orders | 5 - 50 shares |
| **Start/Stop** | Begin/pause the simulation | - |

## Key Concepts Demonstrated

### Market Microstructure
- **Order Book Dynamics**: How limit orders create market depth
- **Bid-Ask Spreads**: The cost of immediacy in trading
- **Price Discovery**: How trades move market prices
- **Liquidity Provision**: The role of market makers in providing trading opportunities

### Market Making
- **Inventory Management**: Balancing long/short positions
- **Adverse Selection**: Adjusting for informed trading
- **Volatility Risk**: Wider spreads during uncertain periods
- **Profit Sources**: Earning the spread while managing risk

## Understanding the Display

### Price Chart
- **Cyan Line**: Market mid-price evolution
- **Green Dashed**: Your bid quote
- **Red Dashed**: Your ask quote

### Order Book
- **Green Bars**: Market bid orders (buy side)
- **Red Bars**: Market ask orders (sell side)
- **Light Colors**: Your market maker orders
- **Dark Colors**: Other market participants

### Performance Panel
- **P&L**: Current profit/loss including unrealized gains
- **Volume**: Total shares traded
- **Inventory**: Current position (positive = long, negative = short)
- **Buy/Sell Ratio**: Trade distribution

## Educational Value

This simulation teaches:

1. **Market Structure**: How electronic markets actually work
2. **Risk Management**: The importance of position limits and dynamic pricing
3. **Profit Mechanics**: How market makers earn money (and lose it)
4. **Market Impact**: How trading affects prices and spreads
5. **Quantitative Finance**: Real-world application of mathematical models

## Advanced Features

### Volatility Adaptation
The market maker automatically adjusts spreads based on recent price volatility, widening quotes during turbulent periods to manage risk.

### Inventory Skewing
When holding inventory, the market maker skews quotes to encourage trades that reduce position risk:
- **Long Position**: Better bid prices to encourage selling
- **Short Position**: Better ask prices to encourage buying

### Dynamic Order Sizing
Order sizes adjust based on:
- Current inventory levels
- Market volatility
- Available capital
- Risk limits

## Code Structure

```
market_maker_simulation.py
├── IndependentMarket      # Autonomous market simulation
│   ├── Order book management
│   ├── Price evolution (GBM)
│   ├── Order flow generation
│   └── Trade execution
├── MarketMaker           # Market making strategy
│   ├── Quote calculation
│   ├── Risk management
│   ├── Order management
│   └── P&L tracking
└── TradingSimulator      # GUI and coordination
    ├── Real-time visualization
    ├── Interactive controls
    └── Performance monitoring
```

## Planned future extension of the Simulation

Consider adding:
- **Multiple Market Makers**: Competition between liquidity providers
- **Informed Trading**: Traders with superior information
- **News Events**: Sudden volatility shocks
- **Different Assets**: Varying volatility and spread characteristics
- **Transaction Costs**: More realistic profit calculations


