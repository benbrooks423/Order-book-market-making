import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from datetime import datetime
import threading
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class IndependentMarket:
    """Independent market simulation that generates its own order book"""
    
    def __init__(self, initial_price=100, volatility=0.1, tick_size=0.01):
        self.mid_price = initial_price
        self.volatility = volatility
        self.tick_size = tick_size
        self.price_history = deque(maxlen=1000)
        
        # Market microstructure parameters
        self.spread_width = 0.002  # Market's natural spread (.2%)
        self.order_book_depth = 10
        self.base_order_size = 20
        
        # Order flow parameters
        self.order_arrival_rate = 0.4  # Probability of order per tick
        self.market_order_prob = 0.3   # Probability order is market vs limit
        
        # Market state
        self.bid_book = {}  # {price: size}
        self.ask_book = {}  # {price: size}
        self.last_trades = deque(maxlen=100)
        self.market_orders = deque(maxlen=50)
        
        # Order ownership
        self.order_owners = {}  # {(price, side): owner_id}
        self.next_order_id = 1
        
        # Price model parameters
        self.price_model = 'gbm'
        self.drift = 0.1  # Annual drift
        self.dt = 1/252/24  # Hourly time step
        
        self.initialise_order_book()
    
    def initialise_order_book(self):
        """Initialise the market's order book"""
        spread = self.mid_price * self.spread_width
        
        # Generate bid levels
        for i in range(self.order_book_depth):
            price = self.mid_price - spread/2 - i * self.tick_size
            price = round(price,2)
            size = max(1, int(self.base_order_size * np.random.exponential(1)))
            self.bid_book[price] = size
            self.order_owners[(price, 'bid')] = 'market'
        
        # Generate ask levels
        for i in range(self.order_book_depth):
            price = self.mid_price + spread/2 + i * self.tick_size
            price = round(price,2)
            size = max(1, int(self.base_order_size * np.random.exponential(1)))
            self.ask_book[price] = size
            self.order_owners[(price, 'ask')] = 'market'
    
    def update_mid_price(self):
        """Update the mid price using Geometric Brownian Motion"""
        if self.price_model == 'gbm':
            shock = np.random.normal(0, 1)
            multiplier = np.exp((self.drift - 0.5 * self.volatility**2) * self.dt + 
                               self.volatility * np.sqrt(self.dt) * shock)
            self.mid_price *= multiplier
        self.price_history.append(self.mid_price)
    
    def place_limit_order(self, side, price, size, owner_id):
        """Place a limit order in the book and return order ID"""
        price = round(price, 2)
        order_id = self.next_order_id
        self.next_order_id += 1
        
        if side == 'buy':
            if price in self.bid_book:
                self.bid_book[price] += size
            else:
                self.bid_book[price] = size
            self.order_owners[(price, 'bid')] = owner_id
        else:
            if price in self.ask_book:
                self.ask_book[price] += size
            else:
                self.ask_book[price] = size
            self.order_owners[(price, 'ask')] = owner_id
        
        return order_id
    
    def cancel_order(self, side, price, size, owner_id):
        """Cancel an order from the book"""
        price = round(price, 2)
        
        if side == 'buy' and price in self.bid_book:
            if self.order_owners.get((price, 'bid')) == owner_id:
                self.bid_book[price] = max(0, self.bid_book[price] - size)
                if self.bid_book[price] == 0:
                    del self.bid_book[price]
                    del self.order_owners[(price, 'bid')]
                return True
        elif side == 'sell' and price in self.ask_book:
            if self.order_owners.get((price, 'ask')) == owner_id:
                self.ask_book[price] = max(0, self.ask_book[price] - size)
                if self.ask_book[price] == 0:
                    del self.ask_book[price]
                    del self.order_owners[(price, 'ask')]
                return True
        
        return False
    
    def generate_market_orders(self):
        """Generate incoming market orders"""
        
        orders = []
        
        if np.random.random() < self.order_arrival_rate:
            is_buy = np.random.random() < 0.5
            
            size = np.random.randint(20, 100)
            
            # Determine if market or limit order
            if np.random.random() < self.market_order_prob:
                order_type = 'market'
            else:
                order_type = 'limit'
            
            order = {
                'side': 'buy' if is_buy else 'sell',
                'size': size,
                'type': order_type,
                'timestamp': datetime.now(),
                'owner': 'market_participant'
            }
            
            if order_type == 'limit':
                # Add limit price slightly outside spread
                if is_buy:
                    bid_price = min(self.bid_book.keys()) if self.bid_book else self.mid_price * 0.99
                    order['price'] = round(bid_price,2)
                    
                else:
                    ask_price = max(self.ask_book.keys()) if self.ask_book else self.mid_price * 1.01
                    order['price'] = round(ask_price,2)
            
            orders.append(order)
            self.market_orders.append(order)
        
        return orders
    
    def execute_market_order(self, order):
        """Execute a market order against the book and return fills"""
        fills = []
        
        if order['type'] != 'market':
            return fills
        
        remaining_size = order['size']
        
        if order['side'] == 'buy':
            # Buy market order hits asks
            ask_prices = sorted(self.ask_book.keys())
            
            for price in ask_prices:
                if remaining_size <= 0:
                    break
                
                available_size = self.ask_book[price]
                fill_size = min(remaining_size, available_size)
                
                # Create fill record
                fill = {
                    'price': price,
                    'size': fill_size,
                    'side': 'buy',
                    'timestamp': order['timestamp'],
                    'aggressor': order['owner'],
                    'provider': self.order_owners.get((price, 'ask'), 'unknown')
                }
                fills.append(fill)
                
                # Update book
                self.ask_book[price] -= fill_size
                if self.ask_book[price] <= 0:
                    del self.ask_book[price]
                    if (price, 'ask') in self.order_owners:
                        del self.order_owners[(price, 'ask')]
                
                remaining_size -= fill_size
                
                # Add to trade history
                trade = {
                    'price': price,
                    'size': fill_size,
                    'side': 'buy',
                    'timestamp': order['timestamp']
                }
                self.last_trades.append(trade)
        
        else:
            # Sell market order hits bids
            bid_prices = sorted(self.bid_book.keys(), reverse=True)
            
            for price in bid_prices:
                if remaining_size <= 0:
                    break
                
                available_size = self.bid_book[price]
                fill_size = min(remaining_size, available_size)
                
                # Create fill record
                fill = {
                    'price': price,
                    'size': fill_size,
                    'side': 'sell',
                    'timestamp': order['timestamp'],
                    'aggressor': order['owner'],
                    'provider': self.order_owners.get((price, 'bid'), 'unknown')
                }
                fills.append(fill)
                
                # Update book
                self.bid_book[price] -= fill_size
                if self.bid_book[price] <= 0:
                    del self.bid_book[price]
                    if (price, 'bid') in self.order_owners:
                        del self.order_owners[(price, 'bid')]
                
                remaining_size -= fill_size
                
                # Add to trade history
                trade = {
                    'price': price,
                    'size': fill_size,
                    'side': 'sell',
                    'timestamp': order['timestamp']
                }
                self.last_trades.append(trade)              
        
        return fills
    
    
    def add_limit_order(self, order):
        """Add a limit order to the book"""
        if order['type'] != 'limit':
            return False
        
        price = round(order['price'], 2)
        size = order['size']
        
        if order['side'] == 'buy':
            if price in self.bid_book:
                self.bid_book[price] += size
            else:
                self.bid_book[price] = size
        else:
            if price in self.ask_book:
                self.ask_book[price] += size
            else:
                self.ask_book[price] = size
        
        return True
    

    def replenish_book(self):
        """Replenish the order book to maintain depth"""
        # Replenish bids
        while len(self.bid_book) < self.order_book_depth:
            if self.bid_book:
                lowest_bid = min(self.bid_book.keys())
                new_price = lowest_bid - self.tick_size
            else:
                new_price = self.mid_price * 0.95
            
            if self.ask_book:
                best_ask = min(self.ask_book.keys())
                while new_price >= best_ask:
                    new_price = best_ask - self.tick_size
        
            size = max(1, int(self.base_order_size * np.random.exponential(1)))
            self.bid_book[round(new_price, 2)] = size
        
        # Replenish asks
        while len(self.ask_book) < self.order_book_depth:
            if self.ask_book:
                highest_ask = max(self.ask_book.keys())
                new_price = highest_ask + self.tick_size
            else:
                new_price = self.mid_price * 1.05
            
            if self.bid_book:
                best_bid = max(self.bid_book.keys())
                while new_price <= best_bid:
                    new_price = best_bid + self.tick_size
        
            size = max(1, int(self.base_order_size * np.random.exponential(1)))
            self.ask_book[round(new_price, 2)] = size

    
    def get_best_bid_ask(self):
        """Get current best bid and ask"""
        best_bid = max(self.bid_book.keys()) if self.bid_book else None
        best_ask = min(self.ask_book.keys()) if self.ask_book else None
        return best_bid, best_ask
    
    
    def get_market_spread(self):
        """Get the current market spread"""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return best_ask - best_bid
        return None
    
    def update(self):
        """Update the market state"""
        # Update mid price
        self.update_mid_price()
        
        # Generate and process orders
        orders = self.generate_market_orders()
        all_fills = []

        
        for order in orders:
            if order['type'] == 'market':
                fills = self.execute_market_order(order)
                all_fills.extend(fills)
            else:
                self.add_limit_order(order)
        
        # Replenish book
        self.replenish_book()
        
        return all_fills

class MarketMaker:
    """Market maker that reacts to the independent market"""
    
    def __init__(self, initial_cash=100000, max_inventory=100):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.inventory = 0
        self.max_inventory = max_inventory
        self.owner_id = 'market_maker'
        
        # Market maker parameters
        self.spread_multiplier = 1.2 
        self.min_spread = 0.002    
        
        # Risk management
        self.max_order_value = initial_cash * 0.02 #Max 2% of capital per order
        self.inventory_target = 0  # Target inventory level
        
        # Dynamic sizing
        self.base_order_size = 10
        self.min_order_size = 5
        self.max_order_size = 25
        
        # Order management
        self.active_orders = {'bid': None, 'ask': None}
        self.order_size = self.base_order_size
        self.pending_fills = deque(maxlen=100)
        
        # Trading history and performance tracking
        self.trades = []
        self.pnl_history = deque(maxlen=1000)
        self.returns = []
        self.total_volume = 0
        self.buy_trades = 0
        self.sell_trades = 0

        

    def calculate_fair_price(self, market):
        """Improved fair price calculation with better stability"""
        fair_price = market.mid_price
        return fair_price

    def calculate_quotes(self, market):
        """Enhanced quote calculation with volatility adjustment and dynamic sizing"""
        fair_price = self.calculate_fair_price(market)
        market_spread = market.get_market_spread()
        
        # Estimate realised volatility
        if len(market.price_history) >= 20:
            recent_prices = np.array(list(market.price_history))
            returns = np.diff(np.log(recent_prices))
            recent_vol = returns[-20:].std() * np.sqrt(20*24)
            realised_vol = np.std(returns)*np.sqrt(252 * 24)  
            vol_multiplier = 1 + 2 * np.log(1 + recent_vol / realised_vol)
        else:
            vol_multiplier = 1.0

        
        vol_multiplier = max(1.0,vol_multiplier)
        
        spread = market_spread * vol_multiplier * self.spread_multiplier 
        # Ensure minimum spread
        spread = max(self.min_spread, spread)
        
        skew_mult = 1 +((abs(self.inventory) - self.inventory_target) / self.max_inventory)
        skew = 0.05*skew_mult
        
        max_skew = spread * 0.4  # Allow skew up to 40% of spread
        skew = min(skew, max_skew)
        
        # Calculate quotes
        if self.inventory>0:
            bid_price = fair_price - spread/2 - skew
            ask_price = fair_price + spread/2 + skew
        elif self.inventory<0:
            bid_price = fair_price - spread/2 + skew
            ask_price = fair_price + spread/2 - skew
        else:
            bid_price = fair_price - spread/2
            ask_price = fair_price + spread/2
            
        best_bid, best_ask = market.get_best_bid_ask()
        
        if best_bid is not None and best_ask is not None:
            # Our bid should be at or below the best bid (but not cross the best ask)
            if bid_price >= best_ask:
                bid_price = best_ask - market.tick_size
            
            # Our ask should be at or above the best ask (but not cross the best bid)  
            if ask_price <= best_bid:
                ask_price = best_bid + market.tick_size
    
        # Final safety check - ensure positive spread
        if ask_price <= bid_price:
            mid_point = (bid_price + ask_price) / 2
            bid_price = mid_point - market.tick_size
            ask_price = mid_point + market.tick_size
        
        return round(bid_price, 2), round(ask_price, 2)


    def calculate_order_size(self, market, side):
        """Calculate dynamic order size based on risk and market conditions"""
        
        if side == 'buy':
            # Reduce buy size if we're long, increase if we're short
            inventory_adjustment = 1.0 - (self.inventory / self.max_inventory) * 0.5
        else:
            # Reduce sell size if we're short, increase if we're long  
            inventory_adjustment = 1.0 + (self.inventory / self.max_inventory) * 0.5

        inventory_adjustment = max(0.3, min(1.5, inventory_adjustment))
        
        # Volatility adjustment - smaller sizes in high volatility
        if len(market.price_history) >= 10:
            recent_prices = np.array(list(market.price_history)[-10:])
            returns = np.diff(np.log(recent_prices))
            recent_vol = np.std(returns)
            vol_adjustment = max(0.5, min(1.5, 1.0 / (1.0 + recent_vol * 10)))
        else:
            vol_adjustment = 1.0
        
        # Calculate size
        dynamic_size = self.base_order_size * inventory_adjustment * vol_adjustment
        dynamic_size = max(self.min_order_size, min(self.max_order_size, dynamic_size))
        
        # Ensure order value doesn't exceed risk limits
        current_price = market.mid_price
        max_size_by_value = self.max_order_value / current_price
        dynamic_size = min(dynamic_size, max_size_by_value)
        
        return int(dynamic_size)

    def process_fills(self, all_fills):
        """Process order fills"""
        for fill in all_fills:
                # We provided liquidity and got filled
                if fill['side'] == 'buy':
                    # Someone bought from us (hit our ask)
                    if abs(self.inventory - fill['size']) <= self.max_inventory:
                        self.cash += fill['price'] * fill['size']
                        self.inventory -= fill['size']
                        self.sell_trades += 1
                        
                        trade_record = {
                            'side': 'sell',
                            'price': fill['price'],
                            'size': fill['size'],
                            'timestamp': fill['timestamp']
                        }
                        self.trades.append(trade_record)
                        self.total_volume += fill['size']
                        
                        # Clear the ask order since it was filled
                        self.active_orders['ask'] = None
                        
                else:
                    # Someone sold to us (hit our bid)
                    if abs(self.inventory + fill['size']) <= self.max_inventory:
                        self.cash -= fill['price'] * fill['size']
                        self.inventory += fill['size']
                        self.buy_trades += 1
                        
                        trade_record = {
                            'side': 'buy',
                            'price': fill['price'],
                            'size': fill['size'],
                            'timestamp': fill['timestamp']
                        }
                        self.trades.append(trade_record)
                        self.total_volume += fill['size']
                        
                        # Clear the bid order since it was filled
                        self.active_orders['bid'] = None


    def update_quotes(self, market):
        """Quote updating with dynamic sizing and better risk management"""
        
        if self.active_orders['bid']:
            old_bid = self.active_orders['bid']['price']
            old_bid_size = self.active_orders['bid']['size']
            market.cancel_order('buy', old_bid, old_bid_size, self.owner_id)
            self.active_orders['bid'] = None
        
        if self.active_orders['ask']:
            old_ask = self.active_orders['ask']['price']
            old_ask_size = self.active_orders['ask']['size']
            market.cancel_order('sell', old_ask, old_ask_size, self.owner_id)
            self.active_orders['ask'] = None
            
        bid_price, ask_price = self.calculate_quotes(market)
        
        min_spread = market.mid_price * self.min_spread  # At least 2 ticks or min spread
        if ask_price - bid_price < min_spread:
            # Adjust prices to maintain minimum spread
            mid_point = (bid_price + ask_price) / 2
            bid_price = mid_point - min_spread / 2
            ask_price = mid_point + min_spread / 2
        
        # Round to nearest tick
        bid_price = round(bid_price / market.tick_size) * market.tick_size
        ask_price = round(ask_price / market.tick_size) * market.tick_size
        
        # Calculate dynamic order sizes
        bid_size = self.calculate_order_size(market, 'buy')
        ask_size = self.calculate_order_size(market, 'sell')
        
        # Cancel existing orders if prices have changed significantly or if we need to adjust size
        if self.active_orders['bid']:
            old_bid = self.active_orders['bid']['price']
            old_bid_size = self.active_orders['bid']['size']
            price_changed = abs(old_bid - bid_price) > market.tick_size
            size_changed = abs(old_bid_size - bid_size) > 2
            
            if price_changed or size_changed:
                market.cancel_order('buy', old_bid, old_bid_size, self.owner_id)
                self.active_orders['bid'] = None
        
        if self.active_orders['ask']:
            old_ask = self.active_orders['ask']['price']
            old_ask_size = self.active_orders['ask']['size']
            price_changed = abs(old_ask - ask_price) > market.tick_size
            size_changed = abs(old_ask_size - ask_size) > 2
            
            if price_changed or size_changed:
                market.cancel_order('sell', old_ask, old_ask_size, self.owner_id)
                self.active_orders['ask'] = None
        
        # Place new bid if we don't have one and risk limits allow
        if (not self.active_orders['bid'] and 
            abs(self.inventory + bid_size) <= self.max_inventory and
            bid_size * bid_price <= self.max_order_value):
            
            order_id = market.place_limit_order('buy', bid_price, bid_size, self.owner_id)
            self.active_orders['bid'] = {
                'id': order_id,
                'price': bid_price,
                'size': bid_size,
                'timestamp': datetime.now()
            }
        
        # Place new ask if we don't have one and risk limits allow
        if (not self.active_orders['ask'] and 
            abs(self.inventory - ask_size) <= self.max_inventory and
            ask_size * ask_price <= self.max_order_value):
            
            order_id = market.place_limit_order('sell', ask_price, ask_size, self.owner_id)
            self.active_orders['ask'] = {
                'id': order_id,
                'price': ask_price,
                'size': ask_size,
                'timestamp': datetime.now()
            }
            
    

    def calculate_pnl(self, market_price):
        """
        Calculate comprehensive P&L including realised and unrealised components
        """
        
        # Calculate realised P&L from completed trades
        total_bought = 0
        total_sold = 0
        buy_cost = 0
        sell_revenue = 0
        
        for trade in self.trades:
            if trade['side'] == 'buy':
                total_bought += trade['size']
                buy_cost += trade['price'] * trade['size']
            else:  # sell
                total_sold += trade['size']
                sell_revenue += trade['price'] * trade['size']
            
        # Calculate unrealised P&L 
        unrealised_pnl = self.inventory * market_price
        
        # Calculate cash position change
        cash_change = self.cash - self.initial_cash
        
        # Total P&L = Cash change + Unrealised P&L
        # This accounts for: cash from trades + mark-to-market of inventory
        total_pnl = cash_change + unrealised_pnl
        
        # Store P&L history
        self.pnl_history.append(total_pnl)
        return total_pnl
    
    def update(self, market, fills):
        """Update the market maker"""
        # Process any fills first
        self.process_fills(fills)
        
        # Update quotes
        self.update_quotes(market)
        
        # Calculate P&L
        pnl = self.calculate_pnl(market.mid_price)
        
        # Calculate returns
        if len(self.pnl_history) > 1:
            returns = (pnl - list(self.pnl_history)[-2]) / self.initial_cash
            self.returns.append(returns)

class TradingSimulator:
    def __init__(self):
        self.market = IndependentMarket()
        self.market_maker = MarketMaker()
        self.running = False
        self.fig = None
        self.axes = {}
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the matplotlib GUI"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('Independent Market with Market Maker', fontsize=18, color='white', y=0.96)
        
        # Create subplots 
        gs = self.fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3, 
                                  top=0.92, bottom=0.12, left=0.05, right=0.95)
        
        # Market price and our quotes
        self.axes['price'] = self.fig.add_subplot(gs[0, :3])
        self.axes['price'].set_title('Market Price vs Our Quotes', color='white', fontsize=14)
        
        # Market order book
        self.axes['market_book'] = self.fig.add_subplot(gs[0, 3])
        self.axes['market_book'].set_title('Order Book', color='white', fontsize=14)
        
        # P&L 
        self.axes['pnl'] = self.fig.add_subplot(gs[1, :3])
        self.axes['pnl'].set_title('Market Maker P&L', color='white', fontsize=14)
        
        # Stats
        self.axes['stats'] = self.fig.add_subplot(gs[1, 3])
        self.axes['stats'].set_title('Performance Stats', color='white', fontsize=14)
        self.axes['stats'].axis('off')
        
        # Market info 
        self.axes['market_info'] = self.fig.add_subplot(gs[2, :])
        self.axes['market_info'].set_title('Market Information', color='white', fontsize=14)
        self.axes['market_info'].axis('off')
        
        self.add_controls()
    
    def add_controls(self):
        """Add interactive controls with better positioning"""
        # Spread multiplier
        ax_spread = plt.axes([0.05, 0.02, 0.15, 0.03])
        self.spread_slider = Slider(ax_spread, 'Spread Mult', 1.0, 3.0, 
                                   valinit=self.market_maker.spread_multiplier)
        self.spread_slider.on_changed(self.update_spread)
        
        # Market volatility
        ax_vol = plt.axes([0.25, 0.02, 0.15, 0.03])
        self.vol_slider = Slider(ax_vol, 'Market Vol', 0.05, 0.5, 
                                valinit=self.market.volatility)
        self.vol_slider.on_changed(self.update_volatility)
        
        # Order size
        ax_size = plt.axes([0.45, 0.02, 0.15, 0.03])
        self.size_slider = Slider(ax_size, 'Order Size', 5, 50, 
                                 valinit=self.market_maker.order_size)
        self.size_slider.on_changed(self.update_order_size)
        
        # Start/Stop
        ax_button = plt.axes([0.82, 0.02, 0.10, 0.05])
        self.start_button = Button(ax_button, 'Start/Stop')
        self.start_button.on_clicked(self.toggle_simulation)
    
    def update_spread(self, val):
        self.market_maker.spread_multiplier = val
    
    def update_volatility(self, val):
        self.market.volatility = val
    
    def update_order_size(self, val):
        self.market_maker.order_size = int(val)
    
    def toggle_simulation(self, event):
        self.running = not self.running
        if self.running:
            self.start_simulation()
    
    def start_simulation(self):
        """Start the simulation"""
        def simulation_loop():
            while self.running:
                # Update market first
                fills = self.market.update()
                
                # Then market maker reacts
                self.market_maker.update(self.market, fills)
                
                time.sleep(0.1)  # 100ms update
        
        if self.running:
            thread = threading.Thread(target=simulation_loop)
            thread.daemon = True
            thread.start()
    
    def update_plots(self, frame):
        """Update all plots with improved styling"""
        if not self.running:
            return
        
        # Clear axes
        for ax in self.axes.values():
            ax.clear()
        
        # Market price with our quotes 
        if len(self.market.price_history) > 0:
            prices = list(self.market.price_history)
            self.axes['price'].plot(prices, 'cyan', linewidth=2.5, label='Market Price', alpha=0.9)
            
            # Show our quotes 
            if (self.market_maker.active_orders.get('bid') is not None and 
                self.market_maker.active_orders.get('ask') is not None):
                bid_price, ask_price = self.market_maker.calculate_quotes(self.market)
                
                self.axes['price'].axhline(y=bid_price, color='lime', 
                                         linestyle='--', alpha=0.8, linewidth=2,
                                         label=f'Our Bid: ${bid_price:.2f}')
                self.axes['price'].axhline(y=ask_price, color='red', 
                                         linestyle='--', alpha=0.8, linewidth=2,
                                         label=f'Our Ask: ${ask_price:.2f}')
            
            self.axes['price'].set_title('Market Price vs Our Quotes', color='white', fontsize=14)
            self.axes['price'].legend(loc='upper left', framealpha=0.8)
            self.axes['price'].grid(True, alpha=0.3)
            self.axes['price'].set_xlabel('Time', color='white')
            self.axes['price'].set_ylabel('Price ($)', color='white')
        
        # Order book 
        if self.market.bid_book and self.market.ask_book:
            bid_prices = sorted(self.market.bid_book.keys(), reverse=True)[:10]
            ask_prices = sorted(self.market.ask_book.keys())[:10]
            
            bid_colors = []
            ask_colors = []
            
            for price in bid_prices:
                owner = self.market.order_owners.get((price, 'bid'), 'unknown')
                if owner == 'market_maker':
                    bid_colors.append('lightgreen')
                else:
                    bid_colors.append('green')
            
            for price in ask_prices:
                owner = self.market.order_owners.get((price, 'ask'), 'unknown')
                if owner == 'market_maker':
                    ask_colors.append('lightcoral')
                else:
                    ask_colors.append('red')
            
            bid_sizes = [self.market.bid_book[p] for p in bid_prices]
            ask_sizes = [self.market.ask_book[p] for p in ask_prices]
            
            y_pos = range(len(bid_prices))
            
            self.axes['market_book'].barh(y_pos, [-s for s in bid_sizes], 
                                        color=bid_colors, alpha=0.8, label='Bids')
            self.axes['market_book'].barh(y_pos, ask_sizes, 
                                        color=ask_colors, alpha=0.8, label='Asks')
            
            # Add price labels with better formatting
            for i, (bid, ask) in enumerate(zip(bid_prices, ask_prices)):
                if bid_sizes:
                    self.axes['market_book'].text(-max(bid_sizes)/2, i, f'{bid:.2f}', 
                                                ha='center', va='center', color='white', 
                                                fontsize=9, fontweight='bold')
                if ask_sizes:
                    self.axes['market_book'].text(max(ask_sizes)/2, i, f'{ask:.2f}', 
                                                ha='center', va='center', color='white', 
                                                fontsize=9, fontweight='bold')
            
            self.axes['market_book'].set_title('Order Book\n(Light=MM Orders)', color='white', fontsize=12)
            self.axes['market_book'].set_xlabel('Size', color='white')
        
        # P&L with improved styling
        if len(self.market_maker.pnl_history) > 0:
            pnl_data = list(self.market_maker.pnl_history)
            # Color the line based on profit/loss
            colors = ['green' if p >= 0 else 'red' for p in pnl_data]
            self.axes['pnl'].plot(pnl_data, 'yellow', linewidth=2.5, alpha=0.9)
            self.axes['pnl'].axhline(y=0, color='white', linestyle='--', alpha=0.7, linewidth=1)
            self.axes['pnl'].fill_between(range(len(pnl_data)), pnl_data, 0, 
                                        alpha=0.3, color='green' if pnl_data[-1] >= 0 else 'red')
            self.axes['pnl'].set_title('Market Maker P&L', color='white', fontsize=14)
            self.axes['pnl'].grid(True, alpha=0.3)
            self.axes['pnl'].set_xlabel('Time', color='white')
            self.axes['pnl'].set_ylabel('P&L ($)', color='white')
        
        # Stats 
        self.axes['stats'].axis('off')
        pnl = self.market_maker.calculate_pnl(self.market.mid_price)
        
        stats_text = f"""Market Maker:
P&L: ${pnl:,.0f}
Volume: {self.market_maker.total_volume:,}
Buy/Sell: {self.market_maker.buy_trades}/{self.market_maker.sell_trades}
Inventory: {self.market_maker.inventory}
Cash: ${self.market_maker.cash:,.0f}

Market:
Mid: ${self.market.mid_price:.2f}
Spread: {self.market.get_market_spread():.3f}
Spread %: {self.market.get_market_spread()/self.market.mid_price*100:.2f}%
Orders: {len(self.market.market_orders)}"""
        
        self.axes['stats'].text(0.05, 0.95, stats_text, transform=self.axes['stats'].transAxes,
                              fontsize=11, verticalalignment='top', color='white',
                              fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", 
                              facecolor='black', alpha=0.8))
        
        # Market info with cleaner layout
        self.axes['market_info'].axis('off')
        best_bid, best_ask = self.market.get_best_bid_ask()
        
        if (self.market_maker.active_orders.get('bid') is not None and 
            self.market_maker.active_orders.get('ask') is not None):
            our_bid = self.market_maker.active_orders['bid']['price']
            our_ask = self.market_maker.active_orders['ask']['price']
            our_spread = our_ask - our_bid
        else:
            our_bid = our_ask = our_spread = 0
        
        market_text = f"""Market vs Our Quotes:     Market Best Bid: ${best_bid:.2f}  |  Our Bid: ${our_bid:.2f}     Market Best Ask: ${best_ask:.2f}  |  Our Ask: ${our_ask:.2f}     Market Spread: ${self.market.get_market_spread():.3f}  |  Our Spread: ${our_spread:.3f}

Configuration:     Spread Multiplier: {self.market_maker.spread_multiplier:.2f}x  |  Order Size: {self.market_maker.order_size}  |  Max Inventory: {self.market_maker.max_inventory}  |  Market Model: {self.market.price_model.upper()}  |  Volatility: {self.market.volatility*100:.1f}%"""
        
        self.axes['market_info'].text(0.05, 0.8, market_text, transform=self.axes['market_info'].transAxes,
                                    fontsize=11, verticalalignment='top', color='white',
                                    fontfamily='monospace')
        
        # Style all axes with consistent colors
        for ax_name, ax in self.axes.items():
            if ax_name not in ['stats', 'market_info']:
                ax.tick_params(colors='white', labelsize=10)
                ax.set_facecolor('black')
                for spine in ax.spines.values():
                    spine.set_color('white')
                    spine.set_linewidth(1.2)
    
    def run(self):
        """Run the simulation"""
        print("Independent Market with Market Maker")
        print("=" * 50)
        print("KEY FEATURES:")
        print("- Independent market generates its own order book")
        print("- Realistic market microstructure")
        print("- Separate order flow and price discovery")
        print()
        print("Controls:")
        print("- Spread Multiplier: How much wider than market spread")
        print("- Market Volatility: Controls market price movements")
        print("- Order Size: Size of market maker orders")
        print("- Start/Stop: Begin/pause simulation")
        print()
        print("The market maker profits from the bid-ask spread")
        print("while managing inventory risk!")
        
        # Set up animation
        self.ani = animation.FuncAnimation(self.fig, self.update_plots, 
                                         interval=200, blit=False)
        
        plt.tight_layout()
        plt.show()

# Run the simulation
if __name__ == "__main__":
    simulator = TradingSimulator()
    simulator.run()