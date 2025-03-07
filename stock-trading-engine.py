import threading
import time
import random
import uuid
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
import ctypes  # For atomic operations

# Define order types
class OrderType(Enum):
    BUY = 0
    SELL = 1

# Order class
class Order:
    def __init__(self, order_id: str, order_type: OrderType, ticker_id: int, 
                 quantity: int, price: float):
        self.id = order_id
        self.type = order_type
        self.ticker_id = ticker_id
        self.quantity = quantity
        self.price = price
        self.timestamp = time.time()
        self.active = True  # Atomic flag for lock-free removal
        self._lock = threading.Lock()  # For atomic updates to quantity
    
    def decrease_quantity(self, amount: int) -> int:
        """Atomically decrease quantity and return new value"""
        with self._lock:
            if amount > self.quantity:
                amount = self.quantity
            self.quantity -= amount
            return self.quantity

# Transaction class
class Transaction:
    def __init__(self, buy_order_id: str, sell_order_id: str, ticker_id: int,
                 quantity: int, price: float):
        self.id = str(uuid.uuid4())
        self.buy_order_id = buy_order_id
        self.sell_order_id = sell_order_id
        self.ticker_id = ticker_id
        self.quantity = quantity
        self.price = price
        self.timestamp = time.time()

# Stock Exchange class
class StockExchange:
    def __init__(self, ticker_count: int = 1024):
        # Generate ticker symbols
        self.tickers = [f"TICK{i:04d}" for i in range(ticker_count)]
        self.ticker_count = ticker_count
        
        # Order books - separate lists for buy and sell orders per ticker
        self.buy_orders = [[] for _ in range(ticker_count)]
        self.sell_orders = [[] for _ in range(ticker_count)]
        
        # Locks for each ticker's order books to allow concurrent operations on different tickers
        self.ticker_locks = [threading.Lock() for _ in range(ticker_count)]
        
        # Transaction log with its own lock
        self.transactions = []
        self.transaction_lock = threading.Lock()
        
        # Atomic counter for active orders
        self.active_order_count = ctypes.c_int64(0)

    def find_ticker_index(self, ticker: str) -> int:
        """Find the index of a ticker symbol"""
        try:
            return self.tickers.index(ticker)
        except ValueError:
            return -1  # Ticker not found

    def add_order(self, order_type: OrderType, ticker: str, quantity: int, price: float) -> str:
        """Add a new order to the trading engine and process any matches"""
        if quantity <= 0 or price <= 0:
            raise ValueError("Quantity and price must be positive")
        
        ticker_id = self.find_ticker_index(ticker)
        if ticker_id < 0:
            raise ValueError(f"Invalid ticker symbol: {ticker}")
            
        # Create a new order
        order_id = str(uuid.uuid4())
        new_order = Order(order_id, order_type, ticker_id, quantity, price)
        
        # Increment active order count
        ctypes.c_int64.from_address(
            ctypes.addressof(self.active_order_count)
        ).value += 1
        
        # Add to appropriate order book
        if order_type == OrderType.BUY:
            with self.ticker_locks[ticker_id]:
                self.buy_orders[ticker_id].append(new_order)
        else:  # SELL
            with self.ticker_locks[ticker_id]:
                self.sell_orders[ticker_id].append(new_order)
        
        # Match the order
        self.match_order(order_id)
        
        return order_id

    def find_order(self, order_id: str) -> Tuple[Optional[Order], int, bool]:
        """Find an order by ID and return (order, ticker_id, is_buy)"""
        # Search buy orders first
        for ticker_id in range(self.ticker_count):
            with self.ticker_locks[ticker_id]:
                for order in self.buy_orders[ticker_id]:
                    if order.id == order_id and order.active:
                        return order, ticker_id, True
        
        # Then search sell orders
        for ticker_id in range(self.ticker_count):
            with self.ticker_locks[ticker_id]:
                for order in self.sell_orders[ticker_id]:
                    if order.id == order_id and order.active:
                        return order, ticker_id, False
        
        return None, -1, False

    def record_transaction(self, buy_order_id: str, sell_order_id: str, 
                          ticker_id: int, quantity: int, price: float) -> None:
        """Record a transaction"""
        transaction = Transaction(buy_order_id, sell_order_id, ticker_id, quantity, price)
        
        with self.transaction_lock:
            self.transactions.append(transaction)
        
        # Print transaction details for visibility
        print(f"MATCHED: {self.tickers[ticker_id]} - {quantity} shares at ${price:.2f} " +
              f"(Buy #{buy_order_id[:8]}, Sell #{sell_order_id[:8]})")

    def match_order(self, order_id: str) -> None:
        """
        Match an order with existing orders of the opposite type
        Time complexity: O(n) where n is the number of orders for the ticker
        """
        # Find the order
        order, ticker_id, is_buy = self.find_order(order_id)
        if not order or ticker_id < 0:
            return  # Order not found
        
        # We need to match this order with orders of the opposite type
        if is_buy:
            # This is a buy order - match with sell orders
            with self.ticker_locks[ticker_id]:
                sell_orders = [o for o in self.sell_orders[ticker_id] if o.active]
                # Sort sell orders by price (ascending) and time (ascending)
                sell_orders.sort(key=lambda x: (x.price, x.timestamp))
                
                # Try to match with each sell order
                for sell_order in sell_orders:
                    if not order.active or order.quantity <= 0:
                        break  # This order is fully matched
                    
                    if order.price >= sell_order.price:
                        # Calculate trade quantity
                        trade_qty = min(order.quantity, sell_order.quantity)
                        
                        # Update order quantities atomically
                        remaining_sell = sell_order.decrease_quantity(trade_qty)
                        remaining_buy = order.decrease_quantity(trade_qty)
                        
                        # Record the transaction
                        self.record_transaction(
                            order.id, sell_order.id, ticker_id, trade_qty, sell_order.price
                        )
                        
                        # Mark orders as inactive if fully filled
                        if remaining_sell == 0:
                            sell_order.active = False
                            # Decrement active order count
                            ctypes.c_int64.from_address(
                                ctypes.addressof(self.active_order_count)
                            ).value -= 1
                            
                        if remaining_buy == 0:
                            order.active = False
                            # Decrement active order count
                            ctypes.c_int64.from_address(
                                ctypes.addressof(self.active_order_count)
                            ).value -= 1
                            break
        else:
            # This is a sell order - match with buy orders
            with self.ticker_locks[ticker_id]:
                buy_orders = [o for o in self.buy_orders[ticker_id] if o.active]
                # Sort buy orders by price (descending) and time (ascending)
                buy_orders.sort(key=lambda x: (-x.price, x.timestamp))
                
                # Try to match with each buy order
                for buy_order in buy_orders:
                    if not order.active or order.quantity <= 0:
                        break  # This order is fully matched
                    
                    if buy_order.price >= order.price:
                        # Calculate trade quantity
                        trade_qty = min(order.quantity, buy_order.quantity)
                        
                        # Update order quantities atomically
                        remaining_buy = buy_order.decrease_quantity(trade_qty)
                        remaining_sell = order.decrease_quantity(trade_qty)
                        
                        # Record the transaction
                        self.record_transaction(
                            buy_order.id, order.id, ticker_id, trade_qty, order.price
                        )
                        
                        # Mark orders as inactive if fully filled
                        if remaining_buy == 0:
                            buy_order.active = False
                            # Decrement active order count
                            ctypes.c_int64.from_address(
                                ctypes.addressof(self.active_order_count)
                            ).value -= 1
                            
                        if remaining_sell == 0:
                            order.active = False
                            # Decrement active order count
                            ctypes.c_int64.from_address(
                                ctypes.addressof(self.active_order_count)
                            ).value -= 1
                            break

    def cleanup_inactive_orders(self) -> None:
        """Remove inactive orders from the order books"""
        for ticker_id in range(self.ticker_count):
            with self.ticker_locks[ticker_id]:
                # Clean buy orders
                self.buy_orders[ticker_id] = [
                    o for o in self.buy_orders[ticker_id] if o.active
                ]
                # Clean sell orders
                self.sell_orders[ticker_id] = [
                    o for o in self.sell_orders[ticker_id] if o.active
                ]

# Thread-safe random order generator
def simulate_trading(exchange: StockExchange, thread_id: int, order_count: int) -> None:
    """Simulate trading by generating random orders"""
    print(f"Thread {thread_id} started, generating {order_count} orders...")
    
    # Set seed for this thread
    random.seed(time.time() + thread_id)
    
    for i in range(order_count):
        # Generate random order parameters
        order_type = OrderType.BUY if random.random() < 0.5 else OrderType.SELL
        ticker_id = random.randint(0, exchange.ticker_count - 1)
        ticker = exchange.tickers[ticker_id]
        quantity = random.randint(1, 1000)
        
        # Price varies by ticker but has some randomness
        base_price = (hash(ticker) % 10000) / 100  # Between $0 and $100
        variation = 0.9 + (random.random() * 0.2)  # +/- 10%
        price = round(base_price * variation, 2)
        
        try:
            # Add the order
            exchange.add_order(order_type, ticker, quantity, price)
            
            # Occasionally cleanup
            if random.random() < 0.02:  # ~2% chance
                exchange.cleanup_inactive_orders()
                
        except ValueError as e:
            print(f"Error in thread {thread_id}: {e}")
        
        # Small delay to reduce contention
        time.sleep(random.random() * 0.01)
    
    print(f"Thread {thread_id} completed.")

def print_order_book_stats(exchange: StockExchange, ticker: str) -> None:
    """Print statistics about an order book"""
    ticker_id = exchange.find_ticker_index(ticker)
    if ticker_id < 0:
        print(f"Invalid ticker: {ticker}")
        return
    
    with exchange.ticker_locks[ticker_id]:
        buy_orders = [o for o in exchange.buy_orders[ticker_id] if o.active]
        sell_orders = [o for o in exchange.sell_orders[ticker_id] if o.active]
    
    print(f"\nOrder book for {ticker}:")
    print(f"- Buy orders: {len(buy_orders)}")
    if buy_orders:
        print("  Top 3 buy orders:")
        # Sort by price (descending) and time (ascending)
        buy_orders.sort(key=lambda x: (-x.price, x.timestamp))
        for i, order in enumerate(buy_orders[:3]):
            print(f"    {i+1}. {order.quantity} shares at ${order.price:.2f}")
    
    print(f"- Sell orders: {len(sell_orders)}")
    if sell_orders:
        print("  Top 3 sell orders:")
        # Sort by price (ascending) and time (ascending)
        sell_orders.sort(key=lambda x: (x.price, x.timestamp))
        for i, order in enumerate(sell_orders[:3]):
            print(f"    {i+1}. {order.quantity} shares at ${order.price:.2f}")

# Main function to test the implementation
def main():
    # Initialize the stock exchange
    exchange = StockExchange(ticker_count=1024)
    
    print("Stock Trading Engine Simulation")
    print("-------------------------------")
    
    # Simulation parameters
    num_threads = 4
    orders_per_thread = 1000
    
    # Create threads
    threads = []
    start_time = time.time()
    
    for i in range(num_threads):
        thread = threading.Thread(
            target=simulate_trading,
            args=(exchange, i, orders_per_thread)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    elapsed_time = time.time() - start_time
    
    # Final cleanup
    exchange.cleanup_inactive_orders()
    
    # Print statistics
    total_transactions = len(exchange.transactions)
    print("\nSimulation complete:")
    print(f"- Duration: {elapsed_time:.2f} seconds")
    print(f"- Threads: {num_threads}")
    print(f"- Orders per thread: {orders_per_thread}")
    print(f"- Total orders: {num_threads * orders_per_thread}")
    print(f"- Transactions executed: {total_transactions}")
    print(f"- Transactions/second: {total_transactions / elapsed_time:.2f}")
    print(f"- Active orders remaining: {exchange.active_order_count.value}")
    
    # Show some sample order books
    sample_tickers = random.sample(exchange.tickers, min(5, exchange.ticker_count))
    for ticker in sample_tickers:
        print_order_book_stats(exchange, ticker)

if __name__ == "__main__":
    main()
