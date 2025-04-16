from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import pandas as pd
import numpy as np
import statistics
import math
import typing
import jsonpickle
import json
from typing import Any
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

n = 10

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

def calc_price(depths):
    asks_sum = 0
    asks_qty = 0
    bid_sum = 0
    bid_qty = 0
    for k, w in depths.sell_orders.items():
        asks_sum -= k * w
        asks_qty -= w
    for k, w in depths.buy_orders.items():
        bid_sum += k * w
        bid_qty += w

    return ((asks_sum/asks_qty) + (bid_sum/bid_qty))/2

def calc_vwap(depths):
    sum = 0
    qty = 0
    for k, w in depths.sell_orders.items():
        sum -= k * w
        qty -= w
    for k, w in depths.buy_orders.items():
        sum += k * w
        qty += w

    return sum/qty

def calc_mm_mid_price(depths):
    sum = 0
    qty = 0
    for k, w in depths.sell_orders.items():
        if w <= -12:
            sum -= k * w
            qty -= w
    for k, w in depths.buy_orders.items():
        if w >= 12:
            sum += k * w
            qty += w

    return sum/qty


def expected_price_basket1(croissants_price, jams_price, djembes_price):
        # Calculate the expected price of the gift basket based on its components
        return (6 * croissants_price) + (3 * jams_price) + djembes_price

def expected_price_basket2(croissants_price, jams_price):
        # Calculate the expected price of the gift basket based on its components
        return (4 * croissants_price) + (2 * jams_price)

def low_sell(depths):
    if depths.sell_orders:
        return min(depths.sell_orders.keys())
    return float('inf')

def cdf(x):
    """Approximation of the cumulative distribution function for the standard normal distribution."""
    # Constants for the approximation
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911

    # Save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2.0)

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * math.exp(-x*x)

    return 0.5 * (1.0 + sign * y)

#Black Scholes Part
def black_scholes_price(self, current_price, time_to_maturity, strike_price, sigma, premium = 0):
    if current_price <= 0 or strike_price <= 0 or time_to_maturity <= 0 or sigma <= 0:
        return 0
    S = current_price
    K = strike_price
    T = time_to_maturity / 365
    r = 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * cdf(d1) - (K + premium) * np.exp(-r * T) * cdf(d2))
    
    return call_price


logger = Logger()

def submit_order(product, orders, price, volume):
    if volume > 0:
        logger.print("BUY ", str(volume) + "x", round(price))
    else:
        logger.print("SELL ", str(-volume) + "x", round(price))
    orders.append(Order(product, round(price), volume))


def find_ema(curr_price, values, period):
    data = pd.Series(values)
    multiplier = 2/(period + 1)
    return (curr_price * multiplier) + (data.iloc[-1] * (1-multiplier))
    #return data.ewm(span=period, adjust=True).mean().iloc[-1]

class Trader:

    symbols = ['KELP', 'RAINFOREST_RESIN', 'SQUID_INK', 'PICNIC_BASKET1','PICNIC_BASKET2', 'CROISSANTS', 'JAMS', 'DJEMBES']
    positionHistory = {symbol: [] for symbol in symbols}
    priceHistory = {symbol: [] for symbol in symbols}
    emah = {symbol: [] for symbol in symbols}
    historical_spreads = {}

    def run(self, state: TradingState):
        #print("traderData: " + state.traderData)
        #print("Observations: " + str(state.observations))
        result = {}
        timestamp = state.timestamp

        for product in ['KELP', 'RAINFOREST_RESIN', 'SQUID_INK']:
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position[product] if state.position.get(product) else 0
            orders: List[Order] = []

            ph = self.positionHistory[product]
            emah = self.emah[product]
            ph.append(position)
            sma_curr = np.mean(ph[max(0, len(ph) - n) : len(ph)])
            if len(ph) != 1:
                sma_prev = np.mean(ph[max(0, len(ph) - n - 1) : (len(ph) - 1)])
            else:
                sma_prev = 0
            trend = sma_curr - sma_prev
            logger.print(trend)

            if product == 'KELP':
                acceptable_price = calc_vwap(order_depth)

            elif product == 'SQUID_INK' and timestamp == 0:
                acceptable_price = calc_price(order_depth)
                emah.append(acceptable_price)
            elif product == 'SQUID_INK':
                acceptable_price = calc_price(order_depth)
            else:
                acceptable_price = 10000

            #print("Acceptable price : " + str(acceptable_price))
            #print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

            buy_orders = sorted(order_depth.buy_orders.items(), reverse = True)
            sell_orders = sorted(order_depth.sell_orders.items())

            buy = abs(50 - position)
            sell = abs(50 + position)

            spread_buy_pos = 0
            spread_sell_pos = 0

            if product == 'SQUID_INK':
                priceHist = self.priceHistory['SQUID_INK']
                priceHist.append(acceptable_price)
                e = 200
                ema = find_ema(calc_vwap(order_depth), emah, e)
                emah.append(ema)
                sd = np.std(priceHist[max(0, len(priceHist) - e) : len(priceHist)])
                if sd == 0 or np.isnan(sd):
                    sd = 1
                z = (acceptable_price - ema)/sd
                x = 2
                zlim = 1.5
                if z > zlim:
                    submit_order(product, orders, acceptable_price + x, -sell)
                elif z < -zlim:
                    submit_order(product, orders, acceptable_price - x, buy)
                '''
                elif abs(z) < 0.5:
                    # clear position
                    if position > 0:
                        submit_order(product, orders, acceptable_price + x, -position)
                    else:
                        submit_order(product, orders, acceptable_price - x, -position)
                '''
            
            else:
                # +EV ORDER MATCH:
                for price, qty in sell_orders:
                    if price < acceptable_price or (price == acceptable_price and (position <= 0 or trend < 0)): # or is better for resin, and is better for kelp
                        buy_qty = min(buy, -qty)
                        submit_order(product, orders, price, buy_qty)
                        buy -= buy_qty
                        spread_sell_pos += 1
                        if buy == 0:
                            break

                for price, qty in buy_orders:
                    if price > acceptable_price or (price == acceptable_price and (position >= 0 or trend > 0)): #  or is better for resin, and is better for kelp
                        sell_qty = min(sell, qty)
                        submit_order(product, orders, price, -sell_qty)
                        sell -= sell_qty
                        spread_buy_pos += 1
                        if sell == 0:
                            break

                # MARKET MAKING:
                if len(buy_orders) > spread_buy_pos:
                    best_buy = buy_orders[spread_buy_pos][0]
                else:
                    best_buy = acceptable_price - 5

                if len(sell_orders) > spread_sell_pos:
                    best_sell = sell_orders[spread_sell_pos][0]
                else:
                    best_sell = acceptable_price + 5

                if buy != 0 and best_buy <= acceptable_price:
                    if abs(best_buy - acceptable_price) <= 1:
                        if position < 0:
                            submit_order(product, orders, acceptable_price, -position)
                            buy += position
                        submit_order(product, orders, best_buy, buy)
                    else:
                        submit_order(product, orders, best_buy + 1, buy)

                if sell != 0 and best_sell >= acceptable_price:
                    if abs(best_sell - acceptable_price) <= 1:
                        if position > 0:
                            submit_order(product, orders, acceptable_price, -position)
                            sell -= position
                        submit_order(product, orders, best_sell, -sell)
                    else:
                        submit_order(product, orders, best_sell - 1, -sell)

            result[product] = orders
        
        '''for product in ['VOLCANIC_ROCK']:
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position[product] if state.position.get(product) else 0
            orders: List[Order] = []

            #Prices of vouchers     
            order_depth_9500: OrderDepth = state.order_depths['VOLCANIC_ROCK_VOUCHER_9500']       
            price_9500 = low_sell(order_depth_9500)
            order_depth_9750: OrderDepth = state.order_depths['VOLCANIC_ROCK_VOUCHER_9750']       
            price_9750 = low_sell(order_depth_9750)
            order_depth_10000: OrderDepth = state.order_depths['VOLCANIC_ROCK_VOUCHER_10000']       
            price_10000 = low_sell(order_depth_10000)
            order_depth_12500: OrderDepth = state.order_depths['VOLCANIC_ROCK_VOUCHER_12500']       
            price_12500 = low_sell(order_depth_12500)
            order_depth_15000: OrderDepth = state.order_depths['VOLCANIC_ROCK_VOUCHER_15000']       
            price_15000 = low_sell(order_depth_15000)'''



            

        for product in ['VOLCANIC_ROCK_VOUCHER_9500', 'VOLCANIC_ROCK_VOUCHER_9750','VOLCANIC_ROCK_VOUCHER_10000',
                        'VOLCANIC_ROCK_VOUCHER_10250','VOLCANIC_ROCK_VOUCHER_10500']:
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position[product] if state.position.get(product) else 0
            orders: List[Order] = []

            buy = abs(200 - position)
            sell = abs(200 + position)

            rock_order_depth: OrderDepth = state.order_depths['VOLCANIC_ROCK']
            sigma = 0.2
            maturity = 7
            
            current_price = low_sell(rock_order_depth)

            '''market_trades = state.market_trades
            own_trades = state.own_trades

            prod_trades: List[Trade] = own_trades.get(self.name, []) + market_trades.get(self.name, [])

            if len(prod_trades) > 0:
                prices = [(trade.quantity, trade.price) for trade in prod_trades]
                cached_prices = prices
            else:
                cached_prices = 0
            
            lambda_ = 0.94  # Decay factor for EWMA, common choice in finance
            if len(cached_prices) > 1:
                returns = np.diff(cached_prices) / cached_prices[:-1]
                var = np.var(returns)
                sigma = np.sqrt(lambda_ * self.sigma**2 + (1 - lambda_) * var) * np.sqrt(252)'''
            if(product == 'VOLCANIC_ROCK_VOUCHER_9500'):
                strike = 9500
            elif(product == 'VOLCANIC_ROCK_VOUCHER_9750'):
                strike = 9750
            elif(product == 'VOLCANIC_ROCK_VOUCHER_10000'):
                strike = 10000
            elif(product == 'VOLCANIC_ROCK_VOUCHER_12500'):
                strike = 12500
            elif(product == 'VOLCANIC_ROCK_VOUCHER_15000'):
                strike = 15000
            
            theoretical_price = black_scholes_price(current_price, maturity, strike, sigma, 0)

            best_asks = sorted(order_depth.sell_orders.keys())
            best_bids = sorted(order_depth.buy_orders.keys(), reverse=True)

            # Go through the asks and determine if we should buy
            for i, ask_price in enumerate(best_asks):
                if ask_price < theoretical_price:
                    buy_qty = min(buy, -i)
                    submit_order(product, orders, ask_price, buy_qty)
                    buy -= buy_qty
                    if buy == 0:
                        break  

            # Go through the bids and determine if we should sell
            for i, bid_price in enumerate(best_bids):
                if bid_price > theoretical_price:
                    sell_qty = min(sell, i)
                    submit_order(product, orders, bid_price, -sell_qty)
                    sell -= sell_qty
                    if sell == 0:
                        break
            
            result[product] = orders

            
        
        for product in ['PICNIC_BASKET2']:
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position[product] if state.position.get(product) else 0
            orders: List[Order] = []
            
            #Prices of components     
            croissants_order_depth: OrderDepth = state.order_depths['CROISSANTS']       
            croissants_price = low_sell(croissants_order_depth)
            jams_order_depth: OrderDepth = state.order_depths['JAMS']    
            jams_price = low_sell(jams_order_depth)

            #Gift Basket Price

            current_expected = expected_price_basket2(croissants_price, jams_price)
            price_hist = self.priceHistory['PICNIC_BASKET2']
            price_hist.append(current_expected)
            if len(price_hist) > n:
                price_hist.pop(0)

            smoothed_expected = np.mean(price_hist)
            acceptable_price = low_sell(order_depth)
            buy = abs(100 - position)
            sell = abs(100 + position)

            if(acceptable_price > current_expected):
                buy_orders = sorted(order_depth.buy_orders.items(), reverse = True)
                for price, qty in buy_orders:
                    sell_qty = min(sell, qty)
                    submit_order(product, orders, price, -sell_qty)
                    sell -= sell_qty
                    if sell == 0:
                        break
            elif(acceptable_price < current_expected):
                for price, qty in sell_orders:
                    buy_qty = min(buy, -qty)
                    submit_order(product, orders, price, buy_qty)
                    buy -= buy_qty
                    if buy == 0:
                        break     
            result[product] = orders
            
        for product in ['PICNIC_BASKET1']:
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position[product] if state.position.get(product) else 0
            orders: List[Order] = []
           
            #Prices of components     
            croissants_order_depth: OrderDepth = state.order_depths['CROISSANTS']       
            croissants_price = low_sell(croissants_order_depth)
            jams_order_depth: OrderDepth = state.order_depths['JAMS']    
            jams_price = low_sell(jams_order_depth)
            djembes_order_depth: OrderDepth = state.order_depths['DJEMBES']    
            djembes_price = low_sell(djembes_order_depth)

            #Gift Basket Price
            buy = abs(60 - position)
            sell = abs(60 + position)
            acceptable_price = low_sell(order_depth)
            expected_price = expected_price_basket1(croissants_price, jams_price, djembes_price)   

            if(acceptable_price > expected_price):
                buy_orders = sorted(order_depth.buy_orders.items(), reverse = True)
                for price, qty in buy_orders:
                    sell_qty = min(sell, qty)
                    submit_order(product, orders, price, -sell_qty)
                    sell -= sell_qty
                    if sell == 0:
                        break
            elif(acceptable_price < expected_price):
                for price, qty in sell_orders:
                    buy_qty = min(buy, -qty)
                    submit_order(product, orders, price, buy_qty)
                    buy -= buy_qty
                    if buy == 0:
                        break
            result[product] = orders


    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    