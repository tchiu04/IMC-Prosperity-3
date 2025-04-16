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
            acceptable_price = low_sell(order_depth)

            buy = abs(100 - position)
            sell = abs(100 + position)
            expected_price = expected_price_basket2(croissants_price, jams_price)
            
                
            spread_buy_pos = 0
            spread_sell_pos = 0
            
            if acceptable_price > expected_price:
                # Execute arbitrage: sell the basket (possibly short it) and buy the underlying components.
                buy_orders = sorted(order_depth.buy_orders.items(), reverse = True)
                for price, qty in buy_orders:
                    sell_qty = min(sell, qty)
                    submit_order(product, orders, price, -sell_qty)
                    sell -= sell_qty
                    spread_buy_pos += 1
                    if sell == 0:
                        break
                
                result[product] = orders
                '''#Long components
                for component in ['CROISSANTS', 'JAMS']:
                    comp_orders: List[Order] = []
                    comp_position = state.position[component] if state.position.get(component) else 0
                    if component == 'CROISSANTS':
                        comp_buy = abs(250 - comp_position)
                    elif component == 'JAMS':
                        comp_buy = abs(350 - comp_position)
                    cur_order_depth : OrderDepth = state.order_depths[component]
                    comp_sell_orders = sorted(cur_order_depth.sell_orders.items())
                    for price, qty in comp_sell_orders:
                        comp_buy_qty = min(comp_buy, -qty)
                        submit_order(component, comp_orders, price, comp_buy_qty)
                        comp_buy -= comp_buy_qty
                        spread_sell_pos += 1
                        if comp_buy == 0:
                            break
                    result[component] = comp_orders'''
            elif acceptable_price < expected_price:
                # Execute arbitrage: buy the basket and sell (or short) the underlying components.
                sell_orders = sorted(order_depth.sell_orders.items())
                for price, qty in sell_orders:
                    buy_qty = min(buy, -qty)
                    submit_order(product, orders, price, buy_qty)
                    buy -= buy_qty
                    spread_sell_pos += 1
                    if buy == 0:
                        break
                result[product] = orders
                #Short components
                '''for component in ['CROISSANTS', 'JAMS']:
                    comp_orders: List[Order] = []
                    comp_position = state.position[component] if state.position.get(component) else 0
                    if component == 'CROISSANTS':
                        comp_sell = abs(250 + comp_position)
                    elif component == 'JAMS':
                        comp_sell = abs(350 + comp_position)
                    cur_order_depth : OrderDepth = state.order_depths[component]
                    comp_buy_orders = sorted(cur_order_depth.buy_orders.items(), reverse = True)
                    for price, qty in comp_buy_orders:
                        comp_sell_qty = min(comp_sell, qty)
                        submit_order(component, orders, price, -comp_sell_qty)
                        comp_sell -= comp_sell_qty
                        spread_buy_pos += 1
                        if comp_sell == 0:
                            break
                    result[component] = comp_orders'''
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
            acceptable_price = low_sell(order_depth)

            buy = abs(60 - position)
            sell = abs(60 + position)
            expected_price = expected_price_basket1(croissants_price, jams_price, djembes_price)   
            
            #print("Acceptable price : " + str(acceptable_price))
            #print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

            spread_buy_pos = 0
            spread_sell_pos = 0
            
            if acceptable_price > expected_price:
                # Execute arbitrage: sell the basket (possibly short it) and buy the underlying components.
                buy_orders = sorted(order_depth.buy_orders.items(), reverse = True)
                for price, qty in buy_orders:
                    sell_qty = min(sell, qty)
                    submit_order(product, orders, price, -sell_qty)
                    sell -= sell_qty
                    spread_buy_pos += 1
                    if sell == 0:
                        break
                
                result[product] = orders
                '''#Long components
                for component in ['CROISSANTS', 'JAMS', 'DJEMBES']:
                    comp_orders: List[Order] = []
                    comp_position = state.position[component] if state.position.get(component) else 0
                    if component == 'CROISSANTS':
                        comp_buy = abs(250 - comp_position)
                    elif component == 'JAMS':
                        comp_buy = abs(350 - comp_position)
                    elif component == 'DJEMBES':
                        comp_buy = abs(60 - comp_position)
                    cur_order_depth : OrderDepth = state.order_depths[component]
                    
                    comp_sell_orders = sorted(cur_order_depth.sell_orders.items())
                    for price, qty in comp_sell_orders:
                        comp_buy_qty = min(comp_buy, -qty) 
                        submit_order(component, comp_orders, price, comp_buy_qty)
                        comp_buy -= comp_buy_qty
                        if comp_buy == 0:
                            break     
                    result[component] = comp_orders'''

            elif acceptable_price < expected_price:
                # Execute arbitrage: buy the basket and sell (or short) the underlying components.
                sell_orders = sorted(order_depth.sell_orders.items())
                for price, qty in sell_orders:
                    buy_qty = min(buy, -qty)
                    submit_order(product, orders, price, buy_qty)
                    buy -= buy_qty
                    spread_sell_pos += 1
                    if buy == 0:
                        break
                result[product] = orders

                #Short components
                '''for component in ['CROISSANTS', 'JAMS', 'DJEMBES']:
                    comp_orders: List[Order] = []
                    spread_buy_pos = 0
                    comp_position = state.position[component] if state.position.get(component) else 0
                    if component == 'CROISSANTS':
                        comp_sell = abs(250 + comp_position)
                    elif component == 'JAMS':
                        comp_sell = abs(350 + comp_position)
                    elif component == 'DJEMBES':
                        comp_sell = abs(60 + comp_position)
                    cur_order_depth : OrderDepth = state.order_depths[component]
                    comp_buy_orders = sorted(cur_order_depth.buy_orders.items(), reverse = True)
                    for price, qty in comp_buy_orders:
                        comp_sell_qty = min(comp_sell, qty)
                        submit_order(component, comp_orders, price, -comp_sell_qty)
                        comp_sell -= comp_sell_qty
                        spread_buy_pos += 1
                        if comp_sell == 0:
                            break  
                    result[component] = comp_orders'''
                    
    

    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    