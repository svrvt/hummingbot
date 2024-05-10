import logging
import math

from hummingbot.connector.utils import split_hb_trading_pair
from hummingbot.core.data_type.common import TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    BuyOrderCreatedEvent,
    MarketOrderFailureEvent,
    SellOrderCompletedEvent,
    SellOrderCreatedEvent,
)
from hummingbot.strategy.script_strategy_base import Decimal, OrderType, ScriptStrategyBase


class TriangularArbitrage(ScriptStrategyBase):
    """
    BotCamp Cohort: Sept 2022
    Design Template: https://hummingbot-foundation.notion.site/Triangular-Arbitrage-07ef29ee97d749e1afa798a024813c88
    Video: https://www.loom.com/share/b6781130251945d4b51d6de3f8434047
    Описание:
    Этот скрипт выполняет арбитражные сделки на 3 рынках одной и той же биржи, когда обнаруживается расхождение цен
    между этими рынками.

    - Все ордера исполняются линейно. То есть второй ордер выставляется после того, как первый
    и третий ордер размещается после второго.
    - Скрипт позволяет держать в инвентаре только один актив (holding_asset).
    - Он всегда начинает сделки с продажи удерживаемого актива и заканчивает их покупкой.
    - Существует 2 возможных направления арбитражных сделок: "прямое" и "обратное".
        Пример с холдинговым активом USDT:
        1. Прямая: купить ADA-USDT > продать ADA-BTC > продать BTC-USDT
        2. Обратный: купить BTC-USDT > купить ADA-BTC > продать ADA-USDT
    - Сумма ордера фиксирована и устанавливается в активе холдинга
    - Стратегия имеет проверку создания 2-го и 3-го ордеров и делает несколько попыток в случае неудачи
    - Прибыль рассчитывается каждый раунд и общая прибыль проверяется на kill_switch для предотвращения чрезмерных потерь
    - !!! При расчете прибыльности не учитываются торговые сборы, установите min_profitability не менее 3 * fee
    """
    # Config params
    connector_name: str = "binance"
    first_pair: str = "BNB-USDT"
    second_pair: str = "BNB-BTC"
    third_pair: str = "BTC-USDT"
    holding_asset: str = "USDT"

    min_profitability: Decimal = Decimal("0.5")
    order_amount_in_holding_asset: Decimal = Decimal("5")

    kill_switch_enabled: bool = True
    kill_switch_rate = Decimal("-2")

    # Class params
    status: str = "NOT_INIT"
    trading_pair: dict = {}
    order_side: dict = {}
    profit: dict = {}
    order_amount: dict = {}
    profitable_direction: str = ""
    place_order_trials_count: int = 0
    place_order_trials_limit: int = 10
    place_order_failure: bool = False
    order_candidate = None
    initial_spent_amount = Decimal("0")
    total_profit = Decimal("0")
    total_profit_pct = Decimal("0")

    markets = {connector_name: {first_pair, second_pair, third_pair}}

    @property
    def connector(self):
        """
        Единственный коннектор в этой стратегии, определите его здесь для удобства доступа
        """
        return self.connectors[self.connector_name]

    def on_tick(self):
        """
        Каждый тик стратегия рассчитывает прибыльность как прямого, так и обратного направления.
        Если прибыльность любого направления достаточно велика, она начинает арбитраж, создавая и обрабатывая
        первого ордера-кандидата.
        """
        if self.status == "NOT_INIT":
            self.init_strategy()

        if self.arbitrage_started():
            return

        if not self.ready_for_new_orders():
            return

        self.profit["direct"], self.order_amount["direct"] = self.calculate_profit(self.trading_pair["direct"],
                                                                                   self.order_side["direct"])
        self.profit["reverse"], self.order_amount["reverse"] = self.calculate_profit(self.trading_pair["reverse"],
                                                                                     self.order_side["reverse"])
        self.log_with_clock(logging.INFO, f"Profit direct: {round(self.profit['direct'], 2)}, "
                                          f"Profit reverse: {round(self.profit['reverse'], 2)}")

        if self.profit["direct"] < self.min_profitability and self.profit["reverse"] < self.min_profitability:
            return

        self.profitable_direction = "direct" if self.profit["direct"] > self.profit["reverse"] else "reverse"
        self.start_arbitrage(self.trading_pair[self.profitable_direction],
                             self.order_side[self.profitable_direction],
                             self.order_amount[self.profitable_direction])

    def init_strategy(self):
        """
        Инициализирует стратегию один раз перед запуском.
        """
        self.status = "ACTIVE"
        self.check_trading_pair()
        self.set_trading_pair()
        self.set_order_side()

    def check_trading_pair(self):
        """
        Проверяет, подходят ли пары, указанные в конфиге, для треугольного арбитража.
        Они должны иметь только 3 общих актива, среди которых есть hold_asset.
        """
        base_1, quote_1 = split_hb_trading_pair(self.first_pair)
        base_2, quote_2 = split_hb_trading_pair(self.second_pair)
        base_3, quote_3 = split_hb_trading_pair(self.third_pair)
        all_assets = {base_1, base_2, base_3, quote_1, quote_2, quote_3}
        if len(all_assets) != 3 or self.holding_asset not in all_assets:
            self.status = "NOT_ACTIVE"
            self.log_with_clock(logging.WARNING, f"Pairs {self.first_pair}, {self.second_pair}, {self.third_pair} "
                                                 f"не подходят для треугольного арбитража!")

    def set_trading_pair(self):
        """
        Переставьте торговые пары так, чтобы первая и последняя пара содержала холдинговый актив.
        Мы начинаем торговый раунд с продажи холдингового актива и заканчиваем его покупкой.
        Создает 2 кортежа для "прямого" и "обратного" направлений и присваивает их соответствующему словарю.
        """
        if self.holding_asset not in self.first_pair:
            pairs_ordered = (self.second_pair, self.first_pair, self.third_pair)
        elif self.holding_asset not in self.second_pair:
            pairs_ordered = (self.first_pair, self.second_pair, self.third_pair)
        else:
            pairs_ordered = (self.first_pair, self.third_pair, self.second_pair)

        self.trading_pair["direct"] = pairs_ordered
        self.trading_pair["reverse"] = pairs_ordered[::-1]

    def set_order_side(self):
        """
        Устанавливает стороны ордеров (1 = buy, 0 = sell) для уже упорядоченных торговых пар.
        Создает 2 кортежа для " direct" и " reverse" направлений и присваивает их соответствующему словарю.
        """
        base_1, quote_1 = split_hb_trading_pair(self.trading_pair["direct"][0])
        base_2, quote_2 = split_hb_trading_pair(self.trading_pair["direct"][1])
        base_3, quote_3 = split_hb_trading_pair(self.trading_pair["direct"][2])

        order_side_1 = 0 if base_1 == self.holding_asset else 1
        order_side_2 = 0 if base_1 == base_2 else 1
        order_side_3 = 1 if base_3 == self.holding_asset else 0

        self.order_side["direct"] = (order_side_1, order_side_2, order_side_3)
        self.order_side["reverse"] = (1 - order_side_3, 1 - order_side_2, 1 - order_side_1)

    def arbitrage_started(self) -> bool:
        """
        Проверяет наличие незавершенного арбитражного раунда.
        В случае неудачи при размещении 2-го или 3-го ордера пытается разместить ордер снова
        пока не будет достигнут лимит place_order_trials_limit.
        """
        if self.status == "ARBITRAGE_STARTED":
            if self.order_candidate and self.place_order_failure:
                if self.place_order_trials_count <= self.place_order_trials_limit:
                    self.log_with_clock(logging.INFO, f"Failed to place {self.order_candidate.trading_pair} "
                                                      f"{self.order_candidate.order_side} order. Trying again!")
                    self.process_candidate(self.order_candidate, True)
                else:
                    msg = f"Error placing {self.order_candidate.trading_pair} {self.order_candidate.order_side} order"
                    self.notify_hb_app_with_timestamp(msg)
                    self.log_with_clock(logging.WARNING, msg)
                    self.status = "NOT_ACTIVE"
            return True

        return False

    def ready_for_new_orders(self) -> bool:
        """
        Проверяет, готовы ли мы к приему новых заказов:
        - Проверка текущего состояния
        - Проверка баланса активов в холдинге
        Возвращает булево значение True, если мы готовы, и False в противном случае
        """
        if self.status == "NOT_ACTIVE":
            return False

        if self.connector.get_available_balance(self.holding_asset) < self.order_amount_in_holding_asset:
            self.log_with_clock(logging.INFO,
                                f"{self.connector_name} {self.holding_asset} баланс слишком низкий. Невозможно оформить заказ.")
            return False

        return True

    def calculate_profit(self, trading_pair, order_side):
        """
        Рассчитывает прибыльность и суммы ордеров для 3 торговых пар на основе глубины портфеля ордеров.
        """
        exchanged_amount = self.order_amount_in_holding_asset
        order_amount = [0, 0, 0]

        for i in range(3):
            order_amount[i] = self.get_order_amount_from_exchanged_amount(trading_pair[i], order_side[i],
                                                                          exchanged_amount)
            # Update exchanged_amount for the next cycle
            if order_side[i]:
                exchanged_amount = order_amount[i]
            else:
                exchanged_amount = self.connector.get_quote_volume_for_base_amount(trading_pair[i], order_side[i],
                                                                                   order_amount[i]).result_volume
        start_amount = self.order_amount_in_holding_asset
        end_amount = exchanged_amount
        profit = (end_amount / start_amount - 1) * 100

        return profit, order_amount

    def get_order_amount_from_exchanged_amount(self, pair, side, exchanged_amount) -> Decimal:
        """
        Рассчитывает сумму ордера, используя сумму, которую мы хотим обменять.
        - Если сторона - покупка, то обмениваемый актив - это котируемый актив. Получение базовой суммы с помощью книги заявок
        - Если сторона - продажа, то обмениваемый актив - базовый актив.
        """
        if side:
            orderbook = self.connector.get_order_book(pair)
            order_amount = self.get_base_amount_for_quote_volume(orderbook.ask_entries(), exchanged_amount)
        else:
            order_amount = exchanged_amount

        return order_amount

    def get_base_amount_for_quote_volume(self, orderbook_entries, quote_volume) -> Decimal:
        """
        Рассчитывает базовую сумму, которую вы получаете за объем котировок, используя записи в журнале заказов
        """
        cumulative_volume = 0.
        cumulative_base_amount = 0.
        quote_volume = float(quote_volume)

        for order_book_row in orderbook_entries:
            row_amount = order_book_row.amount
            row_price = order_book_row.price
            row_volume = row_amount * row_price
            if row_volume + cumulative_volume >= quote_volume:
                row_volume = quote_volume - cumulative_volume
                row_amount = row_volume / row_price
            cumulative_volume += row_volume
            cumulative_base_amount += row_amount
            if cumulative_volume >= quote_volume:
                break

        return Decimal(cumulative_base_amount)

    def start_arbitrage(self, trading_pair, order_side, order_amount):
        """
        Начинает арбитраж с создания и обработки первого кандидата на заказ
        """
        first_candidate = self.create_order_candidate(trading_pair[0], order_side[0], order_amount[0])
        if first_candidate:
            if self.process_candidate(first_candidate, False):
                self.status = "ARBITRAGE_STARTED"

    def create_order_candidate(self, pair, side, amount):
        """
        Создает кандидата на заказ. Проверяет квантованное количество
        """
        side = TradeType.BUY if side else TradeType.SELL
        price = self.connector.get_price_for_volume(pair, side, amount).result_price
        price_quantize = self.connector.quantize_order_price(pair, Decimal(price))
        amount_quantize = self.connector.quantize_order_amount(pair, Decimal(amount))

        if amount_quantize == Decimal("0"):
            self.log_with_clock(logging.INFO, f"Сумма заказа на {pair} слишком мала для размещения ордера")
            return None

        return OrderCandidate(
            trading_pair=pair,
            is_maker=False,
            order_type=OrderType.MARKET,
            order_side=side,
            amount=amount_quantize,
            price=price_quantize)

    def process_candidate(self, order_candidate, multiple_trials_enabled) -> bool:
        """
        Проверяет баланс кандидата на заказ и либо размещает заказ, либо устанавливает отказ на следующие испытания
        """
        order_candidate_adjusted = self.connector.budget_checker.adjust_candidate(order_candidate, all_or_none=True)
        if math.isclose(order_candidate.amount, Decimal("0"), rel_tol=1E-6):
            self.logger().info(f"Скорректированная сумма заказа: {order_candidate.amount} on {order_candidate.trading_pair}, "
                               f"слишком низкая для размещения заказа")
            if multiple_trials_enabled:
                self.place_order_trials_count += 1
                self.place_order_failure = True
            return False
        else:
            is_buy = True if order_candidate.order_side == TradeType.BUY else False
            self.place_order(self.connector_name,
                             order_candidate.trading_pair,
                             is_buy,
                             order_candidate_adjusted.amount,
                             order_candidate.order_type,
                             order_candidate_adjusted.price)
            return True

    def place_order(self,
                    connector_name: str,
                    trading_pair: str,
                    is_buy: bool,
                    amount: Decimal,
                    order_type: OrderType,
                    price=Decimal("NaN"),
                    ):
        if is_buy:
            self.buy(connector_name, trading_pair, amount, order_type, price)
        else:
            self.sell(connector_name, trading_pair, amount, order_type, price)

    # Events
    def did_create_buy_order(self, event: BuyOrderCreatedEvent):
        self.log_with_clock(logging.INFO, f"На рынке создается ордер на покупку {event.trading_pair}")
        if self.order_candidate:
            if self.order_candidate.trading_pair == event.trading_pair:
                self.reset_order_candidate()

    def did_create_sell_order(self, event: SellOrderCreatedEvent):
        self.log_with_clock(logging.INFO, f"На рынке создается ордер на продажу {event.trading_pair}")
        if self.order_candidate:
            if self.order_candidate.trading_pair == event.trading_pair:
                self.reset_order_candidate()

    def reset_order_candidate(self):
        """
        Удаление переменной-кандидата порядка и сброс счетчика
        """
        self.order_candidate = None
        self.place_order_trials_count = 0
        self.place_order_failure = False

    def did_fail_order(self, event: MarketOrderFailureEvent):
        if self.order_candidate:
            self.place_order_failure = True

    def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
        msg = f"Buy {round(event.base_asset_amount, 6)} {event.base_asset} " \
              f"for {round(event.quote_asset_amount, 6)} {event.quote_asset} is completed"
        self.notify_hb_app_with_timestamp(msg)
        self.log_with_clock(logging.INFO, msg)
        self.process_next_pair(event)

    def did_complete_sell_order(self, event: SellOrderCompletedEvent):
        msg = f"Sell {round(event.base_asset_amount, 6)} {event.base_asset} " \
              f"for {round(event.quote_asset_amount, 6)} {event.quote_asset} is completed"
        self.notify_hb_app_with_timestamp(msg)
        self.log_with_clock(logging.INFO, msg)
        self.process_next_pair(event)

    def process_next_pair(self, order_event):
        """
        Обрабатывает 2-й или 3-й ордер и завершает арбитраж
        - Получает индекс завершенного ордера
        - Рассчитывает сумму ордера
        - Создает и обрабатывает ордер-кандидат
        - Завершает арбитраж, если 3-й ордер был завершен
        """
        event_pair = f"{order_event.base_asset}-{order_event.quote_asset}"
        trading_pair = self.trading_pair[self.profitable_direction]
        order_side = self.order_side[self.profitable_direction]

        event_order_index = trading_pair.index(event_pair)

        if order_side[event_order_index]:
            exchanged_amount = order_event.base_asset_amount
        else:
            exchanged_amount = order_event.quote_asset_amount

        # Сохраните первоначально потраченную сумму для дальнейшего расчета прибыли
        if event_order_index == 0:
            self.initial_spent_amount = order_event.quote_asset_amount if order_side[event_order_index] \
                else order_event.base_asset_amount

        if event_order_index < 2:
            order_amount = self.get_order_amount_from_exchanged_amount(trading_pair[event_order_index + 1],
                                                                       order_side[event_order_index + 1],
                                                                       exchanged_amount)
            self.order_candidate = self.create_order_candidate(trading_pair[event_order_index + 1],
                                                               order_side[event_order_index + 1], order_amount)
            if self.order_candidate:
                self.process_candidate(self.order_candidate, True)
        else:
            self.finalize_arbitrage(exchanged_amount)

    def finalize_arbitrage(self, final_exchanged_amount):
        """
        Завершает арбитраж
        - Рассчитывает прибыль торгового раунда
        - Обновляет общую прибыль
        - Проверяет порог срабатывания выключателя
        """
        order_profit = round(final_exchanged_amount - self.initial_spent_amount, 6)
        order_profit_pct = round(100 * order_profit / self.initial_spent_amount, 2)
        msg = f"*** Arbitrage completed! Profit: {order_profit} {self.holding_asset} ({order_profit_pct})%"
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

        self.total_profit += order_profit
        self.total_profit_pct = round(100 * self.total_profit / self.order_amount_in_holding_asset, 2)
        self.status = "ACTIVE"
        if self.kill_switch_enabled and self.total_profit_pct < self.kill_switch_rate:
            self.status = "NOT_ACTIVE"
            self.log_with_clock(logging.INFO, "Достигнут порог срабатывания выключателя. Прекращение торговли")
            self.notify_hb_app_with_timestamp("Достигнут порог срабатывания выключателя. Прекращение торговли")

    def format_status(self) -> str:
        """
        Возвращает статус текущей стратегии, общую прибыль, текущую прибыльность возможных сделок и баланс.
        Эта функция вызывается при выдаче команды status.
        """
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        lines = []
        warning_lines = []
        warning_lines.extend(self.network_warning(self.get_market_trading_pair_tuples()))

        lines.extend(["", "  Strategy status:"] + ["    " + self.status])

        lines.extend(["", "  Total profit:"] + ["    " + f"{self.total_profit} {self.holding_asset}"
                                                         f"({self.total_profit_pct}%)"])

        for direction in self.trading_pair:
            pairs_str = [f"{'buy' if side else 'sell'} {pair}"
                         for side, pair in zip(self.order_side[direction], self.trading_pair[direction])]
            pairs_str = " > ".join(pairs_str)
            profit_str = str(round(self.profit[direction], 2))
            lines.extend(["", f"  {direction.capitalize()}:", f"    {pairs_str}", f"    profitability: {profit_str}%"])

        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        try:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active orders."])

        if self.connector.get_available_balance(self.holding_asset) < self.order_amount_in_holding_asset:
            warning_lines.extend(
                [f"{self.connector_name} {self.holding_asset} баланс слишком низкий. Не могу разместить ордер."])

        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)

        return "\n".join(lines)
