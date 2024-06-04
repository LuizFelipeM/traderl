from typing import Any, Final, SupportsFloat
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.core import RenderFrame


SATOSHI: Final[np.int32] = 100_000_000


class TransactionFees:
    buy: np.float32
    sell: np.float32

    def __init__(self, *, buy: np.float32, sell: np.float32) -> None:
        self.buy = buy
        self.sell = sell


class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    """
    Define the render modes enabled for the environment
    """

    transaction_fee: TransactionFees
    """
    Buy and Sell Transaction fees charged in each transaction
    """

    initial_balance: np.float32 = 10_000
    """
    Initial balance used for resets
    """

    balance: np.float32 = initial_balance
    """
    Current balance in `USDT`
    """

    position: np.int32 = 0
    """
    Current open position in :const:`SATOSHI` (equivalent to `BTC / 100_000_000`)
    """

    pnl: np.float32 = 0
    """
    Profit and Loss metric used as agent's reward
    """

    current_step: np.int32 = 0
    """
    Current step of the Data Frame to keep track of the progress
    """

    done: bool = False
    """
    Define if the end was reached
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        batch_size=300,
        transaction_fees=TransactionFees(buy=0.025, sell=0.025),
    ) -> None:
        super(TradingEnv, self).__init__()

        self.data = data
        self.current_step = 0
        self.batch_size = batch_size
        self.transaction_fee = transaction_fees

        self.action_space = gym.spaces.Discrete(3)

        # Observations: [balance, position, low_price, high_price, open_price, close_price]
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32
        )

    @property
    def current_series(self) -> pd.Series:
        """
        Get data of the current step
        """
        return self.data.iloc[self.current_step]

    @property
    def current_satoshi_price(self) -> np.float32:
        """
        Get the current close price in :const:`SATOSHI`
        """
        return np.float32(self.current_series["close"] / SATOSHI)

    @property
    def net_worth(self) -> np.float32:
        """
        Metric used fo the agent's performance. It can be calculated through:

        net_worth = :attr:`balance` + (:attr:`position` * current_price)
        """
        return np.float32(self.balance + (self.position * self.current_satoshi_price))

    def reset(
        self, *, seed: np.int32 | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._reset_properties()
        return self._next_observation()

    def step(
        self, action: np.float32
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.done:
            return self._next_observation(), 0, self.done, self._get_info()

        match action:
            # Buy - All at once need improvement to do it calculated
            case 0:
                self._buy()

            # Sell - All at once need improvement to do it calculated
            case 1:
                self._sell()

            # Hold
            case 2:
                pass

            case _:
                raise ValueError(f"Invalid action {action}")

        self.current_step += 1

        if self.current_step >= self.batch_size or self.current_step >= len(self.data):
            self.done = True

            # Liquidate remaining position
            self._sell()

        self.pnl = np.float32(self.net_worth - self.initial_balance)
        return (
            self._next_observation(),
            # The pnl is playing the reward role here
            self.pnl,
            self.done,
            self._get_info(),
        )

    def render(self, mode="human") -> RenderFrame | list[RenderFrame] | None:
        if mode != "human":
            raise NotImplementedError(f"mode {mode} not implemented")

        print(
            f"Step {self.current_step} - Balance: {self.balance} | Position: {self.position} | PNL: {self.pnl}"
        )

    def _buy(self) -> None:
        if self.balance > 0 and self.balance >= self.current_satoshi_price:
            position = np.int32(self.balance / self.current_satoshi_price)
            # The real position is calculated through the deduction of transactions fee from the full position
            self.position = np.int32(position * (1 - self.transaction_fee.buy))
            self.balance -= np.float32(position * self.current_satoshi_price)

    def _sell(self) -> None:
        if self.position > 0:
            balance = np.float32(self.position * self.current_satoshi_price)
            # The real balance is calculated through the deduction of transactions fee from the full balance
            self.balance += np.float32(balance * (1 - self.transaction_fee.sell))
            self.position = 0

    def _reset_properties(self) -> None:
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.done = False

    def _next_observation(self) -> tuple[Any, dict[str, Any]]:
        """
        Return the next observation.

        Observations: [balance, position, low_price, high_price, open_price, close_price]
        """
        if self.current_step >= len(self.data):
            self.done = True
            return np.array([self.balance, self.position, 0, 0, 0, 0])

        return np.array(
            [
                self.balance,
                self.position,
                self.current_series["low"],
                self.current_series["high"],
                self.current_series["open"],
                self.current_series["close"],
            ]
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "net_worth": self.net_worth,
            "balance": self.balance,
            "position": self.position,
        }
