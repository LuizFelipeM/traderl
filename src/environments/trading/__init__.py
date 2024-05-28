from typing import Any, Final, SupportsFloat
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.core import RenderFrame


SATOSHI: Final[np.int32] = 100_000_000
INITIAL_BALANCE: Final[np.int32] = 10_000


class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    """
    Define the render modes enabled for the environment
    """

    balance: np.float32 = INITIAL_BALANCE
    """
    Current balance in USDT
    """

    position: np.int32 = 0
    """
    Current open position in :const:`SATOSHI` (equivalent to BTC / 100_000_000)
    """

    profit: np.float32 = 0
    """
    Profit metric used as agent's reward
    """

    current_step: np.int32 = 0
    """
    Current step of the Data Frame to keep track of the progress
    """

    done: bool = False
    """
    Define if the end was reached
    """

    def __init__(self, data: pd.DataFrame, batch_size=300) -> None:
        super(TradingEnv, self).__init__()

        self.data = data
        self.current_step = 0
        self.batch_size = batch_size

        self.action_space = gym.spaces.Discrete(3)

        # Observations: [balance, position, low_price, high_price, open_price, close_price]
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32
        )
        # self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(len(data.columns),), dtype=np.float32)

    @property
    def current_satoshi_price(self) -> np.float32:
        """
        Converts the current close price to :const:`SATOSHI` equivalent
        """
        return np.float32(self.data.iloc[self.current_step]["close"] / SATOSHI)

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

        self.current_step += 1

        if self.current_step >= self.batch_size or self.current_step >= len(self.data):
            self.done = True

            # Liquidate remaining position
            self._sell()

        self.profit = np.float32(self.net_worth - INITIAL_BALANCE)
        return (
            self._next_observation(),
            # The profit is playing the reward role here
            self.profit,
            self.done,
            self._get_info(),
        )

    def render(self, mode="human") -> RenderFrame | list[RenderFrame] | None:
        if mode != "human":
            raise NotImplementedError(f"mode {mode} not implemented")

        print(
            f"Step {self.current_step} - Balance: {self.balance} | Position: {self.position} | Profit: {self.profit}"
        )

    def _buy(self) -> None:
        if self.balance > 0 and self.balance >= self.current_satoshi_price:
            self.position = np.int32(self.balance / self.current_satoshi_price)
            self.balance -= np.float32(self.position * self.current_satoshi_price)
            # self.balance = 0

    def _sell(self) -> None:
        if self.position > 0:
            self.balance += self.position * self.current_satoshi_price
            self.position = 0

    def _reset_properties(self) -> None:
        self.balance = INITIAL_BALANCE
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

        current_step = self.data.iloc[self.current_step]
        return np.array(
            [
                self.balance,
                self.position,
                current_step["low"],
                current_step["high"],
                current_step["open"],
                current_step["close"],
            ]
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "net_worth": self.net_worth,
            "balance": self.balance,
            "position": self.position,
        }
