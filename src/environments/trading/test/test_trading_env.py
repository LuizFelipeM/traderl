from typing import Any, Final
import unittest
import numpy as np
import pandas as pd
from src.environments.trading import TradingEnv


csv_data = pd.read_csv("BTCUSDT-1s-2023-01.csv")

BUY: Final[int] = 0
SELL: Final[int] = 1
HOLD: Final[int] = 2


class Reward:
    def __init__(
        self, received: np.float32, expected: np.float32, it_should_be_equal=True
    ):
        self.received = received
        self.expected = expected
        self.it_should_be_equal = it_should_be_equal


class TradingEnvTestCase(unittest.TestCase):
    initial_balance: np.float32 = 10_000

    def setUp(self):
        self.env = TradingEnv(csv_data)
        self.env.reset()

    def test_initialization(self):
        self.assertEqual(self.env.initial_balance, self.initial_balance)
        self.assertEqual(self.env.balance, self.env.initial_balance)
        self.assertEqual(self.env.position, 0)
        self.assertEqual(self.env.current_step, 0)

    def test_reset(self):
        obs = self.env.reset()
        self.assertEqual(self.env.balance, self.initial_balance)
        self.assertEqual(self.env.position, 0)
        self.assertEqual(self.env.current_step, 0)
        self._test_observation(obs=obs)

    def test_step_buy(self):
        initial_balance = self.env.balance
        initial_position = self.env.position
        obs, reward, done, _ = self.env.step(BUY)
        self.assertGreater(self.env.position, initial_position)
        self.assertLess(self.env.balance, initial_balance)
        self._test_step(obs=obs, reward=Reward(reward, 0.0, False), done=done)

    def test_step_sell(self):
        # First buy to have something to sell
        self.env.step(BUY)
        initial_balance = self.env.balance
        initial_position = self.env.position
        obs, reward, done, _ = self.env.step(SELL)
        self.assertLess(self.env.position, initial_position)
        self.assertGreater(self.env.balance, initial_balance)
        self._test_step(obs=obs, reward=Reward(reward, 0.0, False), done=done)

    def test_step_hold(self):
        initial_balance = self.env.balance
        initial_position = self.env.position
        obs, reward, done, _ = self.env.step(HOLD)
        self.assertEqual(self.env.position, initial_position)
        self.assertEqual(self.env.balance, initial_balance)
        self._test_step(obs=obs, reward=Reward(reward, 0.0), done=done)

    def test_episode_end(self):
        for _ in range(300):
            self.env.step(2)
        self.assertTrue(self.env.done)
        self.assertEqual(self.env.position, 0)

    def test_liquidate_position(self):
        for _ in range(150):
            self.env.step(0)  # Buy until half episode
        for _ in range(150):
            self.env.step(2)  # Hold until the end
        self.assertTrue(self.env.done)
        self.assertEqual(self.env.position, 0)
        self.assertGreater(self.env.balance, 0)

    def _test_step(self, *, obs: Any, reward: Reward, done: bool):
        self.assertIsInstance(reward.received, np.float32)
        if reward.it_should_be_equal:
            self.assertEqual(reward.received, reward.expected)
        else:
            self.assertNotEqual(reward.received, reward.expected)

        self.assertFalse(done)
        self._test_observation(obs=obs)

    def _test_observation(self, *, obs):
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(len(obs), 6)


if __name__ == "__main__":
    unittest.main()
