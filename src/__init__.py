import pandas as pd
import gymnasium as gym
from environments.trading import register_trading_env

register_trading_env()

data = pd.read_csv("BTCUSDT-1s-2023-01.csv")
env = gym.vector.make("TradingEnv-v0", num_envs=2, data=data)

print(env.reset())

action = env.action_space.sample()
print(action)
print(env.step(action))
