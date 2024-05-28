import pandas as pd
from environments.trading import TradingEnv


env = TradingEnv(pd.read_csv('BTCUSDT-1s-2023-01.csv'))

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    env.render()