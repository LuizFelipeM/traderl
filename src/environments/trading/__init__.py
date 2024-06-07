from environments.trading.trading_env import TradingEnv
from environments.trading.transaction_fees import TransactionFees
from gymnasium.envs.registration import register

__all__ = ["TradingEnv", "TransactionFees"]


def register_trading_env() -> None:
    register(
        id="TradingEnv-v0",  # Unique identifier for the environment
        entry_point="environments.trading.trading_env:TradingEnv",  # Module path and class name
        max_episode_steps=300,  # Maximum number of steps per episode
    )


# register_trading_env()

# To make this environment avilable through package
# from setuptools import setup, find_packages

# setup(
#     name='trading_gym',
#     version='0.1',
#     packages=find_packages(),
#     install_requires=[
#         'gymnasium',
#         'numpy',
#         'pandas'
#     ],
#     entry_points={
#         'gymnasium.envs': [
#             'TradingEnv-v0 = trading.trading_env:TradingEnv'
#         ],
#     },
# )
