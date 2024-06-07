import numpy as np


class TransactionFees:
    buy: np.float32
    sell: np.float32

    def __init__(self, *, buy: np.float32, sell: np.float32) -> None:
        self.buy = buy
        self.sell = sell
