import numpy as np
from talib import stream

def compute_delta_ma(hist_last_price, timeperiod):
    return stream.SMA(
        np.fromiter(hist_last_price, float),
        timeperiod=timeperiod
        ) - hist_last_price[-1]

def compute_atr(hist_last_price, hist_high_price, hist_low_price, timeperiod):
    return stream.NATR(
        np.fromiter(hist_high_price, float),
        np.fromiter(hist_low_price, float),
        np.fromiter(hist_last_price, float),
        timeperiod=timeperiod
        )