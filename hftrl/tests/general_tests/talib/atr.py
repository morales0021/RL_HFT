import tqdm
import numpy as np
from talib import stream
from collections import deque

high = deque([1.0, 2.0, 3.0, 4.0, 5.0])
low = deque([0.0, 1.0, 2.0, 3.0, 4.0])
close = deque([1.0, 2.0, 3.0, 4.0, 5.0])


high = deque([])
low = deque([])
close = deque([])

high = np.fromiter(high, dtype=np.float64)
low = np.fromiter(low, dtype=np.float64)
close = np.fromiter(close, dtype=np.float64)


for k in tqdm.tqdm(range(0,100000)):
    val = stream.ATR(high, low, close, timeperiod=5)
    print(val)