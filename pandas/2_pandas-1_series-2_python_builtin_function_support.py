# Pandas Series object support majority of the python builtin functions like
# type, len, list, max, min, dict, sorted etc

import pandas as pd
from random import sample, seed
seed(0)

nums = pd.Series(sample(population=[i for i in range(100)], k=5) )

print(nums)
# 0    49
# 1    97
# 2    53
# 3     5
# 4    33
# dtype: int64

print(type(nums)) # <class 'pandas.core.series.Series'>

print(len(nums)) # 5

print(list(nums)) # [49, 97, 53, 5, 33]

print(max(nums), min(nums)) # 97 5

print(dict(nums)) # {0: np.int64(49), 1: np.int64(97), 2: np.int64(53), 3: np.int64(5), 4: np.int64(33)}

print(sorted(nums)) # [5, 33, 49, 53, 97]