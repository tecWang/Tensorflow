import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()

data = np.array(digits.data)
print(data)
# [[ 0.  0.  5. ...  0.  0.  0.]
#  [ 0.  0.  0. ... 10.  0.  0.]
#  [ 0.  0.  0. ... 16.  9.  0.]
#  ...
#  [ 0.  0.  1. ...  6.  0.  0.]
#  [ 0.  0.  2. ... 12.  0.  0.]
#  [ 0.  0. 10. ... 12.  1.  0.]]
print(data.shape)   # (1797, 64)

target = np.array(digits.target)
print(target) # [0 1 2 ... 8 9 8]
# print(target.shape)   # (1797,)
print(len(target))  # 1797

print(digits.DESCR)
