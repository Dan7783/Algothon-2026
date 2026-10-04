import numpy as np

PAIR_I = np.array([7, 46, 1, 13, 27, 33, 36, 49, 50, 12, 50, 10, 31, 3, 49, 17, 45, 32, 15, 2], dtype=int)
PAIR_J = np.array([40, 10, 20, 45, 8, 42, 41, 28, 28, 40, 37, 25, 40, 34, 24, 18, 46, 28, 25, 42], dtype=int)
PAIRS = tuple(zip(PAIR_I.tolist(), PAIR_J.tolist()))
