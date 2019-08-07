import numpy as np
from pickle import load


with open("Bern_5_4_100_50_50.pkl", "rb") as f:
    data = load(f)

data = data[1:]
dataarr = np.array(data)
print("iteration number ", dataarr[:, 0])
merrb, merrT = dataarr.mean(axis=0)[1:3]
print(merrb, merrT)
