import numpy as np
from pickle import load


with open("Bern_5_5_100_200_200.pkl", "rb") as f:
    data = load(f)

data = data[1:]
dataarr = np.array(data)
print("iteration number ", dataarr[:, 0])
merrb, merrT = dataarr.mean(axis=0)[1:3]
print(merrb, merrT)