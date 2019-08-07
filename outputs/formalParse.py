import numpy as np
from pickle import load


with open("Bern_5_4_100_50_50.pkl", "rb") as f:
    data = load(f)

data = data[1:]
dataarr = np.array(data)
kpidx = dataarr[:, 0] != 1000
kpdata = dataarr[kpidx]
print("iteration number ", dataarr[:, 0])
merrb, merrT = kpdata.mean(axis=0)[1:3]
print(merrb, merrT)
