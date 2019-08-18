import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from pickle import load

root = Path("./")
r, s = 5, 5
p = 100


def prefixBeta(n, m, p, s):
    item1 = np.sqrt(s)*np.sqrt(np.log(p)/m/n)
    item2 = (np.log(p)/m/n)**(1/4)
    itm = np.max([item1, item2])
    return 1/itm


def prefixTheta(n, m, p, r):
    d = np.sqrt(m*n)
    item1 = np.sqrt(r) *  np.sqrt(d*np.log(d)/m/n)
    item2 = (d*np.log(d)/m/n)**(1/4)
    itm = np.max([item1, item2])
    return 1/itm


def sortf(x):
    return float(x.stem[15:])


files = root.glob("Simulation*.pkl")
files = list(files)
files.sort(key=sortf)

Berrs = []
Terrs = []
xlist = []
for f in files:
    with open(f, "rb") as data:
        params, results, errss = load(data)
    resarr = np.array(results)
    m, n = params["m"], params["n"]
    preB = prefixBeta(n, m, p, s)
    preT = prefixTheta(n, m, p, r)
    Berr, Terr = np.median(resarr, axis=0)[[2, 5]]
    xlist.append(n*m)
    Berrs.append(preB*Berr)
    Terrs.append(preT*Terr)

plt.subplots_adjust(hspace=0.5)
plt.subplot(211)
plt.title("beta curve")
plt.plot(xlist, Berrs, "r--")
plt.subplot(212)
plt.title("theta curve")
plt.plot(xlist, Terrs, "g-.")
plt.savefig("curveplt.jpg")

    
        


