import numpy as np
from pickle import load
import matplotlib.pyplot as plt
from pathlib import Path

Cb = 3e-3
m = 200
etab = 1e-0
root = Path('./')
rootdir = root/"img"
if not rootdir.is_dir():
    rootdir.mkdir()

with open(f"./Bern_beta_{m}_{Cb:.0E}_{etab:.0E}.pkl", "rb") as f:
    data = load(f)
_, _, errs, likelis, betahats = data

plt.subplots_adjust(hspace=0.5)
plt.subplot(311)
plt.title("Theta errors plot")
plt.plot(errs, "r-")
plt.subplot(312)
plt.title("Negative likelihood plot")
plt.plot(likelis, "b-.")
plt.subplot(313)
plt.title("norm of betahat")
plt.plot(betahats, "g--")
plt.savefig(rootdir/f"Bern_beta_{m}_{Cb:.0E}_{etab:.0E}.jpg")
