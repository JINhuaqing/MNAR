import numpy as np
from pickle import load
import matplotlib.pyplot as plt
from pathlib import Path

Cb = 10
etab = 5e-1*2*4
root = Path('./')
rootdir = root/"img"
if not rootdir.is_dir():
    rootdir.mkdir()

with open(f"./Bern_beta_100_{Cb}_{etab:.0E}.pkl", "rb") as f:
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
plt.savefig(rootdir/f"Bern_beta_100_{Cb}_{etab:.0E}.jpg")
