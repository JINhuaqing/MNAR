import numpy as np
from pickle import load
import matplotlib.pyplot as plt
from pathlib import Path

Cb = 1e-3
m = 200
etab = 4e-2
root = Path('./')
rootdir = root/"imgmar"
if not rootdir.is_dir():
    rootdir.mkdir()

#with open(f"./Bern_beta_{m}_{Cb:.0E}_{etab:.0E}.pkl", "rb") as f:
with open(f"./Mar_Bern_beta_{m}_{Cb:.0E}_{etab:.0E}.pkl", "rb") as f:
    data = load(f)
_, _, errs, likelis, betahats = data

plt.subplots_adjust(hspace=0.5)
plt.subplot(311)
plt.title("beta errors plot")
plt.plot(errs, "r-")
plt.subplot(312)
plt.title("Negative likelihood plot")
plt.plot(likelis, "b-.")
plt.subplot(313)
plt.title("norm of betahat")
plt.plot(betahats, "g--")
#plt.savefig(rootdir/f"Bern_beta_{m}_{Cb:.0E}_{etab:.0E}.jpg")
plt.savefig(rootdir/f"Mar_Bern_beta_{m}_{Cb:.0E}_{etab:.0E}.jpg")
