import numpy as np
from pickle import load
import matplotlib.pyplot as plt
from pathlib import Path

m = 200 
Cb = 20  
root = Path('./')
imgdir = root/f"img"
if not imgdir.is_dir():
    imgdir.mkdir()

with open(f"Man_Bern_new_{m:.0f}_{Cb:.0f}.pkl", "rb") as f:
    data = load(f)

results, errs = data

Berrs, Terrs, betas, thetas, likelis = errs
plt.subplots_adjust(hspace=0.5)

plt.subplot(321)
plt.title("beta errors plot")
plt.plot(Berrs, "r-")
#plt.ylim([1.1, 1.3])
plt.subplot(322)
plt.title("Theta errors plot")
plt.plot(Terrs, "r-")

plt.subplot(323)
plt.title("norm of betahat")
plt.plot(betas, "g--")
plt.subplot(324)
plt.title("norm of thetahat")
plt.plot(thetas, "g--")

plt.subplot(325)
plt.title("Negative likelihood plot")
plt.plot(likelis, "b-.")
plt.subplot(326)
plt.title("Negative likelihood plot")
plt.plot(likelis, "b-.")

plt.savefig(imgdir/f"Man_Bern_new_{m:.0f}_{Cb:.0f}.jpg")
plt.close()
