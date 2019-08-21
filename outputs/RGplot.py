import numpy as np
from pickle import load
import matplotlib.pyplot as plt
from pathlib import Path

idx = 200
root = Path('./')
imgdir = root/f"imgs{idx}"
if not imgdir.is_dir():
    imgdir.mkdir()



with open(f"./RandGrid_Bern_new.pkl", "rb") as f:
    data = load(f)
ress, errss = data

for errs in errss:
    Berrs, Terrs, betas, thetas, likelis = errs
    #if Berrs[0] > Berrs[-1]:
    #    print(errss.index(errs), np.round(Berrs[0]- Berrs[-1], 4), np.round((Berrs[0]- Berrs[-1])/Berrs[0], 4), np.round( Berrs[0]- np.min(Berrs[-1]), 4))
    #print(Terrs[-1]- Terrs[0])
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(321)
    plt.title("beta errors plot")
    plt.plot(Berrs, "r-")
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

    plt.savefig(imgdir/f"{errss.index(errs)+1}.jpg")
    plt.close()
