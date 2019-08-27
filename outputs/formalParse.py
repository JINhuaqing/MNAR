import numpy as np
from pickle import load
import matplotlib.pyplot as plt
from pathlib import Path

num = 300
root = Path('./')
imgdir = root/f"SimuImgs{num}"
if not imgdir.is_dir():
    imgdir.mkdir()

with open(f"Simulation_demo{num}.pkl", "rb") as f:
    data = load(f)

params, results, errss = data
resarr = np.array(results)
#print(resarr[:, [2, 5]])
print(np.mean(resarr, axis=0)[[2, 5]])

# for errs in errss:
#     Berrs, Terrs, betas, thetas, likelis = errs
#     plt.subplots_adjust(hspace=0.5)
# 
#     plt.subplot(321)
#     plt.title("beta errors plot")
#     plt.plot(Berrs, "r-")
#     plt.subplot(322)
#     plt.title("Theta errors plot")
#     plt.plot(Terrs, "r-")
# 
#     plt.subplot(323)
#     plt.title("norm of betahat")
#     plt.plot(betas, "g--")
#     plt.subplot(324)
#     plt.title("norm of thetahat")
#     plt.plot(thetas, "g--")
# 
#     plt.subplot(325)
#     plt.title("Negative likelihood plot")
#     plt.plot(likelis, "b-.")
#     plt.subplot(326)
#     plt.title("Negative likelihood plot")
#     plt.plot(likelis, "b-.")
# 
#     plt.savefig(imgdir/f"{errss.index(errs)+1}.jpg")
#     plt.close()
