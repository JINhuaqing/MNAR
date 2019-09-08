import numpy as np
from pickle import load
import matplotlib.pyplot as plt

eta = 1e-2
CT = 2e-3
m = 300
with open(f"./Bern_theta_{m}_{CT:.0E}_{eta:.0E}.pkl", "rb") as f:
#with open(f"./Bern_theta_100_test.pkl", "rb") as f:
    data = load(f)
_, errs, likelis, thetahats = data

plt.subplots_adjust(hspace=0.5)
plt.subplot(311)
plt.title("Theta errors plot")
plt.plot(errs, "r-")
plt.subplot(312)
plt.title("Negative likelihood plot")
plt.plot(likelis, "b-.")
plt.subplot(313)
plt.title("norm of thetahat")
plt.plot(thetahats, "g--")
#plt.savefig(f"./img/Bern_theta_100_test.jpg")
plt.savefig(f"./img/Bern_theta_{m}_{CT:.0E}_{eta:.0E}.jpg")
