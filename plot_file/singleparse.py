import numpy as np
from pickle import load
import matplotlib.pyplot as plt

#with open("./outputs/Bern_103.5_0.2146_4660.pkl", "rb") as f:
with open("./outputs/Bern_219.6_0.0450_318.pkl", "rb") as f:
    data = load(f)

Cb, CT, ST = data["Cb"], data["CT"], data["ST"]
Berrs, Terrs = data["Berrs"], data["Terrs"]
Berrs, Terrs = np.array(Berrs), np.array(Terrs)

plt.subplot(211)
plt.title("l2 norm of Errors of Beta")
plt.plot(Berrs, "r-")
plt.subplot(212)
plt.title("Fro norm of Errors of Theta")
plt.plot(Terrs, "b--")
plt.show()
