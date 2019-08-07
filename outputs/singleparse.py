import numpy as np
from pickle import load


with open("Bern_5_4_100_50_50.pkl", "rb") as f:
    data = load(f)

Cb, CT, ST = data["Cb"], data["CT"], data["ST"]
Berrs, Terrs = data["Berrs"], data["Terrs"]
print(Terrs)
