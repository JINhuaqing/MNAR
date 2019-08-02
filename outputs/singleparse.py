import numpy as np
from pickle import load


with open("Bern_103.5_8.5840_4660.pkl", "rb") as f:
    data = load(f)

Cb, CT, ST = data["Cb"], data["CT"], data["ST"]
Berrs, Terrs = data["Berrs"], data["Terrs"]
print(Terrs)
