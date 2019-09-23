import sys
sys.path.append("../")
from utilities import *
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from pickle import load
import torch
from plot_utilities import *
import random
import matplotlib.pyplot as plt

# fix the random seed for several packages
torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
np.random.seed(0) #numpy
random.seed(0) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

torch.cuda.set_device(1)
cuda = torch.cuda.is_available()
# Set default data type
if cuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

root = Path("./")
r, s = 4, 4
p = 100
adj_error = 0
Cb, CT = 8, 2e-3
prob = 0.05
inp = 1.25


def sortf(x):
    return float(x.stem.split("_")[-1])


files = root.glob(f"Simulation_p{p}*.pkl")
files = list(files)
files.sort(key=sortf)

Berrs = []
Terrs = []

print(files)
for f in files:
    with open(f, "rb") as data:
        params, results, errss = load(data)
    resarr = np.array(results)
    Terrs.append(resarr[:, [5]])

plt.subplots_adjust(hspace=0.5)
plt.subplot(331)
plt.title("100")
plt.hist(Terrs[0], bins=50)
plt.subplot(332)
plt.title("120")
plt.hist(Terrs[1], bins=50)
plt.subplot(333)
plt.title("135")
plt.hist(Terrs[2], bins=50)
plt.subplot(334)
plt.title("150")
plt.hist(Terrs[3], bins=50)
plt.subplot(335)
plt.title("165")
plt.hist(Terrs[4], bins=50)
plt.subplot(336)
plt.title("180")
plt.hist(Terrs[5], bins=50)
plt.subplot(337)
plt.title("190")
plt.hist(Terrs[6], bins=50)
plt.subplot(338)
plt.title("200")
plt.hist(Terrs[7], bins=50)

plt.savefig("tmpplot.jpg")



    
        


