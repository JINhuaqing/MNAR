import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from pickle import load
import torch
import argparse
import random


# fix the random seed for several packages
torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
np.random.seed(0) #numpy
random.seed(0) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

torch.cuda.set_device(1)
cuda = torch.cuda.is_available()
# Set default data type
torch.set_default_tensor_type(torch.cuda.DoubleTensor)

# Arguments

root = Path("./")


# Fixed parameters 
typ = "linear"
p = 200
# only needed for logistic setting
def sortf(x):
    return float(x.stem.split("_")[-1])


if typ.startswith("log"):
    files = root.glob(f"./MNARlog/Simulation*p{p}*.pkl")
    Marfiles = root.glob(f"./MARlog/MARSimulation*p{p}*.pkl")
else:
    files = root.glob(f"./MNARlinear/Simulation*p{p}*.pkl")
    Marfiles = root.glob(f"./MARlinear/MARSimulation*p{p}*.pkl")
files = list(files)
files.sort(key=sortf)
files = np.array(files)#[[0, 1, 2, 4, 6]]
Marfiles = list(Marfiles)
Marfiles.sort(key=sortf)

Berrs = []
Terrs = []
MarBerrs = []
MarTerrs = []
xlist = []
Marxlist = []

for f in files:
    with open(f, "rb") as data:
        ress = load(data)
        params, results, errss, EstParas = ress
    results = [xi for xi in results if xi[0] != -100]
    resarr = np.array(results)
    m, n, p = params["m"], params["n"], params["p"]
    xlist.append(n)
    beta0 = torch.tensor(params["beta0"])
    bTheta0 = torch.tensor(params["bTheta0"])
    Berr, Terr = np.mean(resarr, axis=0)[[2, 5]]
    Berrs.append(Berr)
    Terrs.append(Terr)

for marf in Marfiles:
    with open(marf, "rb") as data:
        ress = load(data)
        params, results, errss, EstParas = ress
    m, n, p = params["m"], params["n"], params["p"]
    Marxlist.append(n)
    results = [xi for xi in results if xi[0] != -100]
    resarr = np.array(results)
    #Berr, Terr = np.median(resarr, axis=0)[[2, 5]]
    Berr, Terr = np.mean(resarr, axis=0)[[2, 5]]
    MarBerrs.append(Berr)
    MarTerrs.append(Terr)


# plot MNARxMAR
font_y = {"size": 15, "va":"baseline"}
font_x = {"size": 15}
plt.xlabel("m=n", font_x)
#plt.ylim([1.4, 6])
plt.ylabel(r"$\Vert\widehat{\mathrm{\beta}}-\mathrm{\beta}_0\Vert_2$", font_y)
plt.plot(xlist, Berrs, "g-.h", label="MNAR", linewidth=4)
plt.plot(Marxlist, MarBerrs, "b-.x", label="MAR", linewidth=4)
plt.legend(loc=1)
if typ.startswith("log"):
    plt.savefig(f"{p}_MNARxMAR_beta_Logistic.jpg")
else:
    plt.savefig(f"{p}_MNARxMAR_beta_Linear.jpg")
plt.close()

plt.xlabel("m=n", font_x)
plt.ylabel(r"$\Vert\widehat{\mathrm{\Theta}}-\mathrm{\Theta}_0\Vert_F$", font_y)
plt.plot(xlist, Terrs, "g-.h", label="MNAR", linewidth=4)
plt.plot(Marxlist, MarTerrs, "b-.x", label="MAR", linewidth=4)
plt.legend(loc=1)
if typ.startswith("log"):
    plt.savefig(f"{p}_MNARxMAR_Theta_Logistic.jpg")
else:
    plt.savefig(f"{p}_MNARxMAR_Theta_Linear.jpg")
plt.close()
