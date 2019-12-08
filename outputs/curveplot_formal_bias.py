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
typ = "log"
p = 200
# only needed for logistic setting
def sortf(x):
    return float(x.stem.split("_")[-1])

def MeanParas(EstParas):
    Betahats = [i[0] for i in EstParas]
    Thetahats = [i[1] for i in EstParas]
    Betahats = np.array(Betahats)
    Thetahats = np.array(Thetahats)
    return [Betahats.mean(axis=0), Thetahats.mean(axis=0)]

if typ.startswith("log"):
    files = root.glob(f"./MNARlog/Simulation*p{p}*.pkl")
    Marfiles = root.glob(f"./MARlog/MARSimulation*p{p}*.pkl")
else:
    files = root.glob(f"./MNARlinear/Simulation*p{p}*.pkl")
    Marfiles = root.glob(f"./MARlinear/MARSimulation*p{p}*.pkl")
files = list(files)
files.sort(key=sortf)
files = np.array(files)
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
    beta0 = torch.tensor(params["beta0"])
    bTheta0 = torch.tensor(params["bTheta0"])
    Bmean, Tmean = MeanParas(EstParas)
    Berr = np.sqrt(((beta0.cpu().numpy()-Bmean)**2).sum())
    Terr = np.sqrt(((bTheta0.cpu().numpy()-Tmean)**2).sum())
    m, n, p = params["m"], params["n"], params["p"]
    xlist.append(n)
    Berrs.append(Berr)
    Terrs.append(Terr)

for marf in Marfiles:
    with open(marf, "rb") as data:
        ress = load(data)
        params, results, errss, EstParas = ress
    m, n, p = params["m"], params["n"], params["p"]
    Marxlist.append(n)
    beta0 = params["beta0"]
    bTheta0 = params["bTheta0"]
    Bmean, Tmean = MeanParas(EstParas)
    Berr = np.sqrt(((beta0-Bmean)**2).sum())
    Terr = np.sqrt(((bTheta0-Tmean)**2).sum())
    MarBerrs.append(Berr)
    MarTerrs.append(Terr)

font_y = {"size": 15, "va":"baseline"}
font_x = {"size": 15}
# plot MNARxMAR
plt.xlabel("m=n", font_x)
#plt.ylim([1.4, 6])
plt.ylabel(r"$\Vert E(\widehat{\mathrm{\beta}})-\mathrm{\beta}_0\Vert_2$", font_y)
plt.plot(xlist, Berrs, "g-.h", label="MNAR", linewidth=4)
plt.plot(Marxlist, MarBerrs, "b-.x", label="MAR", linewidth=4)
plt.legend(loc=1)
if typ.startswith("log"):
    plt.savefig(f"{p}_MNARxMAR_bias_beta_Logistic.jpg")
else:
    plt.savefig(f"{p}_MNARxMAR_bias_beta_Linear.jpg")
plt.close()

plt.xlabel("m=n", font_x)
plt.ylabel(r"$\Vert E(\widehat{\mathrm{\Theta}})-\mathrm{\Theta}_0\Vert_F$", font_y)
plt.plot(xlist, Terrs, "g-.h", label="MNAR", linewidth=4)
plt.plot(Marxlist, MarTerrs, "b-.x", label="MAR", linewidth=4)
plt.legend(loc=1)
if typ.startswith("log"):
    plt.savefig(f"{p}_MNARxMAR_bias_Theta_Logistic.jpg")
else:
    plt.savefig(f"{p}_MNARxMAR_bias_Theta_Linear.jpg")
plt.close()
