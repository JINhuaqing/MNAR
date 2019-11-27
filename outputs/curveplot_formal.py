import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from pickle import load
import torch
from plot_utilities import *
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
p = 200
prob = 0.05


def Errbub(a, c0, m, n, p):
    alpha0b = 1
    dh2, ds2 = 1, 1
    sigma1f = 32*c0**2*a**2*(dh2+ds2**2)
    itm1 = (4*alpha0b)**(-1/2)
    itm2 = (2*sigma1f**2*np.log(p)/m/n)**(1/4)
    return itm1*itm2

def ErrTub(a, m, n):
    alpha0t = 1
    dh2, ds2 = 1, 1 
    sigmadf = 32*a**2*(dh2+ds2**2)
    d = np.sqrt(n*m)
    itm1 = (4*alpha0t)**(-1/2)
    itm2 = (2*sigmadf**2*d*np.log(d)/m/n)**(1/4)
    return itm1*itm2


def sortf(x):
    return float(x.stem.split("_")[-1])


files = root.glob(f"./MNARlog/Simulation*p{p}*.pkl")
Marfiles = root.glob(f"./MARlog/MARSimulation*p{p}*.pkl")
files = list(files)
files.sort(key=sortf)
Marfiles = list(Marfiles)
Marfiles.sort(key=sortf)

Berrs = []
Terrs = []
mBerrs = []
mTerrs = []
MarBerrs = []
MarTerrs = []
xlist = []
Marxlist = []
errtubs = []
errbubs = []
aas = []
c0s = []
# load the  results of MNAR results
for f in files:
    with open(f, "rb") as data:
        ress = load(data)
        params, results, errss, EstParas = ress
    beta0 = torch.tensor(params["beta0"])
    bTheta0 = torch.tensor(params["bTheta0"])
    a = np.max([bTheta0.abs().max().item(), beta0.abs().max().item()])
    aas.append(a)
    c0 = p * prob
    c0s.append(c0)

c0 = np.min(c0s)
a = np.max(aas)

for f in files:
    with open(f, "rb") as data:
        ress = load(data)
        params, results, errss, EstParas = ress
    results = [xi for xi in results if xi[0] != -100]
    resarr = np.array(results)
    m, n, p = params["m"], params["n"], params["p"]
    xlist.append(n)
    alpha0b, alpha0t = 1, 1
    beta0 = torch.tensor(params["beta0"])
    bTheta0 = torch.tensor(params["bTheta0"])
    alpha0t, alpha0b = 1, 1
    errbub = Errbub(a, c0, m, n, p)
    errtub = ErrTub(a, m, n)
    errbubs.append(errbub)
    errtubs.append(errtub)
    Berr, Terr = np.mean(resarr, axis=0)[[2, 5]]
    mBerr, mTerr = np.max(resarr, axis=0)[[2, 5]]
    Berrs.append(Berr)
    Terrs.append(Terr)
    mBerrs.append(mBerr)
    mTerrs.append(mTerr)

for marf in Marfiles:
    with open(marf, "rb") as data:
        ress = load(data)
        params, results, errss, EstParas = ress
    m, n, p = params["m"], params["n"], params["p"]
    Marxlist.append(n)
    results = [xi for xi in results if xi[0] != -100]
    resarr = np.array(results)
    Berr, Terr = np.median(resarr, axis=0)[[2, 5]]
    MarBerrs.append(Berr)
    MarTerrs.append(Terr)

# plot MNARxBD 
plt.subplots_adjust(hspace=0.5)
plt.subplot(211)
plt.xlabel("m=n")
plt.ylabel(r"$\Vert\widehat{\mathrm{\beta}}-\mathrm{\beta}_0\Vert_2$")
plt.plot(xlist, mBerrs, "g-.h", label="Maximal Errors", linewidth=2)
plt.plot(xlist, np.array(errbubs), "r--", label="Error Upper Bound", linewidth=2)
plt.legend(loc=1)
plt.subplot(212)
plt.ylim([-0.5, 15])
plt.xlabel("m=n")
plt.ylabel(r"$\Vert\widehat{\mathrm{\Theta}}-\mathrm{\Theta}_0\Vert_F$")
plt.plot(xlist, mTerrs, "g-.h", label="Maximal Error", linewidth=2)
plt.plot(xlist, np.array(errtubs), "r--", label="Error Upper Bound", linewidth=2)
plt.legend(loc=1)
plt.savefig(f"{p}_MNARxBD_Logistic.jpg")

plt.close()

# plot MNARxMAR
plt.subplots_adjust(hspace=0.5)
plt.subplot(211)
plt.xlabel("m=n")
plt.ylabel(r"$\Vert\widehat{\mathrm{\beta}}-\mathrm{\beta}_0\Vert_2$")
plt.plot(xlist, Berrs, "g-.h", label="MNAR", linewidth=2)
plt.plot(Marxlist, MarBerrs, "b-.x", label="MAR", linewidth=2)
plt.legend(loc=1)
plt.subplot(212)
plt.xlabel("m=n")
plt.ylabel(r"$\Vert\widehat{\mathrm{\Theta}}-\mathrm{\Theta}_0\Vert_F$")
plt.plot(xlist, Terrs, "g-.h", label="MNAR", linewidth=2)
plt.plot(Marxlist, MarTerrs, "b-.x", label="MAR", linewidth=2)
plt.legend(loc=1)
#plt.savefig(f"{p}_MNARxMAR_Linear.jpg")
plt.savefig(f"{p}_MNARxMAR_Logistic.jpg")



    
        


