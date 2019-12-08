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

typ = "log"
p = 200

if typ.startswith("log"):
    def Cbsf(m): #logistic
        if m < 200:
            return 20
        elif m <= 225:
            return 30
        else:
            return 40
else:
    def Cbsf(m): # Linear
        if m <= 200:
            return 800
        else:
            return 800

    
# Fixed parameters 
prob = 0.05
r, s = 5, 5
if typ.startswith("log"):
    CT = 2e-3 # logistic 
else:
    CT = 2e-2 # linear


# just constant before the penalty item of beta
def Lambfn(C, n, m):
    rawv = np.sqrt(np.log(m+n))/m/n
    return C*rawv


# To compute the Lambda_bTheta
# just constant before the penalty item of bTheta
def LamTfn(C, n, m, p):
    d = np.sqrt(m*n)
    rawvs = [np.sqrt(np.log(d)/d), (np.log(p))**(1/4)/np.sqrt(d)]
    rawv = np.max(rawvs)
    return C*rawv

def prefixBeta(n, m, p, s, alpha0b):
    item1 = np.sqrt(s)*np.sqrt(np.log(p)/m/n)
    item2 = (np.log(p)/m/n)**(1/4) / np.sqrt(alpha0b)
    item3 = np.sqrt(s) * Lambfn(Cbsf(m), n, m)/alpha0b
    itm = np.max([item1, item2, item3])
    return 1/itm

def prefixTheta(n, m, p, r, alpha0t):
    d = np.sqrt(m*n)
    item1 = np.sqrt(r) *  np.sqrt(d*np.log(d)/m/n)
    item2 = (d*np.log(d)/m/n)**(1/4)/np.sqrt(alpha0t)
    item3 = np.sqrt(r) * LamTfn(CT, n, m, p)/alpha0t
    itm = np.max([item1, item2, item3])
    return 1/itm



def sortf(x):
    return float(x.stem.split("_")[-1])


if typ.startswith("log"):
    files = root.glob(f"./adjLog/Simulation*p{p}*.pkl")
else:
    files = root.glob(f"./adjLinear/Simulation*p{p}*.pkl")
files = list(files)
files.sort(key=sortf)
files = files

AjBerrs = []
AjTerrs = []
xlist = []
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
    beta0 = torch.tensor(params["beta0"])
    bTheta0 = torch.tensor(params["bTheta0"])
    alpha0t, alpha0b = 1, 1
    Berr, Terr = np.mean(resarr, axis=0)[[2, 5]]
    preB = prefixBeta(n, m, p, s, alpha0b)
    preT = prefixTheta(n, m, p, r, alpha0t)
    AjBerrs.append(Berr*preB)
    AjTerrs.append(Terr*preT)


font_y = {"size": 15, "va":"baseline"}
font_x = {"size": 15}
# plot AjMNAR for logistic and Linear
plt.xlabel("m=n", font_x)
plt.ylabel(r"$\Vert\Delta_{\mathrm{\beta}, adj}\Vert_2$", font_y)
plt.plot(xlist, AjBerrs, "g-.h", label="Adjusted Errors", linewidth=4)
plt.legend(loc=1)
if typ.startswith("log"):
    plt.savefig(f"{p}_AjMNAR_beta_Logistic.jpg")
else:
    plt.savefig(f"{p}_AjMNAR_beta_Linear.jpg")
plt.close()

plt.xlabel("m=n", font_x)
#plt.ylim([1.4, 2.1]) #for p=200 linear
plt.ylabel(r"$\Vert\Delta_{\mathrm{\Theta}, adj}\Vert_F$", font_y)
plt.plot(xlist, AjTerrs, "g-.h", label="Adjusted Error", linewidth=4)
plt.legend(loc=1)
if typ.startswith("log"):
    plt.savefig(f"{p}_AjMNAR_Theta_Logistic.jpg")
else:
    plt.savefig(f"{p}_AjMNAR_Theta_Linear.jpg")

plt.close()

