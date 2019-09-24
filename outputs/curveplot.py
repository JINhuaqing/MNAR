import sys
sys.path.append("../")
from utilities import *
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from pickle import load
import torch
from plot_utilities import *
import argparse
import random


parser = argparse.ArgumentParser(description = "This script is to plot the error curve for MNAR project")
parser.add_argument('-p', type=int, default=100, help = "Parameter p")
parser.add_argument('-t', "--type", type=str, choices=["MNARxBd", "MNARxMAR", "AjMNAR", "MAR", "MNAR"], default="MAR", help = "Specify the types of the plot.")
parser.add_argument('-et', "--errortype", type=str, choices=["Bias", "MSE"], default="Bias", help = "Specify the types of error to be computed.")
args = parser.parse_args()

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

# Arguments
p = args.p
typ = args.type
etyp = args.errortype

root = Path("./")

# Fixed parameters 
r, s = 5, 5
Cb, CT = 8, 2e-3
prob = 0.05
inp = 1.25

def MeanParas(EstParas):
    Betahats = [i[0] for i in EstParas]
    Thetahats = [i[1] for i in EstParas]
    Betahats = np.array(Betahats)
    Thetahats = np.array(Thetahats)
    return [Betahats.mean(axis=0), Thetahats.mean(axis=0)]

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
    item3 = np.sqrt(s) * Lambfn(Cb, n, m)/alpha0b
    itm = np.max([item1, item2, item3])
    print("B", item1, item2, item3, itm)
    return 1/itm

def prefixTheta(n, m, p, r, alpha0t):
    d = np.sqrt(m*n)
    item1 = np.sqrt(r) *  np.sqrt(d*np.log(d)/m/n)
    item2 = (d*np.log(d)/m/n)**(1/4)/np.sqrt(alpha0t)
    item3 = np.sqrt(r) * LamTfn(CT, n, m, p)/alpha0t
    itm = np.max([item1, item2, item3])
    print("T", item1, item2, item3, itm)
    return 1/itm


#def Errbub(Y, X, bTheta, beta, inp):
#    alpha0b = Alpha_0b(Y, X, bTheta, beta, inp)
#    alpha0b = 1
#    sigma1f = Sigma_1F(Y, X, bTheta, beta)
#    n, m, p = X.shape
#    itm1 = (4*alpha0b)**(-1/2)
#    itm2 = (2*sigma1f**2*np.log(p)/m/n)**(1/4)
#    return itm1*itm2, alpha0b

def Errbub(a, c0, m, n, p):
    alpha0b = 1
    dh2, ds2 = 1, 1
    sigma1f = 32*c0**2*a**2*(dh2+ds2**2)
    itm1 = (4*alpha0b)**(-1/2)
    itm2 = (2*sigma1f**2*np.log(p)/m/n)**(1/4)
    return itm1*itm2


#def ErrTub(Y, X, bTheta, beta, inp):
#    alpha0t = Alpha_0T(Y, X, bTheta, beta, inp)
#    alpha0t = 1
#    sigmadf = Sigma_dF(Y, X, bTheta, beta)
#    n, m, p = X.shape
#    d = np.sqrt(n*m)
#    itm1 = (4*alpha0t)**(-1/2)
#    itm2 = (2*sigmadf**2*d*np.log(d)/m/n)**(1/4)
#    return itm1*itm2, alpha0t

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


files = root.glob(f"Simulation_p{p}*.pkl")
Marfiles = root.glob(f"Mar_Simulation_p{p}*.pkl")
files = list(files)
files.sort(key=sortf)
Marfiles = list(Marfiles)
Marfiles.sort(key=sortf)

Berrs = []
Terrs = []
MarBerrs = []
MarTerrs = []
xlist = []
Marxlist = []
errtubs = []
errbubs = []
aas = []
c0s = []
if typ in ["MNARxBd", "MNARxMAR", "AjMNAR", "MNAR"]:
    for f in files:
        with open(f, "rb") as data:
            ress = load(data)
            if len(ress) == 3:
                params, results, errss = ress
            else:
                params, results, errss, EstParas = ress
        beta0 = torch.tensor(params["beta0"])
        bTheta0 = torch.tensor(params["bTheta0"])
        a = np.max([bTheta0.abs().max().item(), beta0.abs().max().item()])
        aas.append(a)
        c0 = 1
        c0s.append(c0)
    
    c0 = np.min(c0s)
    a = np.max(aas)

    for f in files:
        with open(f, "rb") as data:
            ress = load(data)
            if len(ress) == 3:
                params, results, errss = ress
            else:
                params, results, errss, EstParas = ress
        resarr = np.array(results)
        m, n, p = params["m"], params["n"], params["p"]
        xlist.append(n*m)
        alpha0b, alpha0t = 1, 1
        beta0 = torch.tensor(params["beta0"])
        bTheta0 = torch.tensor(params["bTheta0"])
        X = genXdis(n, m, p, type="Bern", prob=prob) 
        Y = genYlogit(X, bTheta0, beta0)
        alpha0t = Alpha_0T(Y, X, bTheta0, beta0, inp)
        print(alpha0t, "T")
        #alpha0t = 1
        alpha0b = Alpha_0b(Y, X, bTheta0, beta0, inp)
        #alpha0b = 1
        print(alpha0b, "b")
        errbub = Errbub(a, c0, m, n, p)
        errtub = ErrTub(a, m, n)
        errbubs.append(errbub)
        errtubs.append(errtub)
        if etyp == "Bias":
            assert len(ress) == 4
            Bmean, Tmean = MeanParas(EstParas)
            Berr = np.sqrt(((beta0.cpu().numpy()-Bmean)**2).sum())
            Terr = np.sqrt(((bTheta0.cpu().numpy()-Tmean)**2).sum())
        else:
            Berr, Terr = np.median(resarr, axis=0)[[2, 5]]
        if typ=="AjMNAR":
            preB = prefixBeta(n, m, p, s, alpha0b)
            preT = prefixTheta(n, m, p, r, alpha0t)
            Berrs.append(preB*Berr)
            Terrs.append(preT*Terr)
        else:
            Berrs.append(Berr)
            Terrs.append(Terr)

if typ in ["MNARxMAR", "MAR"]:
    for marf in Marfiles:
        with open(marf, "rb") as data:
            ress = load(data)
            if len(ress) == 3:
                params, results, errss = ress
            else:
                params, results, errss, EstParas = ress
        m, n, p = params["m"], params["n"], params["p"]
        Marxlist.append(n*m)
        resarr = np.array(results)
        if etyp == "Bias":
            assert len(ress) == 4
            beta0 = params["beta0"]
            bTheta0 = params["bTheta0"]
            Bmean, Tmean = MeanParas(EstParas)
            Berr = np.sqrt(((beta0-Bmean)**2).sum())
            Terr = np.sqrt(((bTheta0-Tmean)**2).sum())
        else:
            Berr, Terr = np.median(resarr, axis=0)[[2, 5]]
        MarBerrs.append(Berr)
        MarTerrs.append(Terr)

["MNARxBd", "MNARxMAR", "AjMNAR", "MAR", "MNAR"]
plt.subplots_adjust(hspace=0.5)

Tlabel_MNAR = r"MNAR $\Theta$ errors"
Blabel_MNAR = r"MNAR $\beta$ errors"
Tlabel_MAR = r"MAR $\Theta$ errors"
Blabel_MAR = r"MAR $\beta$ errors"
Tlabel_MNAR_up = r"MNAR $\Theta$ errors UpperBound"
Blabel_MNAR_up = r"MNAR $\beta$ errors UpperBound"
Tlabel_MNAR_aj = r"Adjuted MNAR $\Theta$ errors"
Blabel_MNAR_aj = r"Adjusted MNAR $\beta$ errors"

ylabel_name = f"error"
plt.subplot(211)
plt.xlabel("mxn")
#plt.ylim([0.05, 0.08])
if typ == "AjMNAR":
    plt.ylabel("Adjust Error")
    plt.plot(xlist, Berrs, "g-.h", label=Blabel_MNAR_aj)
elif typ == "MNAR":
    plt.ylabel(ylabel_name)
    plt.plot(xlist, Berrs, "g-.h", label=Blabel_MNAR)
elif typ == "MNARxBd":
    plt.ylabel(ylabel_name)
    plt.plot(xlist, Berrs, "g-.h", label=Blabel_MNAR)
    plt.plot(xlist, np.array(errbubs), "r--", label=Blabel_MNAR_up)
elif typ == "MNARxMAR":
    plt.ylabel(ylabel_name)
    plt.plot(xlist, Berrs, "g-.h", label=Blabel_MNAR)
    plt.plot(Marxlist, MarBerrs, "b-.x", label=Blabel_MAR)
else:
    plt.ylabel(ylabel_name)
    plt.plot(Marxlist, MarBerrs, "b-.x", label=Blabel_MAR)

plt.title(r"$\beta$ curve")
plt.legend()


plt.subplot(212)
plt.xlabel("mxn")
if typ == "AjMNAR":
    plt.ylabel("Adjusted Error")
    plt.plot(xlist, Terrs, "g-.h", label=Tlabel_MNAR_aj)
elif typ == "MNAR":
    plt.ylabel(ylabel_name)
    plt.plot(xlist, Terrs, "g-.h", label=Tlabel_MNAR)
elif typ == "MNARxBd":
    plt.ylabel(ylabel_name)
    plt.plot(xlist, Terrs, "g-.h", label=Tlabel_MNAR)
    plt.plot(xlist, np.array(errtubs), "r--", label=Tlabel_MNAR_up)
elif typ == "MNARxMAR":
    plt.ylabel(ylabel_name)
    plt.plot(xlist, Terrs, "g-.h", label=Tlabel_MNAR)
    plt.plot(Marxlist, MarTerrs, "b-.x", label=Tlabel_MAR)
else:
    plt.ylabel(ylabel_name)
    plt.plot(Marxlist, MarTerrs, "b-.x", label=Tlabel_MAR)

plt.title(r"$\theta$ curve")
plt.legend()
plt.savefig(f"{typ}_{etyp}_{p}_curveplot.jpg")



    
        


