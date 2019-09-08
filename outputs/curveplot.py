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
r, s = 5, 5
p = 100
adj_error = 0
Cb, CT = 8, 2e-3
prob = 0.05
inp = 1.25

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
    alpha0b = 1
    item1 = np.sqrt(s)*np.sqrt(np.log(p)/m/n)
    item2 = (np.log(p)/m/n)**(1/4) / np.sqrt(alpha0b)
    item3 = np.sqrt(s) * Lambfn(Cb, n, m)/alpha0b
    itm = np.max([item1, item2, item3])
    print("B", item1, item2, item3, itm)
    return 1/itm

def prefixTheta(n, m, p, r, alpha0t):
    alpha0t = 1
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
    return float(x.stem[15:])


files = root.glob("Simulation*.pkl")
files = list(files)
files.sort(key=sortf)

Berrs = []
Terrs = []
xlist = []
errtubs = []
errbubs = []
aas = []
c0s = []
for f in files[:-1]:
    with open(f, "rb") as data:
        params, results, errss = load(data)
    beta0 = torch.tensor(params["beta0"])
    bTheta0 = torch.tensor(params["bTheta0"])
    a = np.max([bTheta0.abs().max().item(), beta0.abs().max().item()])
    aas.append(a)
    c0 = 1
    c0s.append(c0)

c0 = np.min(c0s)
a = np.min(aas)

for f in files[:-1]:
    with open(f, "rb") as data:
        params, results, errss = load(data)
    resarr = np.array(results)
    m, n, p = params["m"], params["n"], params["p"]
    alpha0b, alpha0t = 1, 1
    #beta0 = torch.tensor(params["beta0"])
    #bTheta0 = torch.tensor(params["bTheta0"])
    #X = genXdis(n, m, p, type="Bern", prob=prob) 
    #Y = genYlogit(X, bTheta0, beta0)
    #errbub, alpha0b = Errbub(Y, X, bTheta0, beta0, inp)
    #errtub, alpha0t = ErrTub(Y, X, bTheta0, beta0, inp)
    errbub = Errbub(a, c0, m, n, p)
    errtub = ErrTub(a, m, n)
    errbubs.append(errbub)
    errtubs.append(errtub)
    preB = prefixBeta(n, m, p, s, alpha0b)
    preT = prefixTheta(n, m, p, r, alpha0t)
    Berr, Terr = np.median(resarr, axis=0)[[2, 5]]
    xlist.append(n*m)
    if adj_error:
        Berrs.append(preB*Berr)
        Terrs.append(preT*Terr)
    else:
        Berrs.append(Berr)
        Terrs.append(Terr)

plt.subplots_adjust(hspace=0.5)
plt.subplot(211)
plt.xlabel("mxn")
#plt.ylim([0.05, 0.08])
if adj_error:
    plt.ylabel("adjust error")
else:
    plt.ylabel("error")
    plt.plot(xlist, np.array(errbubs)/8, "r--")
plt.title("beta curve")
plt.plot(xlist, Berrs, "g-.h")
plt.subplot(212)
plt.xlabel("mxn")
if adj_error:
    plt.ylabel("adjust error")
else:
    plt.ylabel("error")
    plt.plot(xlist, np.array(errtubs)*1.1, "r--")
plt.title("theta curve")
plt.plot(xlist, Terrs, "g-.h")
if adj_error:
    plt.savefig("curveplt.jpg")
else:
    plt.savefig("curveplt_errvsmn.jpg")



    
        


