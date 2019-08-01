import os
import sys
sys.path.append("/home/user2/Documents/JIN/MNAR")
os.chdir("/home/user2/Documents/JIN/MNAR")
from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit
import time

torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
np.random.seed(0) #numpy
random.seed(0) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

cuda = torch.cuda.is_available()
cuda = False
if cuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
sigma = 0.5


def fn(y, m, bsXs=None, sigma=sigma):
    # y     : n x m
    # m     : n x m 
    # bsXs  : N
    pi = torch.tensor([np.pi])
    prefix = torch.sqrt(1/pi/2/sigma**2)
    if bsXs is not None:
        v = torch.exp(-(y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))**2/2/sigma**2)
    else:
        v = torch.exp(-(y-m)**2/2/sigma**2)
    return prefix*v


def fn2(y, m, bsXs=None, sigma=sigma):
    pi = torch.tensor([np.pi])
    sigma2 = sigma**2
    prefix = torch.sqrt(1/pi/2/sigma2)
    #v1 = torch.exp(-(y-m)**2/2/sigma2)
    #v2 =  (y-m)/sigma2
    if bsXs is not None:
        return prefix*torch.exp(-(y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))**2/2/sigma2)*(y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))/sigma2
    else:
        return prefix*torch.exp(-(y-m)**2/2/sigma2)*(y-m)/sigma2


def fn22(y, m, sigma=sigma):
    pi = torch.tensor([np.pi])
    sigma2 = sigma**2
    prefix = torch.sqrt(1/pi/2/sigma2)
    expitm = torch.exp(-(y-m)**2/2/sigma2)
    v =  (y-m)/sigma2
    linitm = v**2-1/sigma2
    return prefix*linitm*expitm


r = 2
s = 2
n = 100
m = 100
p = 100
N = 20000
STm = np.sqrt(n*m/10000)

prob = 0.1
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, 0, 0, 0]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m, rank=r) * 7 
conDenfs = [fn, fn2, fn22]



numIter = 10
eta = 0.01 
tol = 1e-4
TrueParas = [beta0, bTheta0]
betainit = beta0* 1.1
bThetainit = bTheta0 * 1.1

Cb, CT, ST = 465.9, 0.766, 102.3

outputs = []
outputs.append({"s":s, "r":r, "p":p, "m":m, "n":n, "bTheta0": bTheta0.cpu(), "beta0":beta0.cpu()})
for i in range(numIter):
    X = genXdis(n, m, p, type="Bern", prob=prob) 
    Y = genYnorm(X, bTheta0, beta0, sigma=sigma)
    R = genR(Y, inp=4.65)
    print(R.sum()/R.numel())
    sXs = genXdis(N, p, type="Bern", prob=prob) 
    betahat, bThetahat, _, numI, Berrs, Terrs = MCGDBern(1000, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, eta=eta, Cb=Cb, CT=CT, tol=tol, log=2, ST=ST, prob=prob, betainit=betainit, bThetainit=bThetainit, ErrOpts=1)
    errb = torch.norm(beta0-betahat)
    errT = torch.norm(bTheta0-bThetahat)
    outputs.append((numI, errb.item(), errT.item(), betahat.norm().item(), bThetahat.norm().item()))
    print(
        f"The Iteration number is {numI}, "
        f"The error of beta is {errb.item():.3f}, "
        f"The error of bTheta is {errT.item():.3f}."
    )

f = open(f"./outputs/Bern_{s}_{r}_{p}_{m}_{n}.pkl", "wb")
pickle.dump(outputs, f)
f.close()
