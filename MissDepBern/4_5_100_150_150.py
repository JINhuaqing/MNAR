import os
import sys
#sys.path.append("/home/user2/Documents/JIN/MNAR")
#os.chdir("/home/user2/Documents/JIN/MNAR")
#sys.path.append("/home/feijiang/jin/MissDep")
#os.chdir("/home/feijiang/jin/MissDep")
sys.path.append("/workspace/data/jin/MNAR")
os.chdir("/workspace/data/jin/MNAR")
from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit
import time
from scipy.stats import norm as STN

torch.cuda.set_device("cuda:0")
torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
np.random.seed(0) #numpy
random.seed(0) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

cuda = torch.cuda.is_available()
#cuda = False
torch.set_num_threads(8)
if cuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
sigma = 0.5
a, b = -10, 10 

def torchstnpdf(y):
    pi = torch.tensor([np.pi])
    prefix = torch.sqrt(1/pi/2)
    return prefix*torch.exp(-y**2/2)


def ftn(y, m, bsXs=None, sigma=sigma, a=a, b=b):
    # y     : n x m
    # m     : n x m 
    # bsXs  : N
    Z = (STN.cdf(b/sigma) - STN.cdf(a/sigma)) * sigma
    if bsXs is not None:
        v = (y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))/sigma
        zidx = (y.unsqueeze(-1) > b + (m.unsqueeze(-1) + bsXs)) | (y.unsqueeze(-1) < a + (m.unsqueeze(-1) + bsXs))
    else:
        v = (y-m)/sigma
        zidx = (y > b + m) | (y < a+m)
    tv = torchstnpdf(v)/Z 
    tv[zidx] = 0
    return tv


def ftn2(y, m, bsXs=None, sigma=sigma, a=a, b=b):
    sigma2 = sigma**2
    Z = (STN.cdf(b/sigma) - STN.cdf(a/sigma)) * sigma
    if bsXs is not None:
        v = (y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))/sigma
        v2 = (y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))/sigma2
        zidx = (y.unsqueeze(-1) > b + (m.unsqueeze(-1) + bsXs)) | (y.unsqueeze(-1) < a + (m.unsqueeze(-1) + bsXs))
    else:
        v = (y-m)/sigma
        v2 = (y-m)/sigma2
        zidx = (y > b + m) | (y < a+m)
    tv = torchstnpdf(v)*v2/Z
    tv[zidx] = 0
    return tv


def ftn22(y, m, bsXs=None, sigma=sigma, a=a, b=b):
    sigma2 = sigma**2
    Z = (STN.cdf(b/sigma) - STN.cdf(a/sigma)) * sigma
    if bsXs is not None:
        v = (y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))/sigma
        v2 = ((y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))/sigma2)**2 - 1/sigma2
        zidx = (y.unsqueeze(-1) > b + (m.unsqueeze(-1) + bsXs)) | (y.unsqueeze(-1) < a + (m.unsqueeze(-1) + bsXs))
    else:
        v = (y-m)/sigma
        v2 = ((y-m)/sigma2)**2 - 1/sigma2
        zidx = (y > b + m) | (y < a+m)
    tv = torchstnpdf(v)*v2/Z
    tv[zidx] = 0
    return tv



r = 4
s = 5
n = 150
m = 150
p = 100
N = 20000

prob = 0.1
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, 3, 4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m, rank=r) * 7 
conDenfs = [ftn, ftn2, ftn22]



numIter = 110
eta = 0.01 
tol = 1e-5
TrueParas = [beta0, bTheta0]
betainit = beta0* 1.1
bThetainit = bTheta0 * 1.1

Cb, CT, CST = 146.479, 9.222, 1.322

outputs = []
outputs.append({"s":s, "r":r, "p":p, "m":m, "n":n, "bTheta0": bTheta0.cpu(), "beta0":beta0.cpu()})
for i in range(numIter):
    print(f"Simulation {i+1:>5}/{numIter},")
    X = genXdis(n, m, p, type="Bern", prob=prob) 
    Y = genYtnorm(X, bTheta0, beta0, a, b, sigma=sigma)
    R = genR(Y, inp=6.5)
    print(R.sum()/R.numel())
    Ds2, Dh2 = Dshlowerfnorm(Y, X, beta0, bTheta0, sigma)
    ST = CST*(2*Dh2+2*Ds2**2)
    sXs = genXdis(N, p, type="Bern", prob=prob) 
    betahat, bThetahat, _, numI, Berrs, Terrs = MCGDBern(1500, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, eta=eta, Cb=Cb, CT=CT, tol=tol, log=0, ST=ST, prob=prob, betainit=betainit, bThetainit=bThetainit, ErrOpts=1, sps=0.05)
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
