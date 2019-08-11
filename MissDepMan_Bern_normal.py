from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit
import time
from scipy.stats import norm as STN
from confs import fln, fln2, fln22

torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
np.random.seed(0) #numpy
random.seed(0) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

cuda = torch.cuda.is_available()
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


n = 100
m = 100
p = 100
N = 20000

prob = 0.1
X = genXdis(n, m, p, type="Bern", prob=prob) 
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, 3, 4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m) * 7 
M = bTheta0 + X.matmul(beta0)
Y = genYnorm(X, bTheta0, beta0, sigma=sigma)
R = genR(Y, inp=6.5)
Ds2, Dh2 = Dshlowerfnorm(Y, X, beta0, bTheta0, sigma)
STbd = 2*Dh2+2*Ds2**2
print(R.sum()/R.numel())
print(STbd)
sXs = genXdis(N, p, type="Bern", prob=prob) 
conDenfs = [fn, fn2, fn22]


eta = 0.001 
tol = 1e-4
TrueParas = [beta0, bTheta0]
betainit = beta0*0.99
bThetainit = bTheta0 * 0.99

Cb, CT, ST = 466, 0.766, 102
print(f"Cb is {Cb:>8.4g}, CT is {CT:>8.4g}, ST is {ST:>8.4g}")
betahat, bThetahat, _, numI, Berrs, Terrs = MCGDBern(1000, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, eta=eta, Cb=Cb, CT=CT, tol=tol, log=2, ST=ST, prob=prob, betainit=betainit, bThetainit=bThetainit, ErrOpts=1)
errb = torch.norm(beta0-betahat)
errT = torch.norm(bTheta0-bThetahat)
print(
            f"The Iteration number is {numI}, "
            f"The error of beta is {errb.item():.3f}, "
            f"The error of bTheta is {errT.item():.3f}."
        )

fasdfas
f = open("./outputs/RandGrid_Bern_TN_2w_01_r4s5_11.pkl", "wb")
pickle.dump([results, Errs], f)
f.close()
