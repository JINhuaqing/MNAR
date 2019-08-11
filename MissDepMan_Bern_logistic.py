from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit
import time
from confs import fln, fln2, fln22

torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
np.random.seed(0) #numpy
random.seed(0) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

cuda = torch.cuda.is_available()
#cuda = False
if cuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

n = 100
m = 100
p = 100
N = 50000

prob = 1000/n/m
X = genXdis(n, m, p, type="Bern", prob=prob) 
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, 3, 4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m, rank=5) 
M = bTheta0 + X.matmul(beta0)
Y = genYlogit(X, bTheta0, beta0)
R = genR(Y, inp=1.4)
print(R.sum()/R.numel())
Ds2, Dh2 = Dshlowerflogit(Y, X, beta0, bTheta0)
STbd = 2*Dh2+2*Ds2**2
print(STbd)
sXs = genXdis(N, p, type="Bern", prob=prob) 
conDenfs = [fln, fln2, fln22]


eta = 0.1 
tol = 1e-6
TrueParas = [beta0, bTheta0]
betainit = beta0* 0.95
bThetainit = bTheta0 * 0.95
Lcon= 8

Cb, CT, ST = 1, 1e-4, 10
print(f"Cb is {Cb:>8.4g}, CT is {CT:>8.4g}, ST is {ST:>8.4g}")
betahat, bThetahat, _, numI, Berrs, Terrs = MCGDBern(1000, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, eta=eta, Cb=Cb, CT=CT, tol=tol, log=2, ST=ST, prob=prob, betainit=betainit, bThetainit=bThetainit, Rbinit=torch.tensor([1.0]), ErrOpts=1, Lcon=Lcon)
errb = torch.norm(beta0-betahat)
errT = torch.norm(bTheta0-bThetahat)
print(
    f"The Iteration number is {numI}, "
    f"The error of beta is {errb.item():.3f}, "
    f"The error of bTheta is {errT.item():.3f}."
)

fasdf
f = open(f"./outputs/Bern_lg_{Cb:.1g}_{CT:.1g}_{ST:.1g}.pkl", "wb")
pickle.dump([numI, betahat.cpu(), errb.cpu().item(), Berrs, bThetahat.cpu(), errT.cpu().item(), Terrs], f)
f.close()
