from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit

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


def fn(y, m, sigma=sigma):
    pi = torch.tensor([np.pi])
    prefix = torch.sqrt(1/pi/2/sigma**2)
    v = torch.exp(-(y-m)**2/2/sigma**2)
    return prefix*v


def fn2(y, m, sigma=sigma):
    pi = torch.tensor([np.pi])
    sigma2 = sigma**2
    prefix = torch.sqrt(1/pi/2/sigma2)
    v1 = torch.exp(-(y-m)**2/2/sigma2)
    v2 =  (y-m)/sigma2
    return prefix*v1*v2


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
N = 1000
sigmax = np.sqrt(1/3)

X = genXdis(n, m, p, type="mvnorm", sigmax=sigmax) 
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, 3, 4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m) * 7
M = bTheta0 + X.matmul(beta0)
Y = genYnorm(X, bTheta0, beta0, sigma=sigma)
R = genR(Y)
# print(X.matmul(beta0).abs().mean(), bTheta0.abs().mean())
print(R.sum()/R.numel())
sXs = genXdis(N, p, type="mvnorm", sigmax=sigmax) 
conDenfs = [fn, fn2, fn22]
# etascaleinv = missdepLpbbLoop(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# etascale = etascaleinv.inverse()
# U, S, V = etascale.svd()
# print(S.max())

# Lv = missdepL(bTheta0, beta0, fn, X, Y, R, sXs)
# Lvn = Lnormal(bTheta0, beta0, X, Y, R, sigmax=sigmax, sigma=sigma)
# Lv2 = missdepLLoop(bTheta0, beta0, fn, X, Y, R, sXs)
# LpTv = missdepLpT(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# LpTv2 = missdepLpTLoop(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# Lpbv = missdepLpb(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# Lpbv2 = missdepLpbLoop(bTheta0, beta0, conDenfs, X, Y, R, sXs)





eta = 1/(5*0.75*m*p)
# eta = 0.01 
Cb, CT = 73, 0.13
tol = 1e-4
TrueParas = [beta0, bTheta0]
betahat, bThetahat, _, Inum = MCGDnormal(2000, X, Y, R, TrueParas=TrueParas, eta=eta, Cb=Cb, CT=CT, tol=tol, log=2, sigmax=sigmax,sigma=sigma)
print((bThetahat-bTheta0).norm())
