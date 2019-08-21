from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit
import time
from scipy.stats import norm as STN
from confs import fln, fln2, fln22

# mat = np.random.rand(100, 100)
# svdnp = np.linalg.svd(mat)
# Tmat = torch.tensor(mat)
# svdtorch = torch.svd(Tmat)
# 
# Unp, Snp, VTnp = svdnp
# Utorch, Storch, VTtorch = svdtorch.U, svdtorch.S, svdtorch.V.t()
# Unp, Snp, VTnp = torch.tensor(Unp), torch.tensor(Snp), torch.tensor(VTnp)
# print((Unp-Utorch).norm(), (Snp-Storch).norm(), (VTnp-VTtorch).norm())
# 
# raise SystemExit
torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
np.random.seed(0) #numpy
random.seed(0) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

cuda = torch.cuda.is_available()
# cuda = False
if cuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
sigma = 0.5
a, b = -1, 1


def torchstnpdf(y):
    pi = torch.tensor([np.pi])
    prefix = torch.sqrt(1/pi/2)
    return prefix*torch.exp(-y**2/2)

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


n = 100
m = 100
p = 100
N = 100000
sigmax = np.sqrt(1/3)

X = genXdis(n, m, p, type="bern", prob=0.1) 
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, 3, 4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m) 
M = bTheta0 + X.matmul(beta0)
Y = genYlogit(X, bTheta0, beta0)
R = genR(Y, inp=1.5)
print(R.sum()/R.numel())
sXs = genXdis(N, p, type="Bern", prob=0.1) 
conDenfs = [fln, fln2, fln22]
TrueParas = [beta0, bTheta0]
# TTv1 = missdepLpTT(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# TTv2 = LpTTBern(bTheta0, beta0, conDenfs, X, Y, R, prob=0.1)
# print((TTv1-TTv2).norm(), TTv1.norm())
# Tv1 = missdepLpT(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# Tv2 = LpTBern(bTheta0, beta0, conDenfs, X, Y, R, prob=0.1)
# print((Tv1-Tv2).norm(), Tv1.norm(), Tv2.norm())
# Bv1 = missdepLpb(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# Bv2 = LpbBern(bTheta0, beta0, conDenfs, X, Y, R, prob=0.1)
# print((Bv1-Bv2).norm())
# v1 = missdepL(bTheta0, beta0, conDenfs[0], X, Y, R, sXs)
# v2 = LBern(bTheta0, beta0, conDenfs[0], X, Y, R, prob=0.1)
# print((v1-v2).norm(), v1.norm(), v2.norm())
BthetaBern(1000, X, Y, R, conDenfs, TrueParas, CT=1e-5, log=2, prob=0.1, tol=1e-15, bThetainit=1.1*bTheta0)
