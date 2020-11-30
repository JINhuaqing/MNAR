from utils import *
#from utilities_mar import MarNewBern
import random
import numpy as np
import torch
import pickle
import timeit
import time
import argparse
import pprint
from confs import fn, fn2, fn22


def Cbsf(m):
    if m <= 200:
        return 400
    else:
        return 400
    
cudaid = 0
torch.cuda.set_device(cudaid)
torch.manual_seed(0) # cpu
torch.cuda.manual_seed(2) #gpu
np.random.seed(0) #numpy
random.seed(0) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

# Whether GPU is available, 
cuda = torch.cuda.is_available()
if cuda:
    #torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    
m = 100
n = 100
p = 50
N = 1000
prob = 0.05
sigmaY = 0.1

numSimu = 10
loglv = 2
bs = {
        100:1.4,
        125:1.3,
        150:1.25,
        175:1.2,
        200:1.18,
        225:1.16,
        250:1.16,
        275:1.16,
        300:1.1
     }




initbetapref = 1 + (torch.rand(p)-1/2)/4  #[0.75, 1.25]
initthetapref = 1 + (torch.rand(n, m)-1/2)/4

# generate the parameters
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m, sigVs=np.array([10, 9, 8, 7, 6])*100/np.sqrt(m*n)) 
TrueParas = [beta0, bTheta0]

# initial value of beta and bTheta
betainit = beta0 * initbetapref
bThetainit = bTheta0 * initthetapref

# The likelihood and its derivatives of Y|X
conDenfs = [fn, fn2, fn22]

tols = [2.7e-14, 2.65e-9, 1.9e-9] # [0.5, 1.5]
tols = [0, 1e-5, 5e-4]
Cb, CT = 50000, 2e-2

X = genXBin(n, m, p, prob=prob, is_sparse=True) 
Y = genYnorm(X, bTheta0, beta0, sigmaY)
R = genR(Y, "linear", inp=0, is_sparse=True) # 
sXs = X.to_dense().reshape(-1, p).t().to_sparse()

# I control the missing rate around 0.25
LpTv1 = missdepLpT1(bThetainit, betainit, conDenfs, X, Y, R, sXs)
LpTv0 = missdepLpT(bThetainit, betainit, conDenfs, X, Y, R, fct=10)
LpTv2 = LpTBern(bThetainit, betainit, conDenfs, X, Y, R, prob=prob, fct=10)
print(torch.norm(LpTv0-LpTv2))
print(torch.norm(LpTv0-LpTv1))

Lpbv1 = missdepLpb1(bThetainit, betainit, conDenfs, X, Y, R, sXs)
Lpbv0 = missdepLpb(bThetainit, betainit, conDenfs, X, Y, R, fct=10)
Lpbv2 = LpbBern(bThetainit, betainit, conDenfs, X, Y, R, prob=prob)
print(torch.norm(Lpbv0-Lpbv2))
print(torch.norm(Lpbv0-Lpbv1))

L1 = missdepL1(bThetainit, betainit, fn, X, Y, R, sXs)
L0 = missdepL(bThetainit, betainit, fn, X, Y, R, fct=10)
L2 = LBern(bThetainit, betainit, fn, X, Y, R, prob=prob)
print(torch.norm(L0-L2))
print(torch.norm(L0-L1))
