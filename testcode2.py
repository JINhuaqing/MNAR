from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit
import time
import argparse
import pprint
from confs import fln, fln2, fln22

m = 300
n = 300
d = np.sqrt(n*m)
p = 100
cudaid = 0
N = 20000

torch.cuda.set_device(cudaid)
#------------------------------------------------------------------------------------
# fix the random seed for several packages
torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
np.random.seed(0) #numpy
random.seed(0) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

#------------------------------------------------------------------------------------
# Whether GPU is available, 
cuda = torch.cuda.is_available()
#cuda = False
# Set default data type
if cuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    #torch.set_default_tensor_type(torch.cuda.DoubleTensor)
#ddtype = torch.float64
#torch.set_default_dtype(ddtype)

#------------------------------------------------------------------------------------
# Set the number of n, m, p, N
# N is number of samples used for MCMC

initbetapref = 1 + (torch.rand(p)-1/2)  #[0.75, 1.25]
initthetapref = 1 + (torch.rand(n, m)-1/2)/2

#------------------------------------------------------------------------------------
# The successful probability of each entry of X
prob = 0.05 # 1000/n/m
# generate the parameters
# beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7))) # orginal beta0
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m, sigVs=[24, 16, 16, 11, 8])/2
#print(bTheta0.abs().max(), 1/d)
TrueParas = [beta0, bTheta0]
# initial value of beta and bTheta
betainit = beta0 * initbetapref
betainit = beta0 
bThetainit = bTheta0 * initthetapref
bThetainit = bTheta0
# The likelihood and its derivatives of Y|X
conDenfs = [fln, fln2, fln22]



#------------------------------------------------------------------------------------
# Termination  tolerance.
CBTdic = {
100: [20, 1e-3/2],
150: [25, 1e-3/2],
200: [30, 1e-3/2],
250: [35, 1e-3/2.4],
300: [45, 1e-3/2.8]
}
tols = [2.7e-4, 2.65e-6, 1.9e-6] # [0.5, 1.5]
tols = [0, 0, 0]
Cb, CT = 8, 1e-3/2
Cb, CT = CBTdic[m]
#m=n=100, 20, 1e-3/2
#m=n=150, 25, 1e-3/2
#m=n=200, 30, 1e-3/2
#m=n=250, 35, 1e-3/2.4
#m=n=300, 45, 1e-3/2.8

# generate the samples
X = genXdis(n, m, p, type="Bern", prob=prob) 
Y = genYlogit(X, bTheta0, beta0)
R = genR(Y, "fixed")
# TO find the missing rate, I control the missing rate around 0.25
# generate the samples for MCMC
sXs = genXdis(N, p, type="Bern", prob=prob) 
betahat, bThetahat, numI, Berrs, Terrs, betahats, bThetahats, Likelis, etass = NewBern(10000, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, Cb=Cb, CT=CT, tols=tols, log=2, prob=prob, betainit=betainit, bThetainit=bThetainit, ErrOpts=1)

