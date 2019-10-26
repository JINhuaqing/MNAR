from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit
import time
from confs import fln, fln2, fln22

torch.cuda.set_device(0)
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
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

#------------------------------------------------------------------------------------
# Set the number of n, m, p, N
# N is number of samples used for MCMC
n = 100
m = 100
p = 100
initthetapref = 1 + (torch.rand(n, m)-1/2)/2

#------------------------------------------------------------------------------------
# The successful probability of each entry of X
prob = 0.05
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -0, 0]), torch.zeros(p-7))) 
bTheta0 = genbTheta(n, m, sigVs=[24, 16, 16, 11, 8]) * 8
X = genXdis(n, m, p, type="Bern", prob=prob) 
Y = genYlogit(X, bTheta0, beta0)
R = genR(Y, "fixed")
# TO find the missing rate, I control the missing rate around 0.25
# generate the samples for MCMC
conDenfs = [fln, fln2, fln22]

#------------------------------------------------------------------------------------
# Termination  tolerance.
tol = 0
TrueParas = [beta0, bTheta0]
bThetainit = bTheta0 * 0# * initthetapref 
#------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------
CT = 1e-3  # 2e-3

bThetahat, numI, Terrs, Likelis, bThetahats, etabs = BthetaBern(30000, X, Y, R, conDenfs, TrueParas=TrueParas, CT=CT, tol=tol, log=2, prob=prob, bThetainit=bThetainit, ErrOpts=1)
