from utilities import *
from utilities_mar import MarBthetaBern
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
torch.cuda.manual_seed(10) #gpu
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
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

#------------------------------------------------------------------------------------
# Set the number of n, m, p, N
n = 100
m = 100
p = 2
initthetapref = 1 + (torch.rand(n, m)-1/2)/2

#------------------------------------------------------------------------------------
# The successful probability of each entry of X
prob = 0.05
X = genXdis(n, m, p, type="Bern", prob=prob) 
beta0 = torch.tensor([1.0, 1])
bTheta0 = genbTheta(n, m, sigVs=[24, 16, 16, 11, 8]) * 8/100
Y = genYlogit(X, bTheta0, beta0)
R = genR(Y, "MAR")
R = genR(Y, "fixed")
#print(R.sum()/R.numel())
# The likelihood and its derivatives of Y|X
conDenfs = [fln, fln2, fln22]

#------------------------------------------------------------------------------------
# Termination  tolerance.
tol = 0
TrueParas = [beta0, bTheta0]
# The list to contain output results
bThetainit = bTheta0 * initthetapref
#------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------
CT = 2e-3 
etaT = 50000
bThetahat, numI, Terrs, Likelis, bThetahats = MarBthetaBern(100000, X, Y, R, conDenfs, TrueParas=TrueParas, CT=CT, tol=tol, log=2, prob=prob, bThetainit=bThetainit, ErrOpts=1, etaT=etaT)
