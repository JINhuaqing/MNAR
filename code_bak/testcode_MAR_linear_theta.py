from utilities import *
from utilities_mar import MarBthetaBern
import random
import numpy as np
import torch
import pickle
import timeit
import time
from confs import fn, fn2, fn22

torch.cuda.set_device(0)
#------------------------------------------------------------------------------------ 
# fix the random seed for several packages
torch.manual_seed(0) # cpu
torch.cuda.manual_seed(200) #gpu
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
bs = {
        100:1.4,
        150:1.2,
        300:1.1,
        250:1.1,
        200:1.1,
     }
# Set the number of n, m, p, N
n = 200
m = 200
p = 2
sigmaY = 0.1
prob = 0.05
initthetapref = 1 + (torch.rand(n, m)-1/2)/4

#------------------------------------------------------------------------------------
# The successful probability of each entry of X
X = genXdis(n, m, p, type="Bern", prob=prob) 
beta0 = torch.tensor([1.0, 1])
#bTheta0 = genbTheta(n, m, sigVs=[5, 4, 3, 2, 1]) 
bTheta0 = genbTheta(n, m, sigVs=np.array([10, 9, 8, 7, 6])*100/np.sqrt(m*n))  
Y = genYnorm(X, bTheta0, beta0, sigmaY)
#R = genR(Y, "quadra", a=1, b=bs[m]) # 
R = genR(Y, "linear", inp=bs[m]) # 
# R = genR(Y, "MAR")
print(R.sum()/R.numel())
conDenfs = [fn, fn2, fn22]


#------------------------------------------------------------------------------------
# Termination  tolerance.
tol = 0
TrueParas = [beta0, bTheta0]
# The list to contain output results
bThetainit =  bTheta0 #* initthetapref
#------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------
CT = 1e-3
etaT = 1
bThetahat, numI, Terrs, Likelis, bThetahats = MarBthetaBern(100000, X, Y, R, conDenfs, TrueParas=TrueParas, CT=CT, tol=tol, log=2, prob=prob, bThetainit=bThetainit, ErrOpts=1, etaT=etaT)
