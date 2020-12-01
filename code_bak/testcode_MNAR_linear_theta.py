from utilities import *
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
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
#    torch.set_default_tensor_type(torch.cuda.FloatTensor)

#------------------------------------------------------------------------------------
bs = {
        100:1.4,
        150:1.2,
        300:1.1,
        250:1.1,
        200:1.1,
     }
# Set the number of n, m, p, N
n = 100
m = 100
p = 2
initthetapref = 1 + (torch.rand(n, m)-1/2)/4

#------------------------------------------------------------------------------------
# The successful probability of each entry of X
prob = 0.05
sigmaY = 0.1
#beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7))) 
beta0 = torch.tensor([1.0, 1])
bTheta0 = genbTheta(n, m, sigVs=np.array([10, 9, 8, 7, 6])*100/np.sqrt(m*n)) 
#bTheta0 = genbTheta(n, m, sigVs=np.array([200, 160, 120, 80, 40])*100/np.sqrt(m*n)) 
#print(bTheta0.abs().max(), 100/np.sqrt(m*n))
res = torch.svd(bTheta0)
X = genXdis(n, m, p, type="Bern", prob=prob) 
Y = genYnorm(X, bTheta0, beta0, sigmaY)
#R = genR(Y, "quadra", a=1, b=bs[m]) # 
R = genR(Y, "linear", inp=bs[m]) # 
print(R.sum()/R.numel())
conDenfs = [fn, fn2, fn22]

#------------------------------------------------------------------------------------
# Termination  tolerance.
tol = 0
TrueParas = [beta0, bTheta0]
bThetainit = bTheta0 * initthetapref 
#------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------
CT = 1e-6  # 2e-3

bThetahat, numI, Terrs, Likelis, bThetahats, etabs = BthetaBern(90000, X, Y, R, conDenfs, TrueParas=TrueParas, CT=CT, tol=tol, log=2, prob=prob, bThetainit=bThetainit, ErrOpts=1)
