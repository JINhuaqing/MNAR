from utilities import *
from utilities_mar import MarBthetaBern
import random
import numpy as np
import torch
import pickle
import timeit
import time
from confs import fln, fln2, fln22

torch.cuda.set_device(2)
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
X = genXdis(n, m, p, type="Bern", prob=prob) 
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m, rank=5) * 8 
#M = bTheta0 + X.matmul(beta0)
Y = genYlogit(X, bTheta0, beta0)
R = genR(Y, inp=1.25)
# TO find the missing rate, I control the missing rate around 0.25
print(R.sum()/R.numel())
# The likelihood and its derivatives of Y|X
conDenfs = [fln, fln2, fln22]

#------------------------------------------------------------------------------------
# Termination  tolerance.
tol = 1e-10
TrueParas = [beta0, bTheta0]
# The list to contain output results
results = {"bTheta0":bTheta0.cpu(), "tol": tol}
# initial value of bTheta
bThetainit = bTheta0 * initthetapref
etaTs = [1e-1]
#etaTsc = [300, 180]
etaTsc = []
#------------------------------------------------------------------------------------
print({"bTheta0_norm":bTheta0.cpu().norm().item(), "tol": tol})
# The list to contain training errors 

#------------------------------------------------------------------------------------
CT = 1e-3 
results["CT"] = CT

print(f"CT is {CT:>8.4g}")
bThetahat, numI, Terrs, Likelis, bThetahats = MarBthetaBern(1000, X, Y, R, conDenfs, TrueParas=TrueParas, CT=CT, tol=tol, log=2, prob=prob, bThetainit=bThetainit, ErrOpts=1, etaTs=etaTs, etaTsc=etaTsc)
LpTTvhat = LpTTBern(bThetahat, beta0, conDenfs, X, Y, R, prob) # n x m
errT = torch.norm(bTheta0-bThetahat)
results["errT"], results["bhatnorm"], results["minEigTT"] = errT.item(), bThetahat.norm().item(), LpTTvhat.min().item()
results["numI"], results["CT"] = numI, CT
print(
      f"The Iteration number is {numI}, "
      f"The error of bTheta is {errT.item():.3f}."
     )


    
# Save the output
f = open(f"./outputs/MarBern_theta_{m}_{CT:.0E}_{etaTs[-1]:.0E}.pkl", "wb")
pickle.dump([results, Terrs, Likelis, bThetahats], f)
f.close()
