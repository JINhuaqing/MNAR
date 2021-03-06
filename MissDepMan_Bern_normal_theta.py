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
torch.cuda.manual_seed(0) #gpu
np.random.seed(0) #numpy
random.seed(0) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

#------------------------------------------------------------------------------------
# Whether GPU is available, 
cuda = torch.cuda.is_available()
#cuda = False
#Set default data type
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
sigmaY = 0.1
X = genXdis(n, m, p, type="Bern", prob=prob) 
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m, rank=5, sigVs=[24, 16, 16, 11, 8]) * 8 
#M = bTheta0 + X.matmul(beta0)
Y = genYnorm(X, bTheta0, beta0, sigmaY)
R = genR(Y, "quadra", a=1, b=-13)
# TO find the missing rate, I control the missing rate around 0.25
print(R.sum()/R.numel())
# The likelihood and its derivatives of Y|X
conDenfs = [fn, fn2, fn22]

#------------------------------------------------------------------------------------
# Termination  tolerance.
tol = 1e-10
tol = 0
TrueParas = [beta0, bTheta0]
# The list to contain output results
results = {"bTheta0":bTheta0.cpu(), "tol": tol}
# initial value of bTheta
bThetainit = bTheta0 * initthetapref
#------------------------------------------------------------------------------------
print({"bTheta0_norm":bTheta0.cpu().norm().item(), "tol": tol})
# The list to contain training errors 

#------------------------------------------------------------------------------------
CT = 4e-1 * 0.5
results["CT"] = CT

print(f"CT is {CT:>8.4g}")
bThetahat, numI, Terrs, Likelis, bThetahats, _ = BthetaBern(3100, X, Y, R, conDenfs, TrueParas=TrueParas, CT=CT, tol=tol, log=2, prob=prob, bThetainit=bThetainit, ErrOpts=1)
#LpTTvhat = LpTTBern(bThetahat, beta0, conDenfs, X, Y, R, prob) # n x m
errT = torch.norm(bTheta0-bThetahat)
results["errT"], results["bhatnorm"] = errT.item(), bThetahat.norm().item()
results["numI"], results["CT"] = numI, CT
print(
      f"The Iteration number is {numI}, "
      f"The error of bTheta is {errT.item():.3f}."
     )


    
# Save the output
f = open(f"./outputs/Bern_normal_theta_{m}_{CT:.0E}_{etaTs[-1]:.0E}.pkl", "wb")
pickle.dump([results, Terrs, Likelis, bThetahats], f)
f.close()
