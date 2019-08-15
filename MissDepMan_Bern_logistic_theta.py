from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit
import time
from confs import fln, fln2, fln22

#------------------------------------------------------------------------------------ # fix the random seed for several packages
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
n = 500
m = 500
p = 100

#------------------------------------------------------------------------------------
# The successful probability of each entry of X
prob = 1000/n/m
X = genXdis(n, m, p, type="Bern", prob=prob) 
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m, rank=5) * 8 
#M = bTheta0 + X.matmul(beta0)
Y = genYlogit(X, bTheta0, beta0)
R = genR(Y, inp=1.3)
# TO find the missing rate, I control the missing rate around 0.25
print(R.sum()/R.numel())
# The likelihood and its derivatives of Y|X
conDenfs = [fln, fln2, fln22]

#LpTv0 = LpTBern(bTheta0, beta0, conDenfs, X, Y, R, prob)
#S = torch.svd(LpTv0).S
#print(S.max(), "---")

#------------------------------------------------------------------------------------
# I use random grid search to find good parameters, so the following are the search spaces 
CTpool = np.exp(np.linspace(np.log(1e-6), np.log(1e-3), 100))

#------------------------------------------------------------------------------------
# Termination  tolerance.
tol = 1e-5
TrueParas = [beta0, bTheta0]
# The list to contain output results
results = {"bTheta0":bTheta0.cpu(), "tol": tol}
# initial value of bTheta
bThetainit = bTheta0 * (1 + (torch.rand(n,m)-1/2))

#------------------------------------------------------------------------------------
print({"bTheta0_norm":bTheta0.cpu().norm().item(), "tol": tol})
# The list to contain training errors 

#------------------------------------------------------------------------------------
CT = CTpool[0] * 10000

print(f"CT is {CT:>8.4g}")
bThetahat, numI, Terrs = BthetaBern(1000, X, Y, R, conDenfs, TrueParas=TrueParas, CT=CT, tol=tol, log=2, prob=prob, bThetainit=bThetainit, ErrOpts=1, etaTs=[10, 1e-1, 5e-2], etaTsc=[300, 230])
LpTTvhat = LpTTBern(bThetahat, beta0, conDenfs, X, Y, R, prob) # n x m
errT = torch.norm(bTheta0-bThetahat)
results["errT"], results["bhatnorm"], results["minEigTT"] = errT.item(), bThetahat.norm().item(), LpTTvhat.min().item()
results["numI"], results["CT"] = numI, CT
print(
      f"The Iteration number is {numI}, "
      f"The error of bTheta is {errT.item():.3f}."
     )


    
# Save the output
f = open("./outputs/Bern_theta_100_1.pkl", "wb")
pickle.dump([results, Terrs], f)
f.close()
