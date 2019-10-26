from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit
import time
import pprint
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
ddtype = torch.float64

#------------------------------------------------------------------------------------
# Set the number of n, m, p, N
# N is number of samples used for MCMC
n = 1000
m = 1000
p = 100 
N = 10000
prefix = 1 #n*m/10000
initbetapref = 1 + (torch.rand(p)-1/2)/2 #[-0.75, 1.25]

#------------------------------------------------------------------------------------
# The successful probability of each entry of X
prob = 0.05 #1000/n/m
#beta0 = torch.tensor([1.0, 2, 3])
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m, sigVs=[24, 16, 16, 11, 8]) * 8 
TrueParas = [beta0, bTheta0]
betainit = beta0 * initbetapref
conDenfs = [fln, fln2, fln22]
X = genXdis(n, m, p, type="Bern", prob=prob) 
Y = genYlogit(X, bTheta0, beta0)
R = genR(Y, "fixed")
sXs = genXdis(N, p, type="Bern", prob=prob) 
conDenfs = [fln, fln2, fln22]

#------------------------------------------------------------------------------------
# Termination  tolerance.
Cb_adj =  0.01/2*1.5  # constant before beta algorithm lambda_beta , should be smaller than 1 
#Cb_adj = 40/(100 * np.sqrt(100*np.log(100)/np.log(200))) # constant before beta algorithm lambda_beta
tol = 0 #1e-9
TrueParas = [beta0, bTheta0]
# The list to contain output results
betainit = beta0 * 0 #* initbetapref
#betainit = torch.tensor([0, 2.0, 1])
#betainit = beta0 * (1 + 0.1 )
#------------------------------------------------------------------------------------
results = {}

print(f"Cb_adj is {Cb_adj:>8.4g}")
# adjust constant of Lambda_beta
Cb = Cb_adj * m * np.sqrt(n*np.log(p)/np.log(m+n)) # constant before both algorithm lambda_beta 
betahat, numI, Berrs, Likelis, betahats, _ = BetaBern(13000, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, Cb=Cb, tol=tol, log=2, prob=prob, betainit=betainit, ErrOpts=1)
errb = torch.norm(beta0-betahat)
fsadfdsafsda
results["errb"], results["betanorm"] = errb.item(), betahat.norm().item()
results["numI"], results["Cb"] = numI, Cb
print(
      f"The Iteration number is {numI}, "
      f"The error of beta is {errb.item():.3f}."
     )


    
