from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit
import time
import pprint
from confs import fln, fln2, fln22

torch.cuda.set_device(2)
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

#------------------------------------------------------------------------------------
# Set the number of n, m, p, N
# N is number of samples used for MCMC
n = 100
m = 100
p = 100
N = 20000
prefix = n*m/10000

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
MissRate = R.sum()/R.numel()
sXs = genXdis(N, p, type="Bern", prob=prob) 
# The likelihood and its derivatives of Y|X
conDenfs = [fln, fln2, fln22]

#------------------------------------------------------------------------------------
# Termination  tolerance.
Cb = 10
tol = 1e-9
etabs = [1e-1, 2*5e-1]
etabsc = [1000]
TrueParas = [beta0, bTheta0]
# The list to contain output results
params = {"bTheta0":bTheta0.cpu(), "tol": tol}
params = {"beta0":beta0.cpu().numpy(), "bTheta0":bTheta0.cpu().numpy(), "tol": tol}
params["etabsc"] = etabsc
params["n"] = n
params["m"] = m
params["p"] = p
params["Cb"] = Cb
params["N"] = N
params["Xtype"] = "Bernoulli"
params["Y|X_type"] = "logistic"
params["etabs"] =  etabs
params["etabsc"] =  etabsc
params["MissRate"] = MissRate.item()
pprint.pprint(params)
# initial value of bTheta
betainit = beta0 * (1 + (torch.rand(p)-0.5))
#------------------------------------------------------------------------------------
results = {}

print(f"Cb is {Cb:>8.4g}")
betahat, numI, Berrs, Likelis, betahats = BetaBern(2000, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, Cb=Cb, tol=tol, log=2, prob=prob, betainit=betainit, ErrOpts=1, etabs=etabs, etabsc=etabsc)
errb = torch.norm(beta0-betahat)
results["errb"], results["betanorm"] = errb.item(), betahat.norm().item()
results["numI"], results["Cb"] = numI, Cb
print(
      f"The Iteration number is {numI}, "
      f"The error of beta is {errb.item():.3f}."
     )


    
# Save the output
f = open(f"./outputs/Bern_beta_100_{Cb}_{etabs[-1]:.0E}.pkl", "wb")
pickle.dump([params, results, Berrs, Likelis, betahats], f)
f.close()
