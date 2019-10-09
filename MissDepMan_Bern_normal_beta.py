from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit
import time
import pprint
from confs import fn, fn2, fn22

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

#------------------------------------------------------------------------------------
# Set the number of n, m, p, N
# N is number of samples used for MCMC
n = 100
m = 100
p = 100
N = 20000
prefix = n*m/10000
initbetapref = 1 + (torch.rand(p)-1/2) #[-0.75, 1.25]

#------------------------------------------------------------------------------------
# The successful probability of each entry of X
prob = 0.05 #1000/n/m
sigmaY = 0.1
X = genXdis(n, m, p, type="Bern", prob=prob) 
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m, rank=5, sigVs=[24, 16, 16, 11, 8]) * 8 
#M = bTheta0 + X.matmul(beta0)
Y = genYnorm(X, bTheta0, beta0, sigmaY)
R = genR(Y, "quadra", a=1, b=-8) # 
# n=m=100, b=-13
# n=m=140, b=-8
# TO find the missing rate, I control the missing rate around 0.25
MissRate = R.sum()/R.numel()
print(MissRate)
sXs = genXdis(N, p, type="Bern", prob=prob) 

# The likelihood and its derivatives of Y|X
conDenfs = [fn, fn2, fn22]

#------------------------------------------------------------------------------------
# Termination  tolerance.
Cb_adj = 0.05  # constant before beta algorithm lambda_beta , should be smaller than 1 
#Cb_adj = 40/(100 * np.sqrt(100*np.log(100)/np.log(200))) # constant before beta algorithm lambda_beta
tol = 1e-9
etabs = [prefix*1e-3]
etabsc = []
TrueParas = [beta0, bTheta0]
# The list to contain output results
params = {"bTheta0":bTheta0.cpu(), "tol": tol}
params = {"beta0":beta0.cpu().numpy(), "bTheta0":bTheta0.cpu().numpy(), "tol": tol}
params["etabsc"] = etabsc
params["n"] = n
params["m"] = m
params["p"] = p
params["Cb"] = Cb_adj
params["N"] = N
params["Xtype"] = "Bernoulli"
params["Y|X_type"] = "Normal"
params["etabs"] =  etabs
params["etabsc"] =  etabsc
params["MissRate"] = MissRate.item()
pprint.pprint(params)
# initial value of bTheta
betainit = beta0 * initbetapref
#------------------------------------------------------------------------------------
results = {}

print(f"Cb_adj is {Cb_adj:>8.4g}")
# adjust constant of Lambda_beta
Cb = Cb_adj * m * np.sqrt(n*np.log(p)/np.log(m+n)) # constant before New-algorithm lambda_beta # Cb=100
betahat, numI, Berrs, Likelis, betahats = BetaBern(2000, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, Cb=Cb, tol=tol, log=2, prob=prob, betainit=betainit, ErrOpts=1, etabs=etabs, etabsc=etabsc)
errb = torch.norm(beta0-betahat)
results["errb"], results["betanorm"] = errb.item(), betahat.norm().item()
results["numI"], results["Cb"] = numI, Cb
print(
      f"The Iteration number is {numI}, "
      f"The error of beta is {errb.item():.3f}."
     )


    
# Save the output
f = open(f"./outputs/Bern_normal_beta_{m}_{Cb_adj:.0E}_{etabs[-1]:.0E}.pkl", "wb")
pickle.dump([params, results, Berrs, Likelis, betahats], f)
f.close()
