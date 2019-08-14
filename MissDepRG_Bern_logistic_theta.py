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
n = 100
m = 100
p = 100

#------------------------------------------------------------------------------------
# The successful probability of each entry of X
prob = 1000/n/m
X = genXdis(n, m, p, type="Bern", prob=prob) 
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, 3, 4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m, rank=5) * 8
#M = bTheta0 + X.matmul(beta0)
Y = genYlogit(X, bTheta0, beta0)
R = genR(Y, inp=1.4)
# TO find the missing rate, I control the missing rate around 0.25
print(R.sum()/R.numel())
# The likelihood and its derivatives of Y|X
conDenfs = [fln, fln2, fln22]


#------------------------------------------------------------------------------------
# I use random grid search to find good parameters, so the following are the search spaces 
CTpool = np.exp(np.linspace(np.log(1e-6), np.log(1e-3), 100))

#------------------------------------------------------------------------------------
# The number of times to do random grid search
numRG = 100
# Termination  tolerance.
tol = 1e-8
TrueParas = [beta0, bTheta0]
# The list to contain output results
results = [{"bTheta0":bTheta0.cpu(), "tol": tol}]
# initial value of bTheta
bThetainit = bTheta0 * 0.95

#------------------------------------------------------------------------------------
print({"bTheta0_norm":bTheta0.cpu().norm().item(), "tol": tol})
# The list to contain training errors 
Errs = []

#------------------------------------------------------------------------------------
# Random grid search
for i in range(numRG):
    #----------------------------------------------------------------------------------------------------
    # To get the CT search space
    len1  = len(CTpool)
    CT = 10
    idx1 = np.random.randint(0, len1)
    CT = CTpool[idx1]
    #----------------------------------------------------------------------------------------------------

    print(f"The {i+1}/{numRG}, CT is {CT:>8.4g}")
    # I use try-except statement to avoid error breaking the loop
    try:
        bThetahat, numI, Terrs = BthetaBern(1000, X, Y, R, conDenfs, TrueParas=TrueParas, CT=CT, tol=tol, log=2, prob=prob, bThetainit=bThetainit, ErrOpts=1)
    except RuntimeError as e:
        results.append((-100,  CT, -100, -100))
        Errs.append([])
        print(
            f"The {i+1}th/{numRG},"
            f"Iteration Fails!", 
            e
        )
    else:
        errT = torch.norm(bTheta0-bThetahat)
        results.append((numI, CT, errT.item(), bThetahat.norm().item()))
        Errs.append(Terrs)
        print(
            f"The {i+1}th/{numRG},"
            f"The Iteration number is {numI}, "
            f"The error of bTheta is {errT.item():.3f}."
        )

# Save the output
f = open("./outputs/RandGrid_Bern_theta.pkl", "wb")
pickle.dump([results, Errs], f)
f.close()
