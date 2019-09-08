from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit
import time
from confs import fln, fln2, fln22


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
n = 200
m = 200
p = 100
N = 20000
prefix = (n*m/10000)

#------------------------------------------------------------------------------------
# The successful probability of each entry of X
initbetapref = 1 + (torch.rand(p)-1/2)/2  #[-0.75, 1.25]
initthetapref = 1 + (torch.rand(n, m)-1/2)/2
prob = 0.05
X = genXdis(n, m, p, type="Bern", prob=prob) 
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m, rank=5) * 8
Y = genYlogit(X, bTheta0, beta0)
R = genR(Y, inp=1.25)
# TO find the missing rate, I control the missing rate around 0.25
print(R.sum()/R.numel())
# generate the samples for MCMC
sXs = genXdis(N, p, type="Bern", prob=prob) 
# The likelihood and its derivatives of Y|X
conDenfs = [fln, fln2, fln22]


#------------------------------------------------------------------------------------
# The number of times to do random grid search
etabs = [prefix*1e-1, prefix*5e-1]
etaTs = [5e-1, 1e-2]
etabsc = []
etaTsc = [140]
# Termination  tolerance.
#tol = 1e-8
tols = [1e-10, 2.65e-6, 1.9e-6]
TrueParas = [beta0, bTheta0]
# The list to contain output results
results = [{"beta0":beta0.cpu(), "bTheta0":bTheta0.cpu(), "tols": tols}]

#------------------------------------------------------------------------------------
print(results)
# The list to contain training errors 

#------------------------------------------------------------------------------------
# Random grid search
    # initial value of beta and bTheta
# initial value of beta and bTheta
betainit = beta0 * initbetapref
bThetainit = bTheta0 * initthetapref
Cb, CT = 8, 2e-3 
#   m, Cb
# 100, 6
#
#
#----------------------------------------------------------------------------------------------------

print(f"Cb is {Cb:>8.4g}, CT is {CT:>8.4g}")
betahat, bThetahat, numI, Berrs, Terrs, betahats, bThetahats, Likelis = NewBern(300, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, etabs=etabs, etabsc=etabsc, Cb=Cb, CT=CT, tols=tols, log=2, prob=prob, betainit=betainit, bThetainit=bThetainit, ErrOpts=1, etaTs=etaTs, etaTsc=etaTsc)
errb = torch.norm(beta0-betahat)
errT = torch.norm(bTheta0-bThetahat)
results.append((numI, Cb, errb.item(), betahat.norm().item(), CT, errT.item(), bThetahat.norm().item()))
Errs = [Berrs, Terrs, betahats, bThetahats, Likelis]
print(
    f"The Iteration number is {numI}, "
    f"The error of beta is {errb.item():.3f}, "
    f"The error of bTheta is {errT.item():.3f}."
)

# Save the output
f = open(f"./outputs/Man_Bern_new_{m:.0f}_{Cb:.0f}.pkl", "wb")
pickle.dump([results, Errs], f)
f.close()
