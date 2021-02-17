from utils import *
from utils_EM import EMNewBern
import random
import numpy as np
import torch
import pickle
import timeit
import time
import pprint
from pathlib import Path
from confs import fn, fn2, fn22

cudaid = 2
# bs for prob = 1000/n/m 
bs = {
        50: 15.,
        100:2.55,
        200:0.94,
        400:0.80,
        800:0.77,
        1600:0.76,
        3200:0.75,
     }

torch.cuda.set_device(cudaid)
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
    #torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

#------------------------------------------------------------------------------------
# Set the number of n, m, p
# N is number of samples used for MCMC
n = m = 200
p = 50

initbetapref = 1 + (torch.rand(p)-1/2)/4  #[0.75, 1.25]
initthetapref = 1 + (torch.rand(n, m)-1/2)/4
#------------------------------------------------------------------------------------
# The successful probability of each entry of X
prob =  1000/n/m
sigmaY = 0.1

# generate the parameters
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7))) * 10
bTheta0 = genbTheta(n, m, sigVs=np.array([10, 9, 8, 7, 6])/2) 
TrueParas = [beta0, bTheta0]
# initial value of beta and bTheta
betainit = beta0 * initbetapref
bThetainit = bTheta0 * initthetapref
# The likelihood and its derivatives of Y|X
conDenfs = [fn, fn2, fn22]



#------------------------------------------------------------------------------------
# Termination  tolerance.
loglv = 2
tols = [0, 0, 0]
#tols = [0, 5e-3, 1e-3]
# 100: 
# Cb, CT = 1000, 10e-2; 
# etab, etaT = 0.05, 0.02
# 200
#Cb, CT = 800, 5.1e-2
#etab, etaT = 0.09, 0.05
# 400
# Cb, CT = 800, 5.0e-2
# etab, etaT = 0.01, 0.01
# 800 
# Cb, CT = 800, 5e-2
# etab, etaT = 0.90, 0.08
# 1600
Cb, CT = 800, 5.1e-2; 
etab, etaT = 0.10, 0.10

# The list to contain output results
params = {"beta0":beta0.cpu().numpy(), "bTheta0":bTheta0.cpu().numpy(), "tols": tols, "CT":CT, "Cb":Cb }
params["n"] = n
params["m"] = m
params["p"] = p
params["Xtype"] = "Bernoulli"
params["Y|X_type"] = "Normal"
params["MissRate"] = []

pprint.pprint(params)

# ------------------------------------------------------------------------------------
# The list to contain training errors 

numSimu = 50 
root = Path("./results")
startIdx = 1

#------------------------------------------------------------------------------------
for i in range(numSimu):
    # generate the samples
    X = genXBin(n, m, p, prob=prob, is_sparse=True) 
    Y = genYnorm(X, bTheta0, beta0, sigmaY)
    R = genR(Y, "linear", inp=bs[m], slop=5, is_sparse=True) # 
    X = X.to_dense()
    R = R.to_dense()
    # I control the missing rate around 0.25
    MissRate = R.sum()/R.numel()
    print(MissRate)
    params["MissRate"] = MissRate
    #----------------------------------------------------------------------------------------------------
    print(f"The {i+1}th/{numSimu}")
    if (i+1) >= startIdx:
        betahat, bThetahat, numI, Berrs, Terrs, betahats, bThetahats, Likelis = EMNewBern(100000, X, Y, R, conDenfs, TrueParas=TrueParas, etab=etab, Cb=Cb, CT=CT, tols=tols, log=loglv, prob=prob, betainit=betainit, bThetainit=bThetainit, ErrOpts=1, etaT=etaT)
        errb = torch.norm(beta0-betahat)
        errT = torch.norm(bTheta0-bThetahat)

        print(
        f"The {i+1}th/{numSimu},"
        f"The Iteration number is {numI}, "
        f"The error of beta is {errb.item():.3f}, "
        f"The error of bTheta is {errT.item():.3f}.")

        curResult = {}
        curResult["numI"] = numI
        curResult["Cb"] = Cb
        curResult["CT"] = CT
        curResult["errb"] = errb.cpu().numpy()
        curResult["errT"] = errT.cpu().numpy()
        curResult["Berrs"] = Berrs
        curResult["Terrs"] = Terrs
        curResult["betahat"] = betahat.cpu().numpy()
        curResult["bThetahat"] = bThetahat.cpu().numpy()

        filName = f"EM_linear_p{p}_m{m}_simu{i+1}_iter{numI}.pkl"
        with open(root/filName, "wb") as f:
            pickle.dump([params, curResult], f)
