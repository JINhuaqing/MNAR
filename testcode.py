from utils import *
import random
import numpy as np
import torch
import pickle
import timeit
import time
import argparse
import pprint
from pathlib import Path
from confs import fn, fn2, fn22

cudaid = 2
loglv = 2
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
# bs for prob = 0.05
#bs = {
#        50: 3.13,
#        100:1.40,
#        200:1.18,
#        400:1.16,
#        800:1.16,
#        1600:1.16,
#        3200:1.16,
#     }
#
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
# m=n=100, 200, 400, 800, 1600, 3200
n = m = 1600
p = 50

initbetapref = 1 + (torch.rand(p)-1/2)/4  #[0.875, 1.125]
initthetapref = 1 + (torch.rand(n, m)-1/2)/4
#------------------------------------------------------------------------------------
# The successful probability of each entry of X
prob = 1000/n/m # 0.05
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
tols = [2.7e-14, 2.65e-9, 1.9e-9] # [0.5, 1.5]
#tols = [0, 1e-5, 5e-4]
tols = [0, 5e-3, 1e-3] # tol, tolb, tolT
# 100: 
# Cb, CT = 1000, 10e-2; 
# etab=0.05, etaT=0.02
# 200: 
# Cb, CT = 600, 2.4e-2
# etab=0.1, etaT=0.1
# 400: 
# Cb, CT = 600, 2e-2*2.0
# etab, etaT = 0.25, 0.08
# 800: 
# Cb, CT = 600, 2e-2*0.2
# etab, etaT = 0.90, 0.60
# 1600: 
# Cb, CT = 600, 2e-2*0.2
# etab, etaT = 4.00, 0.60
# 3200: 
# Cb, CT = 600, 2e-2*0.2; 
# etab, etaT = 15.00, 0.60

Cb, CT = 600, 2e-2*0.2
etab, etaT = 4.00, 0.60
# The list to contain output results
params = {"beta0":beta0.cpu().numpy(), "bTheta0":bTheta0.cpu().numpy(), "tols": tols, "CT":CT, "Cb":Cb }
params["n"] = n
params["m"] = m
params["p"] = p
params["Xtype"] = "Bernoulli"
params["Y|X_type"] = "Normal"

pprint.pprint(params)

#------------------------------------------------------------------------------------
numSimu = 50 
root = Path("./results")
startIdx = 35
for i in range(numSimu):
    # generate the samples
    X = genXBin(n, m, p, prob=prob) 
    Y = genYnorm(X, bTheta0, beta0, sigmaY)
    R = genR(Y, "linear", inp=bs[m], slop=5) # 
    # To find the missing rate. 
    # I control the missing rate around 0.25
    MissRate = R.to_dense().sum()/R.to_dense().numel()
    params["MissRate"] = MissRate
    #print(MissRate)
#----------------------------------------------------------------------------------------------------
    print(f"The {i+1}th/{numSimu}")
    if (i+1) >= startIdx:
        betahat, bThetahat, numI, Berrs, Terrs, betahats, bThetahats, Likelis, etass = NewBern(1000, X, Y, R, conDenfs, TrueParas=TrueParas, Cb=Cb, CT=CT, tols=tols, log=loglv, prob=prob, betainit=betainit, bThetainit=bThetainit, ErrOpts=1, etab=etab, etaT=etaT)
        errb = torch.norm(beta0-betahat)
        errT = torch.norm(bTheta0-bThetahat)
        print(
        f"The {i+1}th/{numSimu},"
        f"The Iteration number is {numI}, "
        f"The error of beta is {errb.item():.3f}, "
        f"The error of bTheta is {errT.item():.3f}."
)

        curResult = {}
        curResult["numI"] = numI
        curResult["Cb"] = Cb
        curResult["CT"] = CT
        curResult["errb"] = errb
        curResult["errT"] = errT 
        curResult["Berrs"] = Berrs
        curResult["Terrs"] = Terrs
        curResult["betahat"] = betahat.cpu().numpy()
        curResult["bThetahat"] = bThetahat.cpu().numpy()

        filName = f"MNAR_linear_p{p}_m{m}_simu{i+1}_iter{numI}.pkl"
        with open(root/filName, "wb") as f:
            pickle.dump([params, curResult], f)
