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
from confs import fln, fln2, fln22

cudaid = 2
loglv = 2
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
n = m = 200
p = 50

# Termination  tolerance.
tols = [2e-6, 0, 0] # [0.5, 1.5]
#tols = [0, 5e-3, 1e-3] # tol, tolb, tolT

if m == 100 and p == 50:
    Cb, CT = 1000, 30e-2
    etab, etaT = 0.04, 0.02
elif m == 100 and p == 100:
    Cb, CT = 1000, 30e-2
    etab, etaT = 0.05, 0.022
elif m == 200 and p == 50:
    Cb, CT = 600, 6e-2
    etab, etaT = 0.15, 0.1
elif m == 200 and p == 100:
    Cb, CT = 600, 6e-2
    etab, etaT = 0.18, 0.095
elif m == 400 and p == 50:
    Cb, CT = 600, 3e-2*2.0
    etab, etaT = 0.50, 0.10
elif m == 400 and p == 100:
    Cb, CT = 600, 3e-2*2.0
    etab, etaT = 0.65, 0.098
elif m == 800 and p == 50:
    Cb, CT = 600, 3e-2*1.0
    etab, etaT = 2.30, 0.15
elif m == 800 and p == 100:
    Cb, CT = 600, 3e-2*1.0
    etab, etaT = 2.50, 0.148
elif m == 1600 and p == 50:
    Cb, CT = 600, 2e-2*2.0
    etab, etaT = 9.00, 0.10
elif m == 1600 and p == 100:
    Cb, CT = 600, 2e-2*2.0
    etab, etaT = 10.00, 0.095


initbetapref = 1 + (torch.rand(p)-1/2)/4  #[0.875, 1.125]
initthetapref = 1 + (torch.rand(n, m)-1/2)/4
#------------------------------------------------------------------------------------
# The successful probability of each entry of X
prob = 1000/n/m # 0.05

# generate the parameters
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7))) * 10
bTheta0 = genbTheta(n, m, sigVs=np.array([10, 9, 8, 7, 6])/2) 
TrueParas = [beta0, bTheta0]
# initial value of beta and bTheta
betainit = beta0 * initbetapref
bThetainit = bTheta0 * initthetapref
# The likelihood and its derivatives of Y|X
conDenfs = [fln, fln2, fln22]



#------------------------------------------------------------------------------------

# The list to contain output results
params = {"beta0":beta0.cpu().numpy(), "bTheta0":bTheta0.cpu().numpy(), "tols": tols, "CT":CT, "Cb":Cb }
params["n"] = n
params["m"] = m
params["p"] = p
params["Xtype"] = "Bernoulli"
params["Y|X_type"] = "logi"

pprint.pprint(params)

#------------------------------------------------------------------------------------
numSimu = 10
root = Path("./results")
startIdx = 1
for i in range(numSimu):
    # generate the samples
    X = genXBin(n, m, p, prob=prob) 
    Y = genYlogit(X.to_dense(), bTheta0, beta0)
    R = genR(Y, "fixed") # 
    # To find the missing rate. 
    MissRate = 1 - R.to_dense().mean()
    params["MissRate"] = MissRate
    print(MissRate)
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
        f"The error of bTheta is {errT.item():.3f}.")

        curResult = {}
        curResult["numI"] = numI
        # curResult["X"] = X.to_dense().cpu().numpy()
        # curResult["Y"] = Y.cpu().numpy()
        # curResult["R"] = R.to_dense().cpu().numpy()
        curResult["CT"] = CT
        curResult["errb"] = errb.cpu().numpy()
        curResult["errT"] = errT.cpu().numpy()
        curResult["Berrs"] = Berrs
        curResult["Terrs"] = Terrs
        curResult["beta0"] = beta0.cpu().numpy()
        curResult["bTheta0"] = bTheta0.cpu().numpy()
        curResult["betahat"] = betahat.cpu().numpy()
        curResult["bThetahat"] = bThetahat.cpu().numpy()

        filName = f"MNAR_logi_p{p}_m{m}_simu{i+1}_iter{numI}_emp.pkl"
        with open(root/filName, "wb") as f:
            pickle.dump([params, curResult], f)
