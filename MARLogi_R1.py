from utils import *
from utils_mar import MarNewBern
import random
import numpy as np
import torch
import pickle
import timeit
import time
import pprint
from pathlib import Path
from confs import fln, fln2, fln22

cudaid = 2
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
n = m = 800
p = 100

# Termination  tolerance.
loglv = 0
tols = [2e-6, 0, 0]
if m == 100 and p == 50:
    Cb, CT = 600, 10e-2
    etab, etaT = 0.08, 0.08
elif m == 100 and p == 100:
    Cb, CT = 600, 10e-2
    etab, etaT = 0.10, 0.09
elif m == 200 and p == 50:
    Cb, CT = 800, 8e-2
    etab, etaT = 0.20, 0.10
elif m == 200 and p == 100:
    Cb, CT = 800, 8e-2
    etab, etaT = 0.25, 0.105
elif m == 400 and p == 50:
    Cb, CT = 800, 8.0e-2
    etab, etaT = 0.50, 0.10
elif m == 400 and p == 100:
    Cb, CT = 800, 8.0e-2
    etab, etaT = 0.54, 0.095
elif m == 800 and p == 50:
    Cb, CT = 800, 8.0e-2
    etab, etaT = 1.5, 0.10
elif m == 800 and p == 100:
    Cb, CT = 800, 8.0e-2
    etab, etaT = 1.9, 0.097
elif m == 1600 and p == 50:
    Cb, CT = 800, 8.0e-2
    etab, etaT = 7.00, 0.10
elif m == 1600 and p == 100:
    Cb, CT = 800, 8.0e-2
    etab, etaT = 7.50, 0.105


initbetapref = 1 + (torch.rand(p)-1/2)/4  #[0.75, 1.25]
initthetapref = 1 + (torch.rand(n, m)-1/2)/4
#------------------------------------------------------------------------------------
# The successful probability of each entry of X
prob =  1000/n/m

# generate the parameters
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7))) *10.
bTheta0 = genbTheta(n, m, sigVs=np.array([10, 9, 8, 7, 6])/2) 
TrueParas = [beta0, bTheta0]
# initial value of beta and bTheta
betainit = beta0 * initbetapref
#betainit = beta0
bThetainit = bTheta0 * initthetapref
#bThetainit = bTheta0
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
params["MissRate"] = []

pprint.pprint(params)

#------------------------------------------------------------------------------------
# The list to contain training errors 

numSimu = 10
root = Path("./results")
startIdx = 1

#------------------------------------------------------------------------------------
for i in range(numSimu):
    # generate the samples
    X = genXBin(n, m, p, prob=prob) 
    Y = genYlogit(X.to_dense(), bTheta0, beta0)
    R = genR(Y, "fixed") # 
    X = X.to_dense()
    R = R.to_dense()
    # I control the missing rate around 0.25
    MissRate = R.sum()/R.numel()
    print(MissRate)
    params["MissRate"] = MissRate

    #----------------------------------------------------------------------------------------------------
    print(f"The {i+1}th/{numSimu}")
    if (i+1) >= startIdx:
        betahat, bThetahat, numI, Berrs, Terrs, betahats, bThetahats, Likelis = MarNewBern(1000, X, Y, R, conDenfs, TrueParas=TrueParas, etab=etab, Cb=Cb, CT=CT, tols=tols, log=loglv, prob=prob, betainit=betainit, bThetainit=bThetainit, ErrOpts=1, etaT=etaT)
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
        curResult["X"] = X.cpu().numpy()
        curResult["Y"] = Y.cpu().numpy()
        curResult["R"] = R.cpu().numpy()
        curResult["errb"] = errb.cpu().numpy()
        curResult["errT"] = errT.cpu().numpy()
        curResult["Berrs"] = Berrs
        curResult["Terrs"] = Terrs
        curResult["beta0"] = beta0.cpu().numpy()
        curResult["bTheta0"] = bTheta0.cpu().numpy()
        curResult["betahat"] = betahat.cpu().numpy()
        curResult["bThetahat"] = bThetahat.cpu().numpy()

        filName = f"MAR_logi_p{p}_m{m}_simu{i+1}_iter{numI}.pkl"
        with open(root/filName, "wb") as f:
            pickle.dump([params, curResult], f)
