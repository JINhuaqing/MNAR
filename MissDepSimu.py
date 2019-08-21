from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit
import time
import pprint
from confs import fln, fln2, fln22


torch.cuda.set_device(3)
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
n = 300
m = 300
p = 100
N = 20000

#------------------------------------------------------------------------------------
# The successful probability of each entry of X
prob = 1000/n/m
# generate the parameters
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m, rank=5) * 8
TrueParas = [beta0, bTheta0]
# The likelihood and its derivatives of Y|X
conDenfs = [fln, fln2, fln22]



#------------------------------------------------------------------------------------
# The number of times to do random grid search
numSimu = 50
# eta = 1/(5*0.75*m*p)
# eta, the learning rate of beta
etabs = [1e-1, 5e-1]
etaTs = [1e-1, 1e-2]
etabsc = [400]
etaTsc = [180]
# Termination  tolerance.
tol = 4e-5
Cb, CT = 10, 2e-3
# The list to contain output results
params = {"beta0":beta0.cpu().numpy(), "bTheta0":bTheta0.cpu().numpy(), "tol": tol, "CT":CT, "Cb":Cb }
params["n"] = n
params["m"] = m
params["p"] = p
params["N"] = N
params["Xtype"] = "Bernoulli"
params["Y|X_type"] = "logistic"
params["etaTs"] =  etaTs
params["etaTsc"] =  etaTsc
params["etabs"] =  etabs
params["etabsc"] =  etabsc
params["numSimu"] = numSimu
params["MissRate"] = []

pprint.pprint(params)

#------------------------------------------------------------------------------------
# The list to contain training errors 
Errs = []
results = []

#------------------------------------------------------------------------------------
for i in range(numSimu):
    # generate the samples
    X = genXdis(n, m, p, type="Bern", prob=prob) 
    Y = genYlogit(X, bTheta0, beta0)
    R = genR(Y, inp=1.3)
    # TO find the missing rate, I control the missing rate around 0.25
    MissRate = R.sum()/R.numel()
    params["MissRate"].append(MissRate.item())
    # generate the samples for MCMC
    sXs = genXdis(N, p, type="Bern", prob=prob) 
    # initial value of beta and bTheta
    betainit = beta0 * (1 + (torch.rand(p)-1/2))
    bThetainit = bTheta0 * (1 + (torch.rand(n,m)-1/2))
    #----------------------------------------------------------------------------------------------------
    # I use try-except statement to avoid error breaking the loop
    try:
        betahat, bThetahat, numI, Berrs, Terrs, betahats, bThetahats, Likelis = NewBern(2000, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, etabs=etabs, etabsc=etabsc, Cb=Cb, CT=CT, tol=tol, log=0, prob=prob, betainit=betainit, bThetainit=bThetainit, ErrOpts=1, etaTs=etaTs, etaTsc=etaTsc)
    except RuntimeError as e:
        results.append((-100, Cb, -100, -100,  CT, -100, -100))
        Errs.append([])
        print(
            f"The {i+1}th/{numSimu},"
            f"Iteration Fails!", 
            e
        )
    else:
        errb = torch.norm(beta0-betahat)
        errT = torch.norm(bTheta0-bThetahat)
        results.append((numI, Cb, errb.item(), betahat.norm().item(), CT, errT.item(), bThetahat.norm().item()))
        Errs.append([Berrs, Terrs, betahats, bThetahats, Likelis])
        print(
            f"The {i+1}th/{numSimu},"
            f"The Iteration number is {numI}, "
            f"The error of beta is {errb.item():.3f}, "
            f"The error of bTheta is {errT.item():.3f}."
        )

# Save the output
f = open(f"./outputs/Simulation_demo{m}.pkl", "wb")
pickle.dump([params, results, Errs], f)
f.close()
