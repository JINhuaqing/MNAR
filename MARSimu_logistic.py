from utilities import *
from utilities_mar import MarNewBern
import random
import numpy as np
import torch
import pickle
import timeit
import time
import argparse
import pprint
from confs import fln, fln2, fln22

parser = argparse.ArgumentParser(description = "This script is to run demo simulation for NMAR project")
parser.add_argument('-m', type=int, default=100, help = "Parameter m")
parser.add_argument('-n', type=int, default=100, help = "Parameter n")
parser.add_argument('-p', type=int, default=100, help = "Parameter p")
parser.add_argument('-c', '--cuda', type=int, default=0, help = "GPU number")
parser.add_argument('-num', '--numSimu', type=int, default=10, help = "number of simulation")
parser.add_argument('-log', '--logoutput', type=int, default=2, help = "the log level of the function")
args = parser.parse_args()
#cudaid = args.cuda
m = args.m
n = args.n
p = args.p
cudaid = args.cuda
numSimu = args.numSimu
loglv = args.logoutput
def Cbsf(m):
    if m < 200:
        return 30
    elif m == 200:
        return 45
    else:
        return 60
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
# Set the number of n, m, p, N
# N is number of samples used for MCMC
n = n
m = m 
p = p
N = 10000

initbetapref = 1 + (torch.rand(p)-1/2)/4  #[-0.75, 1.25]
initthetapref = 1 + (torch.rand(n, m)-1/2)/4
prefix = n*m/10000
#------------------------------------------------------------------------------------
# The successful probability of each entry of X
prob = 0.05 # 1000/n/m
# generate the parameters
# beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7))) # orginal beta0
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m, sigVs=np.array([10, 9, 8, 7, 6])*100/np.sqrt(m*n)) 
TrueParas = [beta0, bTheta0]
# initial value of beta and bTheta
betainit = beta0 * initbetapref
bThetainit = bTheta0 * initthetapref
# The likelihood and its derivatives of Y|X
conDenfs = [fln, fln2, fln22]



#------------------------------------------------------------------------------------
# The number of times to do random grid search
numSimu = numSimu 
# eta = 1/(5*0.75*m*p)
# Termination  tolerance.
tols = [0, 5e-6, 1e-4]
Cb, CT = Cbsf(m), 2e-3
etab, etaT = 0.5, 0.05
# The list to contain output results
params = {"beta0":beta0.cpu().numpy(), "bTheta0":bTheta0.cpu().numpy(), "tols": tols, "CT":CT, "Cb":Cb }
params["n"] = n
params["m"] = m
params["p"] = p
params["N"] = N
params["Xtype"] = "Bernoulli"
params["Y|X_type"] = "logistic"
params["numSimu"] = numSimu
params["MissRate"] = []

pprint.pprint(params)

#------------------------------------------------------------------------------------
# The list to contain training errors 
Errs = []
EstParas = []
results = []

#------------------------------------------------------------------------------------
for i in range(numSimu):
    # generate the samples
    X = genXdis(n, m, p, type="Bern", prob=prob) 
    Y = genYlogit(X, bTheta0, beta0)
    R = genR(Y, "fixed")
    # TO find the missing rate, I control the missing rate around 0.25
    MissRate = R.sum()/R.numel()
    params["MissRate"].append(MissRate.item())
    # generate the samples for MCMC
    #----------------------------------------------------------------------------------------------------
    # I use try-except statement to avoid error breaking the loop
    try:
        betahat, bThetahat, numI, Berrs, Terrs, betahats, bThetahats, Likelis = MarNewBern(100000, X, Y, R, conDenfs, TrueParas=TrueParas, etab=etab, Cb=Cb, CT=CT, tols=tols, log=loglv, prob=prob, betainit=betainit, bThetainit=bThetainit, ErrOpts=1, etaT=etaT)
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
        EstParas.append([betahat.cpu().numpy(), bThetahat.cpu().numpy()])
        print(
            f"The {i+1}th/{numSimu},"
            f"The Iteration number is {numI}, "
            f"The error of beta is {errb.item():.3f}, "
            f"The error of bTheta is {errT.item():.3f}."
        )

# Save the output
f = open(f"./outputs/MARSimulationLogistic_p{p}_{m}.pkl", "wb")
pickle.dump([params, results, Errs, EstParas], f)
f.close()

