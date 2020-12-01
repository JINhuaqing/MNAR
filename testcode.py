from utils import *
import random
import numpy as np
import torch
import pickle
import timeit
import time
import argparse
import pprint
from confs import fn, fn2, fn22

cudaid = 0
loglv = 2
bs = {
        100:1.4,
        125:1.3,
        150:1.25,
        175:1.2,
        200:1.18,
        212:1.17,
        225:1.16,
        238:1.16,
        250:1.16,
        263:1.16,
        275:1.16,
        287:1.14,
        300:1.1,
        1000: 1
     }

def Cbsf(m):
    if m <= 200:
        return 800
    else:
        return 800

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
n = 1000
m = 1000
p = 100

initbetapref = 1 + (torch.rand(p)-1/2)/4  #[0.75, 1.25]
initthetapref = 1 + (torch.rand(n, m)-1/2)/4
#------------------------------------------------------------------------------------
# The successful probability of each entry of X
prob = 0.05 # 1000/n/m
sigmaY = 0.1

# generate the parameters
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m, sigVs=np.array([10, 9, 8, 7, 6])*100/np.sqrt(m*n)) 
TrueParas = [beta0, bTheta0]
# initial value of beta and bTheta
betainit = beta0 * initbetapref
bThetainit = bTheta0 * initthetapref
# The likelihood and its derivatives of Y|X
conDenfs = [fn, fn2, fn22]



#------------------------------------------------------------------------------------
# Termination  tolerance.
tols = [2.7e-14, 2.65e-9, 1.9e-9] # [0.5, 1.5]
tols = [0, 1e-5, 5e-4]
Cb, CT = 1600, 2e-0
# The list to contain output results
params = {"beta0":beta0.cpu().numpy(), "bTheta0":bTheta0.cpu().numpy(), "tols": tols, "CT":CT, "Cb":Cb }
params["n"] = n
params["m"] = m
params["p"] = p
params["Xtype"] = "Bernoulli"
params["Y|X_type"] = "Normal"
params["MissRate"] = []

pprint.pprint(params)

#------------------------------------------------------------------------------------
# The list to contain training errors 
Errs = []
EstParas = []
results = []

#------------------------------------------------------------------------------------
# generate the samples
X = genXBin(n, m, p, prob=prob) 
Y = genYnorm(X, bTheta0, beta0, sigmaY)
R = genR(Y, "linear", inp=bs[m]) # 
# To find the missing rate. 
# I control the missing rate around 0.25
MissRate = R.to_dense().sum()/R.to_dense().numel()
#print(MissRate)
#----------------------------------------------------------------------------------------------------
betahat, bThetahat, numI, Berrs, Terrs, betahats, bThetahats, Likelis, etass = NewBern(50000, X, Y, R, conDenfs, TrueParas=TrueParas, Cb=Cb, CT=CT, tols=tols, log=loglv, prob=prob, betainit=betainit, bThetainit=bThetainit, ErrOpts=1)
errb = torch.norm(beta0-betahat)
errT = torch.norm(bTheta0-bThetahat)
print(
    f"The Iteration number is {numI}, "
    f"The error of beta is {errb.item():.3f}, "
    f"The error of bTheta is {errT.item():.3f}."
)


