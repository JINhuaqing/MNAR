from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit
import time
import argparse
import pprint
from confs import fln, fln2, fln22
from scipy.optimize import minimize

m = 100
n = 100
p = 3
cudaid = 0

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
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
ddtype = torch.float64

#------------------------------------------------------------------------------------
# Set the number of n, m, p, N
# N is number of samples used for MCMC
n = n
m = m 
p = p
N = 20000

initbetapref = 1 + (torch.rand(p)-1/2)  #[0.75, 1.25]
#------------------------------------------------------------------------------------
# The successful probability of each entry of X
prob = 0.05 
# generate the parameters
# beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7))) # orginal beta0
beta0 = torch.tensor([1.0, 2, 3])
bTheta0 = genbTheta(n, m, rank=5) * 0
TrueParas = [beta0, bTheta0]
betainit = beta0 * initbetapref
conDenfs = [fln, fln2, fln22]
X = genXdis(n, m, p, type="Bern", prob=prob) 
Y = genYlogit(X, bTheta0, beta0)
R = genR(Y, "fixed")
sXs = genXdis(N, p, type="Bern", prob=prob) 

# def ObjFun(beta):
#     betaTS = torch.tensor(beta, dtype=ddtype)
#     print(beta)
#     fv = LBern(bTheta0, betaTS, fln, X, Y, R, prob)
#     return fv.cpu().numpy() 
# 
# def ObjFunD1(beta):
#     betaTS = torch.tensor(beta, dtype=ddtype)
#     fv = LpbBern(bTheta0, betaTS, conDenfs, X, Y, R, prob)
#     return fv.cpu().numpy()
# 
# res = minimize(ObjFun, 
#             x0=np.array([0, 2, 1]), 
#             #jac=ObjFunD1, 
#             method="L-BFGS-B",
#             options={"iprint":-1})
# print(res)

