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
n = 100
m = 100
p = 100
N = 20000

#------------------------------------------------------------------------------------
# The successful probability of each entry of X
prob = 1000/n/m
X = genXdis(n, m, p, type="Bern", prob=prob) 
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, 3, 4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m, rank=5) 
#M = bTheta0 + X.matmul(beta0)
Y = genYlogit(X, bTheta0, beta0)
R = genR(Y, inp=1.4)
# TO find the missing rate, I control the missing rate around 0.25
print(R.sum()/R.numel())
# Compute the D2, H2 
Ds2, Dh2 = Dshlowerflogit(Y, X, beta0, bTheta0)
# Get the lower bound of sigma_bTheta
STbd = 2*Dh2+2*Ds2**2
# generate the samples for MCMC
sXs = genXdis(N, p, type="Bern", prob=prob) 
# The likelihood and its derivatives of Y|X
conDenfs = [fln, fln2, fln22]


#------------------------------------------------------------------------------------
# I use random grid search to find good parameters, so the following are the search spaces 
Cbpool = np.exp(np.linspace(np.log(1), np.log(1e4), 200))
CTpool = Cbpool/10
# ST,  sigma_bTheta, 
STpool = np.exp(np.linspace(np.log(1), np.log(1e4), 100))
#np.random.shuffle(Cbpool)
#np.random.shuffle(CTpool)
#np.random.shuffle(STpool)

#------------------------------------------------------------------------------------
# The number of times to do random grid search
numRG = 100
# eta = 1/(5*0.75*m*p)
# eta, the learning rate of beta
eta = 0.01 
# Termination  tolerance.
tol = 1e-5
TrueParas = [beta0, bTheta0]
# The list to contain output results
results = [{"beta0":beta0.cpu(), "bTheta0":bTheta0.cpu(), "eta":eta, "tol": tol}]
# initial value of beta and bTheta
betainit = beta0* 1.1
bThetainit = bTheta0 * 1.1

#------------------------------------------------------------------------------------
print(results)
# The list to contain training errors 
Errs = []

#------------------------------------------------------------------------------------
# Random grid search
for i in range(numRG):
    #----------------------------------------------------------------------------------------------------
    # To get the Cb, CT and ST from search space
    len1, len2, len3  = len(Cbpool), len(CTpool), len(STpool)
    Cb, CT = 1, 10
    # I have the prior that in general, Cb should be greater than CT
    while Cb < CT:
        idx1, idx2, idx3 = np.random.randint(0, len1), np.random.randint(0, len2), np.random.randint(0, len3)
        Cb, CT, ST = Cbpool[idx1], CTpool[idx2], STpool[idx3]*STbd
    #----------------------------------------------------------------------------------------------------

    print(f"The {i+1}/{numRG}, Cb is {Cb:>8.4g}, CT is {CT:>8.4g}, ST is {ST:>8.4g}")
    # I use try-except statement to avoid error breaking the loop
    try:
        betahat, bThetahat, _, numI, Berrs, Terrs = MCGDBern(1000, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, eta=eta, Cb=Cb, CT=CT, tol=tol, log=0, ST=ST, prob=prob, betainit=betainit, bThetainit=bThetainit, Rbinit=torch.tensor([1.0]), ErrOpts=1)
    except RuntimeError as e:
        results.append((-100, Cb, -100, -100,  CT, -100, -100, ST/STbd))
        Errs.append([])
        print(
            f"The {i+1}th/{numRG},"
            f"Iteration Fails!", 
            e
        )
    else:
        errb = torch.norm(beta0-betahat)
        errT = torch.norm(bTheta0-bThetahat)
        results.append((numI, Cb, errb.item(), betahat.norm().item(), CT, errT.item(), bThetahat.norm().item(), ST/STbd))
        Errs.append([Berrs, Terrs])
        print(
            f"The {i+1}th/{numRG},"
            f"The Iteration number is {numI}, "
            f"The error of beta is {errb.item():.3f}, "
            f"The error of bTheta is {errT.item():.3f}."
        )

# Save the output
f = open("./outputs/RandGrid_Bern_lg_5w_01_r5s5_11_100_tol5_eta2.pkl", "wb")
pickle.dump([results, Errs], f)
f.close()
