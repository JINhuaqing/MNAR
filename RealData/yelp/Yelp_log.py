from utilities import RealDataAlg
import random
import numpy as np
import torch
import pickle
from confs import fln, fln2, fln22, fn, fn2, fn22
import pandas as pd


torch.cuda.set_device(0)
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

#------------------------------------------------------------------------------------
# Load real data
Yp = "./Ymat.pkl"
Xp = "./Xmat.pkl"
Xspsp = "./Xsps.pkl"

with open(Yp, "rb") as f:
    Yraw = pickle.load(f)
with open(Xp, "rb") as f:
    X = pickle.load(f)
with open(Xspsp, "rb") as f:
    sXsall = pickle.load(f)

dfsXs = pd.DataFrame(sXsall)
means = np.array(dfsXs.mean())
stds = np.array(dfsXs.std())
means = np.expand_dims(means, axis=0)
stds = np.expand_dims(stds, axis=0)
#------------------------------------------------------------------------------------
# N is number of samples used for MCMC
n, m, p = X.shape
N = 30000
Nmax, _ = sXsall.shape
assert Nmax >= N
sXs = sXsall[:N, :] 
sXs = (sXs-means)/stds
sXs = sXs.transpose(1, 0)

means = np.expand_dims(means, axis=0)
stds = np.expand_dims(stds, axis=0)
X = (X-means)/stds

X = torch.tensor(X)
sXs = torch.tensor(sXs)
logist = True
if logist:
    Y = Yraw.copy()
    thre = 3.5
    Y[Yraw>=thre] = 1
    Y[Yraw<thre] = 0
    # The likelihood and its derivatives of Y|X
    conDenfs = [fln, fln2, fln22]
else: 
    Y = Yraw.copy()
    Ym = Y[Yraw!=-1].mean()
    Ysd = Y[Yraw!=-1].std()
    Y = (Y - Ym)/Ysd
    conDenfs = [fn, fn2, fn22]
    
Y = torch.tensor(Y)


#------------------------------------------------------------------------------------
# initial value of beta and bTheta
if logist:
    betainit = torch.zeros(p) 
    #betainit = torch.ones(p) 
    #betainit = torch.rand(p) + 1
    bThetainit = torch.rand(n, m) + 0.1
    Cb, CT = 60, 4e-3 
    etab, etaT = 0.1, 1 * 5 
    tols = [0, 1.6e-5, 1e-2]
    #tols = [0, 5e-6, 7e-3]
else:
    #betainit = torch.ones(p) 
    #betainit = torch.randn(p) + 1
    betainit = torch.zeros(p) 
    bThetainit = torch.rand(n, m) + 0.1
    Cb, CT = 800, 2e-2*5
    etab, etaT = 0.05, 0.05*100
    tols = [0, 5e-6, 7e-3]
#tols = [0, 0, 0]
#----------------------------------------------------------------------------------------------------

resdic = {}
for expidx in range(1, 21):
    idx1, idx2 = (expidx-1)*5, expidx*5
    R = Yraw.copy()
    R[Yraw!=-1] = 1
    R[Yraw==-1] = 0
    R = torch.tensor(R)
    R[idx1:idx2,:] = 0

    betahat, bThetahat, numI, betahats, bThetahats, Likelis = RealDataAlg(3000, X, Y, R, sXs, conDenfs, etab=etab, Cb=Cb, CT=CT, tols=tols, log=2, betainit=betainit, bThetainit=bThetainit, ErrOpts=1, etaT=etaT)
    print(
    f"Now it is {expidx}/10, "
    f"The Iteration number is {numI}."
)

    Yrawt = torch.tensor(Yraw)
    betaX = torch.matmul(X, betahat)
    TbX = bThetahat + betaX
    hatprobs = torch.exp(TbX) / (1+torch.exp(TbX))
    mask = (R == 0) & (Yrawt != -1)
    estprobs = hatprobs[mask]
    gtY = Y[mask]
    resdic[expidx] = [estprobs, gtY]
    

# Save the output
if logist:
    f = open(f"./MNARyelp_log{int(thre*10)}.pkl", "wb")
else:
    f = open(f"./MNARyelp_linear.pkl", "wb")
pickle.dump(resdic, f)
f.close()
