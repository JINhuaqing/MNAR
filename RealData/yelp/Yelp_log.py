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

def TransYf(Y, Yraw, byrow=False, is_sd=False):
    n, m = Y.shape
    if byrow:
        for j in range(n):
            tmp = Y[j, :][Yraw[j, :]!=-1]
            sd = tmp.std()
            Y[j, :] = Y[j, :] - np.mean(tmp)
            if is_sd:
                Y[j, :] = Y[j, :]/sd
    else:
        for i in range(m):
            tmp = Y[:, i][Yraw[:, i]!=-1]
            sd = tmp.std()
            Y[:, i] = Y[:, i] - np.mean(tmp)
            if is_sd:
                Y[:, i] = Y[:, i]/sd
    return Y

def TransYlogf(Y, Yraw, byrow=False):
    n, m = Y.shape
    if byrow:
        for j in range(n):
            tmp = Yraw[j, :][Yraw[j, :]!=-1]
            Y[j, :] = np.log(Y[j, :] / np.sum(tmp))
    else:
        for i in range(m):
            tmp = Yraw[:, i][Yraw[:, i]!=-1]
            Y[:, i] = np.log(Y[:, i]/np.sum(tmp))
    Y[Yraw==-1] = -1
    Y = TransYf(Y, Yraw, byrow, True)
    return Y
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
N = 20000
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
logist = False
if logist:
    Y = Yraw.copy()
    thre = 4.5
    Y[Yraw>=thre] = 1
    Y[Yraw<thre] = 0
    # The likelihood and its derivatives of Y|X
    conDenfs = [fln, fln2, fln22]
else: 
    Y = Yraw.copy()
    #Ym = Y[Yraw!=-1].mean()
    #Ysd = Y[Yraw!=-1].std()
    #Y = (Y - Ym)/Ysd
    Y = TransYlogf(Y, Yraw, True)
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
    #tols = [0, 1.6e-5, 1e-2] # thre = 3.5
    #tols = [0, 5e-6, 7e-3]
    tols = [0, 1e-5, 5e-3]
else:
    #betainit = torch.ones(p) 
    #betainit = torch.randn(p) + 1
    betainit = torch.zeros(p) 
    bThetainit = torch.randn(n, m)
    Cb, CT = 800*4, 2e-2
    etab, etaT = 0.00001, 1
    tols = [0, 5e-6, 7e-3]
tols = [0, 0, 0]
#----------------------------------------------------------------------------------------------------

resdic = {}
for expidx in range(1, 21):
    idx1, idx2 = (expidx-1)*5, expidx*5
    R = Yraw.copy()
    R[Yraw!=-1] = 1
    R[Yraw==-1] = 0
    R = torch.tensor(R)
    R[idx1:idx2,:] = 0

    betahat, bThetahat, numI, betahats, bThetahats, Likelis = RealDataAlg(300000, X, Y, R, sXs, conDenfs, etab=etab, Cb=Cb, CT=CT, tols=tols, log=2, betainit=betainit, bThetainit=bThetainit, ErrOpts=1, etaT=etaT)
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
