from utilities_mar import MarRealDataAlg
import random
import numpy as np
import torch
import pickle
from confs import fln, fln2, fln22, fn, fn2, fn22
import pandas as pd

torch.cuda.set_device(1)
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

def TransYf(Y, Yraw, byrow=False, is_sd=False):
    n, m = Y.shape
    if byrow:
        for j in range(n):
            tmp = Yraw[j, :][Yraw[j, :]!=-1]
            sd = tmp.std()
            Y[j, :] = Y[j, :] - np.mean(tmp)
            if is_sd:
                Y[j, :] = Y[j, :]/sd
    else:
        for i in range(m):
            tmp = Yraw[:, i][Yraw[:, i]!=-1]
            sd = tmp.std()
            Y[:, i] = Y[:, i] - np.mean(tmp)
            if is_sd:
                Y[:, i] = Y[:, i]/sd
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
n, m, p = X.shape
means = np.expand_dims(means, axis=0)
stds = np.expand_dims(stds, axis=0)
X = (X-means)/stds

X = torch.tensor(X)
logist = True

if logist:
    Y = Yraw.copy()
    thre = 3.5
    Y[Yraw>=thre] = 1
    Y[Yraw<thre] = 0
    conDenfs = [fln, fln2, fln22]
else: 
    Y = Yraw.copy()
    #Ym = Y[Yraw!=-1].mean()
    #Ysd = Y[Yraw!=-1].std()
    #Y = (Y - Ym)/Ysd
    Y = TransYf(Y, Yraw)
    conDenfs = [fn, fn2, fn22]

Y = torch.tensor(Y)
    
if logist:
    # initial value of beta and bTheta
    #betainit = torch.rand(p) + 1
    betainit = torch.zeros(p) 
    bThetainit = torch.rand(n, m) + 0.1
    
    # Termination  tolerance.
    tols = [0, 2.5e-6, 5e-3]
    Cb, CT = 40, 4e-3
    etab, etaT = 0.1, 1
else:
    # initial value of beta and bTheta
    betainit = torch.zeros(p) 
    #betainit = torch.randn(p) 
    bThetainit = torch.randn(n, m)
    tols = [0, 2.5e-6, 5e-3]
    Cb, CT = 800, 4e-3*5
    etab, etaT = 0.01, 0.01

#tols = [0, 0, 0]

resdic = {}
numdid = 20
numper = int(n/numdid)
for expidx in range(1, numdid+1):
    idx1, idx2 = (expidx-1)*numper, expidx*numper
    R = Yraw.copy()
    R[Yraw!=-1] = 1
    R[Yraw==-1] = 0
    R = torch.tensor(R)
    R[idx1:idx2,:] = 0
    
    betahat, bThetahat, numI, betahats, bThetahats, Likelis = MarRealDataAlg(5000, X, Y, R, conDenfs, etab=etab, Cb=Cb, CT=CT, tols=tols, log=2, betainit=betainit, bThetainit=bThetainit, ErrOpts=1, etaT=etaT)
    
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
    f = open(f"./MARyelp_log{int(thre*10)}.pkl", "wb")
else:
    f = open(f"./MARyelp_linear.pkl", "wb")
pickle.dump(resdic, f)
f.close()
