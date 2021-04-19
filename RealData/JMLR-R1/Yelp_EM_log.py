from utils import YelpMissing
from utils_EM import EMRealDataAlg
import random
import numpy as np
import torch
import pickle
from confs import fln, fln2, fln22, fln2_raw
import pandas as pd


torch.cuda.set_device(2)
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

# +
# Load real data
Yp = "./Ymat.pkl"
Xp = "./Xmat.pkl"

with open(Yp, "rb") as f:
    Yraw = pickle.load(f)
with open(Xp, "rb") as f:
    X = pickle.load(f)
# -

# #### Normalized the X mat

means = X.mean(axis=(0, 1))
stds = X.std(axis=(0, 1))
X = (X-means)/stds
X = torch.tensor(X)

# #### Dichotomize the Y

Y = Yraw.copy()
thre = 3.5
Y[Yraw>=thre] = 1
Y[Yraw<thre] = 0
Y = torch.tensor(Y)

# The likelihood and its derivatives of Y|X

conDenfs = [fln, fln2_raw, fln22]


# ## CV for tuning

# Termination  tolerance.
tols = [0, 0, 0]
etab, etaT = 1.0, 0.2
CbTs = [(20, 8e-3), (40, 4e-3), (100, 8e-3), (10, 8e-4), (80, 4e-3)]

# initial value of beta and bTheta
#betainit = torch.rand(p) + 1
n, m, p = X.shape
betainit = torch.zeros(p) 
bThetainit = torch.rand(n, m) + 0.1

OR = 0.08
R = YelpMissing(Yraw, OR=OR)
R = torch.tensor(R)

CVres = []
for CbT in CbTs:
    Cb, CT = CbT
    inpY = Y.clone()
    betahat, bThetahat, numI, betahats, bThetahats, Likelis = EMRealDataAlg(250, X, inpY, R, conDenfs, etab=etab, etaT=etaT, 
                                                                             Cb=Cb, CT=CT, tols=tols, log=0, 
                                                                             betainit=betainit, bThetainit=bThetainit, ErrOpts=1)
    
    print( f"The Iteration number is {numI}." )
    
    Yrawt = torch.tensor(Yraw)
    betaX = torch.matmul(X, betahat)
    TbX = bThetahat + betaX
    hatprobs = torch.exp(TbX) / (1+torch.exp(TbX))
    mask = (R == 0) & (Yrawt != -1)
    estprobs = hatprobs[mask]
    gtY = Y[mask]
    CVres.append([estprobs.cpu().numpy(), gtY.cpu().numpy()])

# #### The results

from sklearn import metrics

# +
aucs = []
for i in range(len(CVres)):
    auc = metrics.roc_auc_score(CVres[i][1], CVres[i][0])
    aucs.append(auc)
    
CbT = CbTs[np.argmax(aucs)]
# -

# ### Real data applcation given tuning parameters

# Termination  tolerance.
tols = tols
etab, etaT = etab, etaT
Cb, CT = CbT


# +
resdic = []
paras = []
OR = OR # [0.08, 0.065, 0.05]
for expidx in range(1, 11):
    #idx1, idx2 = (expidx-1)*5, expidx*5
    #R = Yraw.copy()

    n, m, p = X.shape
    betainit = torch.zeros(p) 
    bThetainit = torch.rand(n, m) + 0.1
    
    R = YelpMissing(Yraw, OR=OR)
    R = torch.tensor(R)
    
    inpY = Y.clone()

    betahat, bThetahat, numI, betahats, bThetahats, Likelis = EMRealDataAlg(250, X, inpY, R, conDenfs, etab=etab, etaT=etaT,
                                                                          Cb=Cb, CT=CT, tols=tols, log=0, betainit=betainit, 
                                                                          bThetainit=bThetainit, ErrOpts=1)
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
    resdic.append([estprobs.cpu().numpy(), gtY.cpu().numpy()])
    paras.append({"beta":betahat, "theta":bThetahat})
    
    
f = open(f"./EMyelp_log_{int(1000*OR)}.pkl", "wb")
pickle.dump(resdic, f)
f.close()
# -



