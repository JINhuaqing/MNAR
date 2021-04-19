from utils import RealDataAlg, YelpMissing
from utils_mar import MarRealDataAlg
from utils_EM import EMRealDataAlg
import random
import numpy as np
import torch
import pickle
from sklearn import metrics
from confs import fln, fln2, fln22, fln2_raw
import pandas as pd


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

## Save the X and Y in txt form
#Ynp = Y.cpu().numpy()
#Xnp = X.cpu().numpy()
#with open("./npData/Y.npz", "wb") as f:
#    np.save(f, Ynp)
#with open("./npData/X.npz", "wb") as f:
#    np.save(f, Xnp)

# observed rate
OR = 0.07 # 0.05, 0.06, 0.07, 0.08

# The likelihood and its derivatives of Y|X

conDenfs = [fln, fln2, fln22]
conDenfsEM = [fln, fln2_raw, fln22]


# ### Tuning parameters 

# Termination  tolerance.
tols = [0, 0, 0]
etab, etaT = 1.0, 0.2
etabM, etaTM = 2.0, 0.4
CbTs = [(20, 8e-3), (40, 4e-3), (100, 8e-3), (10, 8e-4), (80, 4e-3)]

n, m, p = X.shape
R = YelpMissing(Yraw, OR=OR)
R = torch.tensor(R)

# #### EM

EMCVres = []
betainit = torch.zeros(p) 
bThetainit = torch.rand(n, m) + 0.1
for CbT in CbTs:
    Cb, CT = CbT
    inpY = Y.clone()
    betahat, bThetahat, numI, betahats, bThetahats, Likelis = EMRealDataAlg(250, X, inpY, R, conDenfsEM, etab=etab, etaT=etaT, 
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
    EMCVres.append([estprobs.cpu().numpy(), gtY.cpu().numpy()])

# #### MNAR and MAR

MARCVres = []
MNARCVres = []
betainit = torch.zeros(p) 
bThetainit = torch.rand(n, m) + 0.1
for CbT in CbTs:
    Cb, CT = CbT
    marbetahat, marbThetahat, numI, betahats, bThetahats, Likelis = MarRealDataAlg(250, X, Y, R, conDenfs, etab=etab, etaT=etaT, 
                                                                             Cb=Cb, CT=CT, tols=tols, log=0, 
                                                                             betainit=betainit, bThetainit=bThetainit, ErrOpts=1)
    
    
    # marbetahat = torch.zeros(p) 
    # marbThetahat= torch.rand(n, m) + 0.1 
    sR = R.to_sparse()
    betahat, bThetahat, numI, betahats, bThetahats, Likelis = RealDataAlg(100, X, Y, sR, conDenfs, etab=etabM, etaT=etaTM,
                                                                          Cb=Cb, CT=CT, tols=tols, log=0, betainit=marbetahat, 
                                                                          bThetainit=marbThetahat, ErrOpts=1)

    Yrawt = torch.tensor(Yraw)
    betaX = torch.matmul(X, betahat)
    TbX = bThetahat + betaX

    marbetaX= torch.matmul(X, marbetahat)
    marTbX = marbThetahat + marbetaX
    
    hatprobs = torch.exp(TbX) / (1+torch.exp(TbX))
    marhatprobs = torch.exp(marTbX) / (1+torch.exp(marTbX))
    
    
    mask = (R == 0) & (Yrawt != -1)
    estprobs = hatprobs[mask]
    marestprobs = marhatprobs[mask]
    gtY = Y[mask]
    
    MARCVres.append([marestprobs.cpu().numpy(), gtY.cpu().numpy()])
    MNARCVres.append([estprobs.cpu().numpy(), gtY.cpu().numpy()])


def res2AUCs(res):
    aucs = []
    for i in range(len(res)):
        auc = metrics.roc_auc_score(res[i][1], res[i][0])
        aucs.append(auc)
    return aucs


aucs = {}
aucs["EM"] = res2AUCs(EMCVres)
aucs["MAR"] = res2AUCs(MARCVres)
aucs["MNAR"] = res2AUCs(MNARCVres)

emCb, emCT = CbTs[np.argmax(aucs["EM"])]
marCb, marCT = CbTs[np.argmax(aucs["MAR"])]
Cb, CT = CbTs[np.argmax(aucs["MNAR"])]

# ### ---

# #### initial value of beta and bTheta and tuning parameters

etab, etaT = 1, 0.2
tols = [0, 0, 0]

mnarres = []
marres = []
emres = []
for expidx in range(1, 21):
    R = YelpMissing(Yraw, OR=OR)
    R = torch.tensor(R)
    Rnp = R.cpu().numpy()
    with open(f"./npData/R_{int(1000*OR)}_{expidx}.npz", "wb") as f:
        np.save(f, Rnp)

    print(
    f"Now it is {expidx}/20, " )
    
    # EM 
    betainit = torch.zeros(p) 
    bThetainit = torch.rand(n, m) + 0.1
    inpY = Y.clone()
    betahat, bThetahat, numI, betahats, bThetahats, Likelis = EMRealDataAlg(250, X, inpY, R, conDenfsEM, etab=etab, etaT=etaT, 
                                                                             Cb=emCb, CT=emCT, tols=tols, log=0, 
                                                                             betainit=betainit, bThetainit=bThetainit, ErrOpts=1)
    
    Yrawt = torch.tensor(Yraw)
    betaX = torch.matmul(X, betahat)
    TbX = bThetahat + betaX
    hatprobs = torch.exp(TbX) / (1+torch.exp(TbX))
    mask = (R == 0) & (Yrawt != -1)
    estprobs = hatprobs[mask]
    gtY = Y[mask]
    emres.append([estprobs.cpu().numpy(), gtY.cpu().numpy()])
    
    # MAR
    betainit = torch.zeros(p) 
    bThetainit = torch.rand(n, m) + 0.1
    marbetahat, marbThetahat, _, marbetahats, marbThetahats, marLikelis = MarRealDataAlg(250, X, Y, R, conDenfs, 
                                                                                         etab=etab, etaT=etaT, 
                                                                                         Cb=marCb, CT=marCT, tols=tols,  log=0, 
                                                                                         betainit=betainit, bThetainit=bThetainit, 
                                                                                         ErrOpts=1)
    # marbetahat = torch.zeros(p) 
    # marbThetahat= torch.rand(n, m) + 0.1 
    sR = R.to_sparse()
    # MNAR
    betahat, bThetahat, numI, betahats, bThetahats, Likelis = RealDataAlg(100, X, Y, sR, conDenfs, etab=etabM, etaT=etaTM,
                                                                          Cb=Cb, CT=CT, tols=tols, log=0, betainit=marbetahat, 
                                                                          bThetainit=marbThetahat, ErrOpts=1)

    Yrawt = torch.tensor(Yraw)
    
    betaX = torch.matmul(X, betahat)
    TbX = bThetahat + betaX
    marbetaX= torch.matmul(X, marbetahat)
    marTbX = marbThetahat + marbetaX
    hatprobs = torch.exp(TbX) / (1+torch.exp(TbX))
    marhatprobs = torch.exp(marTbX) / (1+torch.exp(marTbX))
    
    mask = (R == 0) & (Yrawt != -1)
    estprobs = hatprobs[mask]
    marestprobs = marhatprobs[mask]
    gtY = Y[mask]
    
    mnarres.append([estprobs.cpu().numpy(), gtY.cpu().numpy()])
    marres.append([marestprobs.cpu().numpy(), gtY.cpu().numpy()])



res = {
"MNARres": mnarres,
"MARres": marres,
"EMres": emres
}
f = open(f"./MNARxMARxEMyelp_log_{int(1000*OR)}.pkl", "wb")
pickle.dump(res, f)
f.close()
