from IPython import display
from utilsNew import *
from utils import *
import random
import numpy as np
import torch
import pickle
import timeit
import time
import argparse
import pprint
from pathlib import Path
from confs import fn, fn2, fn22, LogFn
#from tqdm import tqdm_notebook as tqdm
from tqdm.notebook import tqdm
import numpy.random as npr

import matplotlib.pyplot as plt
import pickle


# Generate X from Uniform distribution
def genXUnif(*args, limits=[0, 1], is_sparse=True):
    assert len(args) in [2, 3]
    p, size = args[-1], args[:-1]
    X = npr.uniform(limits[0], limits[1], args)
    if len(args) == 2:
        X = X.transpose()
    if is_sparse:
        return torch.tensor(X, device="cpu").to(dtorchdtype).to_sparse().cuda()
    else:
        return torch.tensor(X).to(dtorchdtype)


cudaid = 2
loglv = 2
torch.cuda.set_device(cudaid)
#------------------------------------------------------------------------------------
# fix the random seed for several packages
torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
np.random.seed(0) #numpy
random.seed(0) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

cuda = torch.cuda.is_available()
if cuda:
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)


def GenRpart(R, rvRate=0.2):
    """
    Remove rvRate 1's to calculate prediction error
    R: orginal missing matrix 
    rvRate: The percentage to remove
    """
    numN0s = int(R.to_dense().sum().item())
    numRVs = int(rvRate*numN0s)
    rvIdxs = np.random.choice(numN0s, numRVs, replace=False)
    RpartVec = torch.ones(numN0s)
    RpartVec[rvIdxs] = 0
    Rpart = R.to_dense().clone()
    Rpart[Rpart.bool()] = RpartVec
    return Rpart.to_sparse()


def calPreMSErr(Y, Yhat, R, Rpart):
    """
    calculate MSE given predicted Y and Y.
    """
    mvR = R.to_dense().bool() ^ Rpart.to_dense().bool()
    Ydiff = Yhat - Y
    mse = torch.mean(Ydiff[mvR]**2)
    nmse = mse / torch.mean(Y[mvR]**2)
    return mse.item()


def fRun(R, CT, Cb, etaT, etab, maxIter=100, is_plot=False, is_showPro=False):
    LamT = LamTfn(CT, n, m, p)
    Lamb = Lambfn(Cb, n, m)
    betaOld, bThetaOld = betainit, bThetainit
    
    
    reCh = 10
    Losses = []
    betaDiffs = []
    bThetaDiffs = []
    betaErrs = []
    bThetaErrs = []
    betaL0s = []
    bThetaRanks = []
    
    if is_showPro:
        IterShowBar = tqdm(range(maxIter))
    else:
        IterShowBar = range(maxIter)
    for t in IterShowBar:
        t0 = time.time()
        LvNow = lossL(bThetaOld, betaOld, f, X, Y, R, fct=fct, is_logf=False)
        LossNow = missdepLR(LvNow, bThetaOld, betaOld, LamT, Lamb)
       
        t1 = time.time()
        # update beta
        LpbvOld = lossLpb(bThetaOld, betaOld, conDenfs[:3], X, Y, R, fct=fct)
        betaNewRaw = betaOld - etab * LpbvOld
        betaNew = SoftTO(betaNewRaw, etab*Lamb)
       
        t2 = time.time()
        LpTvOld = lossLpT(bThetaOld, betaNew, conDenfs[:3], X, Y, R, fct=fct)
        #LvNew = lossL(bThetaOld, betaNew, f, X, Y, R, fct=fct, is_logf=False)
        #LossNew = missdepLR(LvNew, bThetaOld, betaNew, LamT, Lamb)
        Losses.append(LossNow.item())
       
        t3 = time.time()
        # update Theta
        svdres = torch.svd(bThetaOld-LpTvOld*etaT)
        U, S, V =  svdres.U, svdres.S, svdres.V
        softS = (S-LamT*etaT).clamp_min(0)
        bThetaNew = U.matmul(torch.diag(softS)).matmul(V.t())
        t4 = time.time()
        
        ts = np.array([t0, t1, t2, t3, t4])
        #print(np.diff(ts))
    
        if t >= 1:
            Lk1 = Losses[-1]
            Lk = Losses[-2]
            reCh = np.abs(Lk1-Lk)/np.max(np.abs((Lk, Lk1, 1))) 
        if (reCh < tol):
            break
        betaDiff = (betaOld-betaNew).norm().item()/(betaNew.norm().item()+1e-6)
        bThetaDiff = (bThetaOld-bThetaNew).norm().item()/(bThetaNew.norm().item()+1e-6)
        betaDiffs.append(betaDiff)
        bThetaDiffs.append(bThetaDiff)
        betaErrs.append((betaNew-beta0).norm().item())
        bThetaErrs.append((bThetaNew-bTheta0).norm().item())
        betaL0s.append(len(torch.nonzero(betaNew)))
        bThetaRanks.append(torch.linalg.matrix_rank(bThetaNew).item())
        
        if (bThetaDiff < tolT) and (betaDiff < tolb):
            break
        betaOld, bThetaOld = betaNew, bThetaNew 
        etaT = etaT * 0.95
        etab = etab * 0.95
        
        
        # plot res
        if t >= 1 and is_plot:
            plt.figure(figsize=[15, 10])
            
            plt.subplot(231)
            plt.xlim([0, maxIter])
            #plt.ylim([-10, 5])
            plt.title("Loss Diff")
            lossDifs = np.log(np.abs(np.diff(Losses)))
            plt.plot(list(range(0, t)), lossDifs)
            plt.scatter(t-1, lossDifs[-1], color="red", s=20)
            plt.axhline(y=np.log(tol), color="green", linewidth=2, linestyle='--')
            
            plt.subplot(232)
            plt.xlim([0, maxIter])
            #plt.ylim([-5, 0])
            plt.title("Beta Err")
            betaErrsArr = np.log(np.array(betaErrs))
            plt.plot(list(range(0, t+1)), betaErrsArr)
            plt.scatter(t, betaErrsArr[-1], color="red", s=20)
            plt.text(t, betaErrsArr[-1], f"Error is {np.exp(betaErrsArr[-1]):.3e}")
            
            plt.subplot(233)
            plt.xlim([0, maxIter])
            #plt.ylim([-5, 0])
            plt.title("Theta Err")
            bThetaErrsArr = np.log(np.array(bThetaErrs))
            plt.plot(list(range(0, t+1)), bThetaErrsArr)
            plt.scatter(t, bThetaErrsArr[-1], color="red", s=20)
            plt.text(t, bThetaErrsArr[-1], f"Error is {np.exp(bThetaErrsArr[-1]):.3e}")
            
            plt.subplot(235)
            plt.xlim([0, maxIter])
            #plt.ylim([0, p])
            plt.title("Beta L0")
            plt.plot(list(range(0, t+1)), betaL0s)
            plt.scatter(t, betaL0s[-1], color="red", s=20)
            plt.text(t, betaL0s[-1], f"L0 norm is {betaL0s[-1]:.0f}")
            plt.axhline(y=5, color="green", linewidth=2, linestyle='--')
            
            plt.subplot(236)
            plt.xlim([0, maxIter])
            #plt.ylim([0, np.min([n, m])])
            plt.title("Theta Rank")
            plt.plot(list(range(0, t+1)), bThetaRanks)
            plt.scatter(t, bThetaRanks[-1], color="red", s=20)
            plt.text(t, bThetaRanks[-1], f"rank is {bThetaRanks[-1]:.0f}")
            plt.axhline(y=5, color="green", linewidth=2, linestyle='--')
            
            display.clear_output(wait=True)
            plt.pause(1e-7)
    
    res = {}
    res["beta"] = betaNew
    res["bTheta"] = bThetaNew
    res["IterNum"] = t
    res["betaDiffs"] = betaDiffs
    res["bThetaDiffs"] = bThetaDiffs
    res["betaErrs"] = betaErrs
    res["bThetaErrs"] = bThetaErrs
    res["loss"] = Losses
    res["betaNorm0"] = len(torch.nonzero(betaNew))
    res["bThetaRank"] = torch.linalg.matrix_rank(bThetaNew).item()
    return res

conDenfs = [fn, fn2, fn22, LogFn]
f = fn
f2 = fn2

# for nv in [100, 200, 400, 800, 1600]:
#     n = m = nv
#     vs = []
#     for i in range(10):
#         bTheta0 = genbTheta(n, m, sigVs=np.array([10, 9, 8, 7, 6])*2*np.sqrt(m*n)/np.log(n*m)) 
#         vs.append(bTheta0.abs().max().item())
#     print(np.mean(vs))

# +
n = m = 50
p = 50
prob = 0.2

sigmaY = 1.0

# generate the parameters
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7))) 
bTheta0 = genbTheta(n, m, sigVs=np.array([10, 9, 8, 7, 6])*10) 
initbetapref = 1 + (torch.rand(p)-1/2)/40  #[0.875, 1.125]
initthetapref = 1 + (torch.rand(n, m)-1/2)/4
betainit = beta0 * initbetapref
bThetainit = bTheta0 * initthetapref
#betainit = torch.cat((torch.tensor([0.8, 0, 1.5, 0, -2.5, -4.3, 4.8]), torch.zeros(p-7))) * 10
#bThetainit = genbTheta(n, m, sigVs=np.array([15, 10, 7, 5, 5])/2) 

#X = genXUnif(n, m, p) 
X = genXBin(n, m, p, prob=prob) 
Y = genYnorm(X, bTheta0, beta0, sigmaY)
# around 95% missing rate
inps = {}
inps[50] = - 8.0
inps[100] = - 5.93
inps[200] = -5.30
inps[400] = -5.2
inps[800] = -5.15
inps[1600] = -5.1
R = genR(Y, "linear", inp=inps[n], slop=1)
R.to_dense().mean()


#Rpart = GenRpart(R)
# -

tol = 1e-6
tolT = 0
tolb = 0

# **p=50** 
# - n = 50:
#     paraX = [2e1, 4e2, 0.1, 0.1] 
#     errs = [1.806, 24.97]
#
# - n = 100:
#     paraX = [2e1, 8e2, 0.1, 0.1] 
#     errs = [1.211, 18.97]
#     
# - n = 200:
#     paraX = [2e1, 2e3, 0.1, 0.1] 
#     errs = [0.76, 14.5]
#     
# - n = 400:
#     paraX = [2e1, 4e3, 0.1, 0.1] 
#     errs = [0.596, 11.2]
#
# - n=800: 
#     paraX = [2e1, 1e4, 0.1, 0.1]
#     errs = [0.35., 8.3]
#     
# - n=1600: 
#     paraX = [2e1, 3e4, 0.1, 0.1]
#     errs = 

fct = 1
# CT, Cb, etaT, etab
paraX = [2e1, 2e3, 0.1, 0.1] 
res = fRun(R, paraX[0], paraX[1], paraX[2], paraX[3], 200, True)

res["betaErrs"][-1], res["bThetaErrs"][-1], res["betaNorm0"], res["bThetaRank"]]

# ### 1. Tuning parameters

# +
# CT, Cb, etaT, etab
CTs = [1e-2, 1e-1, 0.5]
Cbs = [5, 50, 500]
etaTs = [0.1, 1]
etabs = [1, 5]

paraXs = []
for ix in range(len(CTs)):
    for jx in range(len(Cbs)):
        for lx in range(len(etaTs)):
            for sx in range(len(etabs)):
                paraX = [CTs[ix], Cbs[jx], etaTs[lx], etabs[sx]]
                paraXs.append(paraX)
paraXs = torch.tensor(paraXs)
# -

optParas = {}
for nv in [100, 200, 400, 800]:
    fct = 1
    if nv >= 400:
        fct = 5
    n = m = nv
    p = 50
    prob = 0.2
    
    sigmaY = 1.0
    
    # generate the parameters
    beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7))) 
    bTheta0 = genbTheta(n, m, sigVs=np.array([10, 9, 8, 7, 6])*2*np.sqrt(n*m)/np.log(n*m)) 
    initbetapref = 1 + (torch.rand(p)-1/2)/4  #[0.875, 1.125]
    initthetapref = 1 + (torch.rand(n, m)-1/2)/4
    betainit = beta0 * initbetapref
    bThetainit = bTheta0 * initthetapref
    #betainit = torch.cat((torch.tensor([0.8, 0, 1.5, 0, -2.5, -4.3, 4.8]), torch.zeros(p-7))) * 10
    #bThetainit = genbTheta(n, m, sigVs=np.array([15, 10, 7, 5, 5])/2) 
    
    #X = genXUnif(n, m, p) 
    X = genXBin(n, m, p, prob=prob) 
    Y = genYnorm(X, bTheta0, beta0, sigmaY)
    # around 95% missing rate
    inps = {}
    inps[100] = - 8.2
    inps[200] = -7.5
    inps[400] = -7.1
    inps[800] = -6.7
    inps[1600] = -6.5
    R = genR(Y, "linear", inp=inps[n], slop=1)
    R.to_dense().mean()
    
    
    Rpart = GenRpart(R)
    paraYs = []
    paraYsSep = []
    for idx in tqdm(range(len(paraXs))): 
        paraX = paraXs[idx, :]
        paraX = paraX.cpu().numpy()
        res = fRun(Rpart, paraX[0], paraX[1], paraX[2], paraX[3])
        Yhat = torch.matmul(X.to_dense(), res["beta"]) + res["bTheta"]
        nmse = calPreMSErr(Y, Yhat, R, Rpart)
        paraYsSep.append([nmse, res["betaErrs"][-1], res["bThetaErrs"][-1], res["betaNorm0"], res["bThetaRank"]])
        #print(idx, paraYsSep[-1])
        paraY = (nmse + 0.0*res["betaNorm0"]/p + 0.0*res["bThetaRank"]/np.min([m, n])) * 100
        paraYs.append(paraY)
        
    paraYsSepArr = np.array(paraYsSep)
    errs = paraYsSepArr[:, 1]/np.sqrt(p) + paraYsSepArr[:, 2]/np.sqrt(n*m)
    minIdx = np.argmin(np.array(errs))
    print(n, paraYsSepArr[minIdx, 1]/np.sqrt(p), paraYsSepArr[minIdx, 2]/np.sqrt(n*m))
    optParas[nv] = list(paraXs[minIdx, :])

# paraYsSepArr = np.array(paraYsSep)
# MseIdx = np.argsort(np.argsort(paraYsSepArr[:, 0]))
# betaErrIdx = np.argsort(np.argsort(paraYsSepArr[:, 1]))
# bThetaErrIdx = np.argsort(np.argsort(paraYsSepArr[:, 2]))

# plt.figure(figsize=[15, 5])
# plt.subplot(131)
# plt.plot(MseIdx, betaErrIdx, "o")
# plt.subplot(132)
# plt.plot(MseIdx, bThetaErrIdx, "o")
# plt.subplot(133)
# plt.plot(MseIdx, (betaErrIdx+bThetaErrIdx)/2, "o")

# errs = paraYsSepArr[:, 1]/np.sqrt(p) + paraYsSepArr[:, 2]/np.sqrt(n*m)
# errsIdx = np.argsort(np.argsort(errs))
# plt.figure(figsize=[5, 5])
# plt.plot(MseIdx, errsIdx, "o")

# print(np.argmin(np.array(errs)))
# paraXs[np.argmin(np.array(errs)), :]

# ### opt paras
# CT, Cb, etaT, etab
#
# - 1600: 
# - 800: 
# - 400: 
# - 200: 
# - 100: 

# ## Obtain the results 

# +
optParas = {}
optParas[50] = [2e1, 4e2, 0.1, 0.1] 
optParas[100] = [2e1, 8e2, 0.5, 0.1] 
optParas[200] = [2e1, 2e3, 2.0, 0.1] 
optParas[400] = [2e1, 4e3, 5.0, 0.1] 
optParas[800] = [2e1, 1e4, 10, 0.1]
optParas[1600] = [2e1, 3e4, 50, 0.1]

#optParas[50] = [2e1, 4e2, 0.1, 0.1] 
#optParas[100] = [2e1, 8e2, 0.1, 0.1] 
#optParas[200] = [2e1, 2e3, 0.1, 0.1] 
#optParas[400] = [2e1, 4e3, 0.1, 0.1] 
#optParas[800] = [2e1, 1e4, 0.1, 0.1]
#optParas[1600] = [2e1, 3e4, 0.1, 0.1]
# -

for n in [50, 100, 200, 400, 800]:
    fct = 1
    if n ==1600:
        fct = 20
    elif n == 800:
        fct = 10
    elif n == 400:
        fct = 5
    m = n
    
    beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7))) 
    bTheta0 = genbTheta(n, m, sigVs=np.array([10, 9, 8, 7, 6])*10) 
    
    ress = []
    paraX = optParas[n]
    for j in tqdm(range(50)):
        initbetapref = 1 + (torch.rand(p)-1/2)/40
        initthetapref = 1 + (torch.rand(n, m)-1/2)/4
        betainit = beta0 * initbetapref
        bThetainit = bTheta0 * initthetapref
        X = genXBin(n, m, p, prob=prob) 
        Y = genYnorm(X, bTheta0, beta0, sigmaY)
        # around 95% missing rate
        R = genR(Y, "linear", inp=inps[n], slop=1)
        res = fRun(R, paraX[0], paraX[1], paraX[2], paraX[3], 200)
        ress.append(res)
    
    
    ressDic = {}
    ressDic["ress"] = ress
    ressDic["beta0"] = beta0
    ressDic["bTheta0"] = bTheta0
    with open(f"./JMLRR2_linear_p{p}_n{n}_New.pkl", "wb") as wf:
        pickle.dump(ressDic, wf)

errss = []
p = 50
for n in [50, 100, 200, 400, 800]:
    with open(f"./JMLRR2_linear_p{p}_n{n}.pkl", "rb") as rf:
        ressDic = pickle.load(rf)
        
    errs = [(torch.norm(res["beta"]-ressDic["beta0"]).item(), torch.norm(res["bTheta"]-ressDic["bTheta0"]).item()) for res in ressDic["ress"]]
    errss.append(np.array(errs).mean(axis=0))

errss

# ---- 
# ## BayOpt (not used)

# +
# for BayOpt
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to

import pyro
import pyro.contrib.gp as gp

assert pyro.__version__.startswith('1.7.0')
pyro.set_rng_seed(1)


# +
def update_posterior(x_new, y_new):
    x_new = x_new.reshape(1, -1)
    X = torch.cat([gpmodel.X, x_new]) # incorporate new evaluation
    y = torch.cat([gpmodel.y, y_new])
    gpmodel.set_data(X, y)
    # optimize the GP hyperparameters using Adam with lr=0.001
    optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
    gp.util.train(gpmodel, optimizer)
    
    
def lower_confidence_bound(x, kappa=2):
    mu, variance = gpmodel(x, full_cov=False, noiseless=False)
    sigma = variance.sqrt()
    return mu - kappa * sigma


def find_a_candidate(x_init, lower_bound=-10, upper_bound=10):
    # transform x to an unconstrained domain
    _, numEle = x_init.shape
    #print(x_init.shape)
    if not isinstance(lower_bound, (list, tuple, torch.Tensor)):
        lower_bound = torch.tensor([lower_bound]*numEle).double()
        upper_bound = torch.tensor([upper_bound]*numEle).double()
    elif isinstance(lower_bound, torch.Tensor):
        lower_bound = lower_bound.double()
        upper_bound = upper_bound.double()
    else:
        lower_bound = torch.tensor(lower_bound).double()
        upper_bound = torch.tensor(upper_bound).double()
        
    constraint = constraints.interval(lower_bound, upper_bound)
    unconstrained_x_init = transform_to(constraint).inv(x_init)
    unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
    minimizer = optim.LBFGS([unconstrained_x], line_search_fn='strong_wolfe')

    def closure():
        minimizer.zero_grad()
        x = transform_to(constraint)(unconstrained_x)
        y = lower_confidence_bound(x)
        autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
        return y

    minimizer.step(closure)
    # after finding a candidate in the unconstrained domain,
    # convert it back to original domain.
    x = transform_to(constraint)(unconstrained_x)
    return x.detach()

def next_x(lower_bound=-10, upper_bound=10, num_candidates=5):
    candidates = []
    values = []
    
        

    x_init = gpmodel.X[-1, :].reshape(1, -1)
    _, numEle = x_init.shape
    
    if not isinstance(lower_bound, (list, tuple, torch.Tensor)):
        lower_bound = torch.tensor([lower_bound]*numEle).double()
        upper_bound = torch.tensor([upper_bound]*numEle).double()
    else:
        lower_bound = torch.tensor(lower_bound).double()
        upper_bound = torch.tensor(upper_bound).double()
    for i in range(num_candidates):
        x = find_a_candidate(x_init, lower_bound, upper_bound)
        y = lower_confidence_bound(x)
        candidates.append(x)
        values.append(y)
        x_init = [(torch.rand(1)*(up - low)+low).item() for low, up in zip(lower_bound, upper_bound)]
        x_init = torch.tensor(x_init).reshape(1, -1).double()

    argmin = torch.min(torch.cat(values), dim=0)[1].item()
    return candidates[argmin]

# +
fct = 1

tol = 0
tolT = 1e-4
tolb = 1e-4
Rpart = GenRpart(R)

# +
lower_bound = [1e-4, 1e-4, 1e-4, 1e-4]
upper_bound = [10, 1000, 10, 10]

Cb, CT = 5e1, 1e-1
etab, etaT = 1.0, 1.0
paraXs = torch.tensor([[CT, Cb, etaT, etab],
                       [1, 10, 0.5, 1], 
                       [0.5, 100, 0.1, 0.1], 
                       [0.01, 40, 2, 2]
                      ])
# -

paraYs = []
paraYsSep = []
for paraX in paraXs: 
    paraX = paraX.cpu().numpy()
    res = fRun(Rpart, paraX[0], paraX[1], paraX[2], paraX[3])
    Yhat = torch.matmul(X.to_dense(), res["beta"]) + res["bTheta"]
    nmse = calPreMSErr(Y, Yhat, R, Rpart)
    paraYsSep.append([nmse, res["betaNorm0"], res["bThetaRank"]])
    paraY = (nmse + 0.0*res["betaNorm0"]/p + 0.0*res["bThetaRank"]/np.min([m, n])) * 100
    paraYs.append(paraY)

# +
#paraYs = torch.tensor(np.array(paraYsSep)[:, 0]).double()
#paraXs = gpmodel.X
paraYs = torch.tensor(paraYs).double()
gpmodel = gp.models.GPRegression(paraXs, paraYs, gp.kernels.Matern52(input_dim=4),
                                 noise=torch.tensor(0.1), jitter=1.0e-4)

optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
_ = gp.util.train(gpmodel, optimizer)
# -

numBOIters = 10
for i in range(numBOIters):
    paraXmin = next_x(lower_bound, upper_bound, 10)
    paraX = paraXmin.cpu().numpy()[0, :]
    res = fRun(Rpart, paraX[0], paraX[1], paraX[2], paraX[3])
    Yhat = torch.matmul(X.to_dense(), res["beta"]) + res["bTheta"]
    
    nmse = calPreMSErr(Y, Yhat, R, Rpart)
    paraYsSep.append([nmse, res["betaNorm0"], res["bThetaRank"]])
    paraY = (nmse + 0.0*res["betaNorm0"]/p + 0.0*res["bThetaRank"]/np.min([m, n])) * 100
    
    paraYnew = torch.tensor([paraY]).float()
    print(i, paraXmin, lower_confidence_bound(paraXmin))
    update_posterior(paraXmin, paraYnew)
