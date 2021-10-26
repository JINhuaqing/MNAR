import torch
import numpy as np
import numpy.random as npr
from torch.distributions.normal import Normal
from prettytable import PrettyTable
import time

# seps: small number to avoid zero in log funciton and denominator. 
#seps = 1e-15
seps = 1e-320
seps = 0
# dtorchdtype and dnpdtype are default data types used for torch package and numpy package, respectively.
dtorchdtype = torch.float64
dnpdtype = np.float64


# All the `lossL`, `lossLpT` and `lossLpb` functions for `MNAR`, `MAR` and `EM` are checked

# ###  MNAR

# Compute the value of L with MCMC method for any distributions X.
def lossL(bTheta, beta, f, X, Y, R, fct=10, is_logf=False, N=10000):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    f: likelihood function of Y|X
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    N: number of samples for MCMC
    """
    # reshape is by row 
    
    # convert X, R to sparse format
    if not R.is_sparse:
        R = R.to_sparse()
    if not X.is_sparse:
        X = X.to_sparse()
    
    n, m, p = X.shape
    # Choose the first
    sXs = (X.to_dense().reshape(-1, p).t()[:, :N]).to_sparse() # p x N


    
    YvecP = Y[R.to_dense().bool()] # M 
    bThetaVecP = bTheta[R.to_dense().bool()] # M 
    XvecP = X.to_dense()[R.to_dense().bool()].to_sparse() # M x p
    betaXvecP = torch.matmul(XvecP.to_dense(), beta) # M 
    TbXvecP = bThetaVecP + betaXvecP
    bsXs = beta.matmul(sXs.to_dense())
    
    if is_logf:
        itm1 = f(YvecP, TbXvecP)
    else:
        itm1 = torch.log(f(YvecP, TbXvecP)+seps)
    
    
    M, _ = XvecP.shape # M: number of non missing term 
    lenSeg = int(np.ceil(M/fct))
    itm2 = torch.zeros(M)
    for i in np.arange(0, M, lenSeg):
        lower, upper = i, i+lenSeg
        YvecPP = YvecP[lower:upper]
        bThetaVecPP = bThetaVecP[lower:upper]
        if is_logf:
            itm2Part = torch.log(torch.exp(f(YvecPP, bThetaVecPP, bsXs)).mean(dim=-1)+seps)
        else:
            itm2Part = torch.log(f(YvecPP, bThetaVecPP, bsXs).mean(dim=-1)+seps)
        itm2[lower:upper] = itm2Part
    

    itm = -(itm1 - itm2)
    #return itm.sum()
    return itm.sum()/m/n


# Compute the value of first derivative of L w.r.t bTheta with MCMC method for any distributions X.
def lossLpT(bTheta, beta, conDenfs, X, Y, R, fct=10, N=10000):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    fct: the number of times to integrate 
    """
    # convert X, R to sparse format
    if not R.is_sparse:
        R = R.to_sparse()
    if not X.is_sparse:
        X = X.to_sparse()
    
    n, m, p = X.shape
    sXs = (X.to_dense().reshape(-1, p).t()[:, :N]).to_sparse()
    f, f2, _= conDenfs

    # remove the elements whose corresponding betaK is zero
    idxNon0 = torch.nonzero(beta).view(-1)
    if idxNon0.shape[0] == 0:
        idxNon0 = torch.tensor([0, 1], dtype=torch.int64)
    beta = beta[idxNon0] # p0 x 1
    sXs = sXs.to_dense()[idxNon0].to_sparse() # p0 x N
    X = X.to_dense()[:, :, idxNon0].to_sparse() # n x m x p0
    
    YvecP = Y[R.to_dense().bool()] # M 
    bThetaVecP = bTheta[R.to_dense().bool()] # M 
    XvecP = X.to_dense()[R.to_dense().bool()].to_sparse() # M x p
    betaXvecP = torch.matmul(XvecP.to_dense(), beta) # M 
    TbXvecP = bThetaVecP + betaXvecP
    bsXs = beta.matmul(sXs.to_dense()) # N
    M, _ = XvecP.shape

    itm1 = (f2(YvecP, TbXvecP)/(f(YvecP, TbXvecP)+seps))
    torch.cuda.empty_cache()

    lenSeg = int(np.ceil(M/fct))
    itm2 = torch.zeros(M)
    for i in np.arange(0, M, lenSeg):
        lower, upper = i, i+lenSeg
        YvecPP = YvecP[lower:upper]
        bThetaVecPP = bThetaVecP[lower:upper]
        itm2denPart = f(YvecPP, bThetaVecPP, bsXs).mean(dim=-1) + seps
        itm2numPart = f2(YvecPP, bThetaVecPP, bsXs).mean(dim=-1)
        itm2Part = itm2numPart / itm2denPart
        itm2[lower:upper] = itm2Part

    #itmP = - (itm1 - itm2)
    itmP = - (itm1 - itm2)/(m*n)
    itm = torch.zeros((n, m))
    itm[R.to_dense().bool()] = itmP
    return itm


# Compute the value of first derivative of L w.r.t beta with MCMC method for any distributions X.
def lossLpb(bTheta, beta, conDenfs, X, Y, R, fct=10, N=10000):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    N: number of samples for MCMC
    """
    # convert X, R to sparse format
    if not R.is_sparse:
        R = R.to_sparse()
    if not X.is_sparse:
        X = X.to_sparse()
        
    n, m, p = X.shape
    sXs = (X.to_dense().reshape(-1, p).t()[:, :N]).to_sparse()

    f, f2, _ = conDenfs
    # remove the elements whose corresponding betaK is zero
    idxNon0 = torch.nonzero(beta).view(-1)
    if idxNon0.shape[0] == 0:
        idxNon0 = torch.tensor([0, 1], dtype=torch.int64)
    betaNon0 = beta[idxNon0]
    sXsNon0 = sXs.to_dense()[idxNon0].to_sparse()
    XNon0 = X.to_dense()[:, :, idxNon0].to_sparse()

    YvecP = Y[R.to_dense().bool()] # M 
    bThetaVecP = bTheta[R.to_dense().bool()] # M 
    XvecP0 = XNon0.to_dense()[R.to_dense().bool()].to_sparse() # M x p0
    XvecP = X.to_dense()[R.to_dense().bool()].to_sparse() #M x p
    betaXvecP = torch.matmul(XvecP0.to_dense(), betaNon0) # M 
    TbXvecP = bThetaVecP + betaXvecP
    bsXs = betaNon0.matmul(sXsNon0.to_dense()) # N
    M, _ = XvecP0.shape
    
    lenSeg = int(np.ceil(M/fct))
    sumRes = torch.zeros(p)
    for i in np.arange(0, M, lenSeg):
        lower, upper = i, i+lenSeg
        TbXvecPP = TbXvecP[lower:upper]
        YvecPP = YvecP[lower:upper]
        bThetaVecPP = bThetaVecP[lower:upper]
        XvecPP = (XvecP.to_dense()[lower:upper, :]).to_sparse() #lenSeg x p 

        itm1Part = ((f2(YvecPP, TbXvecPP)/(f(YvecPP, TbXvecPP)+seps)).unsqueeze(dim=1) * XvecPP.to_dense()).to_sparse() # lenSeg x p 

        itm2denPart = (f(YvecPP, bThetaVecPP, bsXs).mean(dim=-1) + seps) # lenSeg 
        itm2numinPart =  f2(YvecPP, bThetaVecPP, bsXs).to_sparse() # lenSeg x N 
        itm2numPart = (itm2numinPart.to_dense().unsqueeze(dim=1) * sXs.to_dense().unsqueeze(dim=0)).mean(dim=-1).to_sparse() # lenSeg x p
        itm2Part = (itm2numPart.to_dense()/itm2denPart.unsqueeze(dim=-1)).to_sparse()
        itmPart = (itm1Part.to_dense() - itm2Part.to_dense()).to_sparse()
        sumRes += itmPart.to_dense().sum(dim=0)
        torch.cuda.empty_cache()
    #return -sumRes
    return -sumRes/n/m


# ### MNAR exact

# The below Blist, intBernh, and intBernhX functions are to compute the 
# exact integration when X is Bernoulli and beta is sparse ( nonzeros value of beta is less than 14)

# Blist is to generate all possible results for s binary data which will be used in exact integration.
# e.g, s = 3, then Blist return a matrix like
#  0,  0,  0 
#  0,  0,  1
#  0,  1,  0
#  0,  1,  1
#  1,  0,  0
#  1,  0,  1
#  1,  1,  0 
#  1,  1,  1
def Blist(s):
    slist = list(range(2**s))
    mat = []
    for i in slist:
        strbi = bin(i)
        strbi = strbi.split("b")[-1]
        strbi = (s-len(strbi)) * "0" + strbi
        mat.append(list(strbi))
    matarr = np.array(mat) 
    return matarr.astype(dnpdtype)


# Compute the exact integration of \int f(Y, bTheta + beta\trans X ) g(X) dX when beta is sparse
def intBernh(f, bTheta, beta, Y, probs):
    idxNon0 = torch.nonzero(beta).view(-1)
    s = (beta != 0).sum().to(dtorchdtype).item()
    s = int(s)
    if isinstance(probs, float) or len(probs)==1:
        probs = torch.ones(beta.shape) * probs
    else:
        probs = torch.tensor(probs)
    if s != 0:
        Blistmat = torch.tensor(Blist(s))
        beta = beta[idxNon0]
        probs = probs[idxNon0]
        Blstsum = Blistmat.sum(1)
        sum1 = f(Y, bTheta, Blistmat.matmul(beta)) # n x m x 2^s
        sum2 = torch.prod(probs**Blistmat * (1-probs)**(1-Blistmat), axis=1) # 2^s
        #sum2 = prob**Blstsum * (1-prob)**(s-Blstsum) # 2^s
        summ = sum1 * sum2
        return summ.sum(-1)
    else:
        summ = f(Y, bTheta)
        return summ


# Compute the exact integration of \int f(Y, bTheta + beta\trans X ) X g(X) dX when beta is sparse
def intBernhX(f, bTheta, beta, Y, probs):
    p = beta.shape[0]
    idxNon0 = torch.nonzero(beta).view(-1)
    s = (beta != 0).sum().to(dtorchdtype).item()
    s = int(s)
    if isinstance(probs, float) or len(probs)==1:
        probs = torch.ones(beta.shape) * probs
    else:
        probs = torch.tensor(probs)
    if s != 0:
        Blistmat = torch.tensor(Blist(s))
        probsPart = probs[idxNon0]
        beta = beta[idxNon0]
        Blstsum = Blistmat.sum(1)
        #sum2 = prob**Blstsum * (1-prob)**(s-Blstsum) # 2^s
        sum2 = torch.prod(probsPart**Blistmat * (1-probsPart)**(1-Blistmat), axis=1) # 2^s
        summ = []
        for ii in range(p):
            if ii not in idxNon0:
                sump = (f(Y, bTheta, Blistmat.matmul(beta))* probs[ii] * sum2).sum(-1) #n x m 
                summ.append(sump)
            else:
                idx = torch.nonzero(idxNon0 == ii).view(-1)
                Xc = Blistmat[:, idx].squeeze()
                sump = (f(Y, bTheta, Blistmat.matmul(beta))* Xc* sum2).sum(-1) #n x m 
                summ.append(sump)
        return torch.stack(summ, dim=-1) 
    else:
        expX = torch.ones(p) * probs
        return f(Y, bTheta).unsqueeze(-1) * expX


# Compute the value of L with MCMC method for any binary X
def lossLBern(bTheta, beta, f, X, Y, R, probs, fct=10, is_logf=False):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    f: likelihood function of Y|X
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    """
    # reshape is by row 
    
    # convert X, R to sparse format
    if not R.is_sparse:
        R = R.to_sparse()
    if not X.is_sparse:
        X = X.to_sparse()
    
    n, m, p = X.shape
    # Choose the first
    YvecP = Y[R.to_dense().bool()] # M 
    bThetaVecP = bTheta[R.to_dense().bool()] # M 
    XvecP = X.to_dense()[R.to_dense().bool()].to_sparse() # M x p
    betaXvecP = torch.matmul(XvecP.to_dense(), beta) # M 
    TbXvecP = bThetaVecP + betaXvecP
    
    if is_logf:
        itm1 = f(YvecP, TbXvecP)
    else:
        itm1 = torch.log(f(YvecP, TbXvecP)+seps)
    
    if is_logf:
        def inpF(y, m, bsXs):
            return torch.exp(f(y, m, bsXs))
    else:
        inpF = f

    
    M, _ = XvecP.shape # M: number of non missing term 
    lenSeg = int(np.ceil(M/fct))
    itm2 = torch.zeros(M)
    for i in np.arange(0, M, lenSeg):
        lower, upper = i, i+lenSeg
        YvecPP = YvecP[lower:upper]
        bThetaVecPP = bThetaVecP[lower:upper]
        itm2Part = torch.log(intBernh(inpF, bThetaVecPP, beta, YvecPP, probs)+seps)
        itm2[lower:upper] = itm2Part
    

    itm = -(itm1 - itm2)
    #return itm.sum()
    return itm.sum()/m/n


# Compute the value of first derivative of L w.r.t beta with MCMC method for any binary X.
def lossLpbBern(bTheta, beta, conDenfs, X, Y, R, probs, fct=10):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    """
    # convert X, R to sparse format
    if not R.is_sparse:
        R = R.to_sparse()
    if not X.is_sparse:
        X = X.to_sparse()
        
    n, m, p = X.shape

    f, f2, _ = conDenfs
    # remove the elements whose corresponding betaK is zero
    idxNon0 = torch.nonzero(beta).view(-1)
    if idxNon0.shape[0] == 0:
        idxNon0 = torch.tensor([0, 1], dtype=torch.int64)
    betaNon0 = beta[idxNon0]
    XNon0 = X.to_dense()[:, :, idxNon0].to_sparse()

    YvecP = Y[R.to_dense().bool()] # M 
    bThetaVecP = bTheta[R.to_dense().bool()] # M 
    XvecP0 = XNon0.to_dense()[R.to_dense().bool()].to_sparse() # M x p0
    XvecP = X.to_dense()[R.to_dense().bool()].to_sparse() #M x p
    betaXvecP = torch.matmul(XvecP0.to_dense(), betaNon0) # M 
    TbXvecP = bThetaVecP + betaXvecP
    M, _ = XvecP0.shape
    
    lenSeg = int(np.ceil(M/fct))
    sumRes = torch.zeros(p)
    for i in np.arange(0, M, lenSeg):
        lower, upper = i, i+lenSeg
        TbXvecPP = TbXvecP[lower:upper]
        YvecPP = YvecP[lower:upper]
        bThetaVecPP = bThetaVecP[lower:upper]
        XvecPP = (XvecP.to_dense()[lower:upper, :]).to_sparse() #lenSeg x p 

        itm1Part = ((f2(YvecPP, TbXvecPP)/(f(YvecPP, TbXvecPP)+seps)).unsqueeze(dim=1) * XvecPP.to_dense()).to_sparse() # lenSeg x p 

        itm2denPart = intBernh(f, bThetaVecPP, beta, YvecPP, probs) + seps
        itm2numPart = intBernhX(f2, bThetaVecPP, beta, YvecPP, probs).to_sparse()
        itm2Part = (itm2numPart.to_dense()/itm2denPart.unsqueeze(dim=-1)).to_sparse()
        itmPart = (itm1Part.to_dense() - itm2Part.to_dense()).to_sparse()
        sumRes += itmPart.to_dense().sum(dim=0)
        torch.cuda.empty_cache()
    #return -sumRes
    return -sumRes/n/m


# Compute the value of first derivative of L w.r.t bTheta with MCMC method for binary X.
def lossLpTBern(bTheta, beta, conDenfs, X, Y, R, probs, fct=10):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    fct: the number of times to integrate 
    """
    # convert X, R to sparse format
    if not R.is_sparse:
        R = R.to_sparse()
    if not X.is_sparse:
        X = X.to_sparse()
    
    n, m, p = X.shape
    f, f2, _= conDenfs

    # remove the elements whose corresponding betaK is zero
    idxNon0 = torch.nonzero(beta).view(-1)
    if idxNon0.shape[0] == 0:
        idxNon0 = torch.tensor([0, 1], dtype=torch.int64)
    beta = beta[idxNon0] # p0 x 1
    X = X.to_dense()[:, :, idxNon0].to_sparse() # n x m x p0
    
    YvecP = Y[R.to_dense().bool()] # M 
    bThetaVecP = bTheta[R.to_dense().bool()] # M 
    XvecP = X.to_dense()[R.to_dense().bool()].to_sparse() # M x p
    betaXvecP = torch.matmul(XvecP.to_dense(), beta) # M 
    TbXvecP = bThetaVecP + betaXvecP
    M, _ = XvecP.shape

    itm1 = (f2(YvecP, TbXvecP)/(f(YvecP, TbXvecP)+seps))
    torch.cuda.empty_cache()

    lenSeg = int(np.ceil(M/fct))
    itm2 = torch.zeros(M)
    for i in np.arange(0, M, lenSeg):
        lower, upper = i, i+lenSeg
        YvecPP = YvecP[lower:upper]
        bThetaVecPP = bThetaVecP[lower:upper]
        itm2denPart = intBernh(f, bThetaVecPP, beta, YvecPP, probs) + seps
        itm2numPart = intBernh(f2, bThetaVecPP, beta, YvecPP, probs)
        itm2Part = itm2numPart / itm2denPart
        itm2[lower:upper] = itm2Part

    #itmP = - (itm1 - itm2)
    itmP = - (itm1 - itm2)/(m*n)
    itm = torch.zeros((n, m))
    itm[R.to_dense().bool()] = itmP
    return itm


# calaulte the updated bTheta under MNAR for linear, approximately
# I didn't check it before. 
# the theta is before soft-thresholding
def mnarLinearUpdateThetaApprox(X, Y, R, bThetaOld, beta, sigma=5):
    bThetaNew = bThetaOld.clone()
    betaX = X.to_dense().matmul(beta)
    mu = torch.mean(betaX)
    sigma2sq = torch.var(betaX)
    sigma1sq = sigma**2
    YvecP = Y[R.to_dense().bool()] # M 
    betaXvecP = betaX[R.to_dense().bool()]
    bThetaNewVecP = (YvecP * sigma2sq + mu * sigma1sq  - (sigma1sq+sigma2sq)*betaXvecP)/sigma2sq
    bThetaNew[R.to_dense().bool()] = bThetaNewVecP
    return bThetaNew


# ### MAR 

# Compute the value of L with MCMC method for any distributions X under MAR model
def marLossL(bTheta, beta, f, X, Y, R, is_logf=False):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    f: likelihood function of Y|X
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    """
    # reshape is by row 
    
    # convert X, R to sparse format
    if not R.is_sparse:
        R = R.to_sparse()
    if not X.is_sparse:
        X = X.to_sparse()
    
    n, m, p = X.shape
    # Choose the first


    
    YvecP = Y[R.to_dense().bool()] # M 
    bThetaVecP = bTheta[R.to_dense().bool()] # M 
    XvecP = X.to_dense()[R.to_dense().bool()].to_sparse() # M x p
    betaXvecP = torch.matmul(XvecP.to_dense(), beta) # M 
    TbXvecP = bThetaVecP + betaXvecP
    
    if is_logf:
        itm1 = f(YvecP, TbXvecP)
    else:
        itm1 = torch.log(f(YvecP, TbXvecP)+seps)
    
    
    itm = -itm1
    return itm.sum()/m/n


# Compute the value of first derivative of L w.r.t bTheta with MCMC method for any distributions X under MAR 
def marLossLpT(bTheta, beta, conDenfs, X, Y, R):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    """
    # convert X, R to sparse format
    if not R.is_sparse:
        R = R.to_sparse()
    if not X.is_sparse:
        X = X.to_sparse()
    
    n, m, p = X.shape
    f, f2, _= conDenfs

    # remove the elements whose corresponding betaK is zero
    idxNon0 = torch.nonzero(beta).view(-1)
    if idxNon0.shape[0] == 0:
        idxNon0 = torch.tensor([0, 1], dtype=torch.int64)
    beta = beta[idxNon0] # p0 x 1
    X = X.to_dense()[:, :, idxNon0].to_sparse() # n x m x p0
    
    YvecP = Y[R.to_dense().bool()] # M 
    bThetaVecP = bTheta[R.to_dense().bool()] # M 
    XvecP = X.to_dense()[R.to_dense().bool()].to_sparse() # M x p
    betaXvecP = torch.matmul(XvecP.to_dense(), beta) # M 
    TbXvecP = bThetaVecP + betaXvecP
    M, _ = XvecP.shape

    itm1 = (f2(YvecP, TbXvecP)/(f(YvecP, TbXvecP)+seps))
    
    itmP = - itm1/(m*n)
    itm = torch.zeros((n, m))
    itm[R.to_dense().bool()] = itmP
    return itm


# Compute the value of first derivative of L w.r.t beta with MCMC method for any distributions X under MAR models
def marLossLpb(bTheta, beta, conDenfs, X, Y, R):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    """
    # convert X, R to sparse format
    if not R.is_sparse:
        R = R.to_sparse()
    if not X.is_sparse:
        X = X.to_sparse()
        
    n, m, p = X.shape

    f, f2, _ = conDenfs
    # remove the elements whose corresponding betaK is zero
    idxNon0 = torch.nonzero(beta).view(-1)
    if idxNon0.shape[0] == 0:
        idxNon0 = torch.tensor([0, 1], dtype=torch.int64)
    betaNon0 = beta[idxNon0]
    XNon0 = X.to_dense()[:, :, idxNon0].to_sparse()

    YvecP = Y[R.to_dense().bool()] # M 
    bThetaVecP = bTheta[R.to_dense().bool()] # M 
    XvecP0 = XNon0.to_dense()[R.to_dense().bool()].to_sparse() # M x p0
    XvecP = X.to_dense()[R.to_dense().bool()].to_sparse() #M x p
    betaXvecP = torch.matmul(XvecP0.to_dense(), betaNon0) # M 
    TbXvecP = bThetaVecP + betaXvecP
    
    itm1 = ((f2(YvecP, TbXvecP)/(f(YvecP, TbXvecP)+seps)).unsqueeze(dim=1) * XvecP.to_dense()).to_sparse()
    sumRes = itm1.to_dense().sum(dim=0)
    return -sumRes/n/m


# calaulte the updated beta under MAR for linear
# I didn't check it before. 
# the beta is before soft-thresholding
def marLinearUpdateBeta(X, Y, R, bTheta):
    if not R.is_sparse:
        R = R.to_sparse()
    if not X.is_sparse:
        X = X.to_sparse()
    YvecP = Y[R.to_dense().bool()] # M 
    bThetaVecP = bTheta[R.to_dense().bool()] # M 
    XvecP = X.to_dense()[R.to_dense().bool()].to_sparse() #M x p
    itm1 = ((YvecP-bThetaVecP).reshape(-1, 1) * XvecP.to_dense()).sum(axis=0)
    itm2inv = XvecP.to_dense().T.matmul(XvecP.to_dense())
    itm2 = torch.inverse(itm2inv)
    betaHat = itm1.matmul(itm2)
    return betaHat


# calaulte the updated bTheta under MAR for linear
# I didn't check it before. 
# the theta is before soft-thresholding
def marLinearUpdateTheta(X, Y, R, bThetaOld, beta):
    bThetaNew = bThetaOld.clone()
    betaX = X.to_dense().matmul(beta)
    YvecP = Y[R.to_dense().bool()] # M 
    betaXvecP = betaX[R.to_dense().bool()]
    bThetaNewVecP = YvecP  - betaXvecP
    bThetaNew[R.to_dense().bool()] = bThetaNewVecP
    return bThetaNew


# calculate the loss under MAR  andl linear setting
# unchecked
def marLinearLossL(X, Y, R, bTheta, beta, sigma):
    n, m = Y.shape
    YvecP = Y[R.to_dense().bool()] # M 
    bThetaVecP = bTheta[R.to_dense().bool()] # M 
    betaX = X.to_dense().matmul(beta)
    betaXvecP = betaX[R.to_dense().bool()]
    res = torch.sum(-(YvecP - bThetaVecP - betaXvecP)**2/2/sigma**2)/n/m
    return res


# ## EM

# Compute the value of L w.r.t beta for any distributions X under EM setting
def emLossL(bTheta, beta, f, X, Y, is_logf=False):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    f: likelihood function of Y|X
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    
    """
    if X.is_sparse:
        X = X.to_dense()
        
    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    if is_logf:
        itm1 = f(Y, TbX)
    else:
        itm1 = torch.log(f(Y, TbX)+seps)

    itm = itm1
    return -itm.mean(dim=[0, 1])


#----------------------------------------------------------------------------------------------------------------
# Compute the value of first derivative of L w.r.t beta for any distributions X under EM setting
def emLossLpb(bTheta, beta, conDenfs, X, Y):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    """
    if X.is_sparse:
        X = X.to_dense()
    f, f2, _ = conDenfs
    # remove the elements whose corresponding betaK is zero
    idxNon0 = torch.nonzero(beta).view(-1)
    if idxNon0.shape[0] == 0:
        idxNon0 = torch.tensor([0, 1], dtype=torch.int64)
    betaNon0 = beta[idxNon0]
    XNon0 = X[:, :, idxNon0]

    betaX = torch.matmul(XNon0, betaNon0) # n x m
    del XNon0
    TbX = bTheta + betaX # n x m

    itm = (f2(Y, TbX)/(f(Y, TbX)+seps)).unsqueeze(dim=2) * X # n x m x p 
    del X

    return -itm.mean(dim=[0, 1])


# Compute the value of first derivative of L w.r.t bTheta for any distributions X under EM method
def emLossLpT(bTheta, beta, conDenfs, X, Y):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    """
    n, m, p = X.shape
    f, f2, _ = conDenfs

    if X.is_sparse:
        X = X.to_dense()
    # remove the elements whose corresponding betaK is zero
    idxNon0 = torch.nonzero(beta).view(-1)
    if idxNon0.shape[0] == 0:
        idxNon0 = torch.tensor([0, 1], dtype=torch.int64)
    beta = beta[idxNon0]
    X = X[:, :, idxNon0]

    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    itm1 = (f2(Y, TbX)/(f(Y, TbX)+seps))

    itm = itm1/(m*n)
    return -itm


# ## Misc

# To compute the Loss value with penalties. 
# i.e. L + Lambda_T ||\bTheta|| + Lambda_b ||\beta||
def LossWP(Lv, bThetaS, beta, LamT, Lamb):
    if len(bThetaS.shape) == 1:
        itm2 = LamT * bThetaS.abs().sum()
    else:
        itm2 = LamT * torch.norm(bThetaS, p="nuc")
    itm3 = Lamb * beta.abs().sum()
    return Lv + itm2 + itm3


# To generate bTheta_0, (Grand-truth of bTheta)
def genbTheta(n, m, rank=None, sigVs=None):
    bTheta = torch.randn(n, m) * 7
    if rank is None:
        rank = len(sigVs)
    U, S, V = torch.svd_lowrank(bTheta)
    idx = torch.randperm(S.shape[0])[:rank]
    if sigVs is not None:
        sigVs = torch.tensor(sigVs, dtype=dtorchdtype)
        bTheta = U[:, :rank].matmul(torch.diag(sigVs)).matmul(V[:, :rank].transpose(1, 0))
    else:
        bTheta = U[:, idx].matmul(torch.diag(torch.ones(rank)*16)).matmul(V[:, idx].transpose(1, 0))
    return bTheta 


# To compute the Lambda_bTheta
# just constant before the penalty item of bTheta
def LamTfn(C, n, m, p):
    d = np.sqrt(m*n)
    rawvs = [np.sqrt(np.log(d)/d), (np.log(p))**(1/4)/np.sqrt(d)]
    rawv = np.max(rawvs)
    return torch.tensor([C*rawv], dtype=dtorchdtype)


# To compute the Lambda_beta
# just constant before the penalty item of beta
def Lambfn(C, n, m):
    rawv = np.sqrt(np.log(m+n))/m/n
    return torch.tensor([C*rawv], dtype=dtorchdtype)


# Just the \rho function in optimization algorithm
def SoftTO(x, a):
    rx = torch.zeros(x.shape)
    idx1 = x > a
    idx2 = x < -a
    rx[idx1] = x[idx1] - a
    rx[idx2] = x[idx2] + a 
    return rx


# Generatae Y when Y|X \sim N(m, sigma**2)
def genYnorm(X, bTheta, beta, sigma=0.1): 
    n, m, _ = X.shape
    M = bTheta + X.to_dense().matmul(beta)
    Y = torch.randn(n, m)*sigma + M
    return Y


# generatae Y when Y|X  is logistic
def genYlogit(X, bTheta, beta):
    M = bTheta + X.matmul(beta)
    Marr = M.cpu().numpy()
    probMarr = 1/(1+np.exp(-Marr))
    Yarr = np.random.binomial(1, probMarr)
    return torch.tensor(Yarr).to(dtorchdtype)


# generate missing matrix R under linear and quadratic relation.
def genR(Y, typ="Linear", a=2, b=0.4, slop=5, inp=6.5, is_sparse=True):
    typ = typ.lower()
    if "linear".startswith(typ):
        #torch.manual_seed(10)
        #torch.cuda.manual_seed_all(10)
        Thre = slop*Y - inp
        probs = Normal(0, 1).cdf(Thre)
        ranUnif = torch.rand_like(probs)
        R = probs <= ranUnif
        
    elif "quadratic".startswith(typ):
        Thre = Y**2 + a*Y + b
        probs = Normal(0, 1).cdf(Thre)
        ranUnif = torch.rand_like(probs)
        R = probs <= ranUnif
    elif "fixed".startswith(typ):
        probs = torch.zeros(Y.shape)
        probs[Y==1] = 0.05
        probs[Y==0] = 0.65
        ranUnif = torch.rand_like(probs)
        R = probs <= ranUnif
    elif "mar".startswith(typ):
        probs = torch.zeros(Y.shape) + 0.25
        ranUnif = torch.rand_like(probs)
        R = probs <= ranUnif
    else:
        raise TypeError("Wrong dependence typ!")
    if is_sparse:
        return R.to(dtorchdtype).to_sparse()
    else:
        return R.to(dtorchdtype)


# Generate X from Bernoulli distribution
def genXBin(*args, prob=0.1, is_sparse=True):
    assert len(args) in [2, 3]
    p, size = args[-1], args[:-1]
    X = npr.uniform(0, 1, args)
    idx0, idx1 = X>=prob, X<prob
    X[idx0] = 0 
    X[idx1] = 1
    if len(args) == 2:
        X = X.transpose()
    if is_sparse:
        return torch.tensor(X, device="cpu").to(dtorchdtype).to_sparse().cuda()
    else:
        return torch.tensor(X).to(dtorchdtype)
