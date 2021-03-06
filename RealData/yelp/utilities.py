import torch
import numpy as np
import numpy.random as npr
from torch.distributions.normal import Normal
from prettytable import PrettyTable
from scipy.stats import truncnorm
from tqdm import tqdm

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

# This file contains the functions for main simulation.
#
#

#----------------------------------------------------------------------------------------------------------------

# seps: small number to avoid zero in log funciton and denominator. 
seps = 1e-15
# dtorchdtype and dnpdtype are default data types used for torch package and numpy package, respectively.
dtorchdtype = torch.float64
dnpdtype = np.float64

#----------------------------------------------------------------------------------------------------------------


# To compute the H_2 value when Y|X is normal or truncated normal
def H2fnorm(y, m, sigma=0.5):
    n, m = y.shape
    return -torch.ones(n, m) /sigma**2

# To compute the S_2 value when Y|X is normal or truncated normal
def S2fnorm(y, m, sigma):
    return (y-m)/sigma**2

# To compute the H_2 value when Y|X is logistic
def H2flogit(y, m):
    return torch.exp(m)/(1+torch.exp(m))**2

# To compute the S_2 value when Y|X is logistic
def S2flogit(y, m):
    return torch.exp(m)/(1 + torch.exp(m)) - y


# To compute the lower bounded of Sigma_bTheta when Y|X is logistic
def Dshlowerflogit(Y, X, beta, bTheta):
    m = bTheta + X.matmul(beta)
    H2v = H2flogit(Y, m)
    S2v = S2flogit(Y, m)
    Ds2 = S2v.abs().max().item()
    Dh2 = H2v.abs().max().item()
    return Ds2, Dh2

# To compute the lower bounded of Sigma_bTheta when Y|X is normal/ truncated normal
def Dshlowerfnorm(Y, X, beta, bTheta, sigma=0.5):
    m = bTheta + X.matmul(beta)
    H2v = H2fnorm(Y, m, sigma)
    S2v = S2fnorm(Y, m, sigma)
    Ds2 = S2v.abs().max().item()
    Dh2 = H2v.abs().max().item()
    return Ds2, Dh2

#----------------------------------------------------------------------------------------------------------------


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
def intBernh(f, bTheta, beta, Y, prob):
    idxNon0 = torch.nonzero(beta).view(-1)
    s = (beta != 0).sum().to(dtorchdtype).item()
    s = int(s)
    if s != 0:
        Blistmat = torch.tensor(Blist(s))
        beta = beta[idxNon0]
        Blstsum = Blistmat.sum(1)
        sum1 = f(Y, bTheta, Blistmat.matmul(beta)) # n x m x 2^s
        sum2 = prob**Blstsum * (1-prob)**(s-Blstsum) # 2^s
        summ = sum1 * sum2
        return summ.sum(-1)
    else:
        summ = f(Y, bTheta)
        return summ


# Compute the exact integration of \int f(Y, bTheta + beta\trans X ) X g(X) dX when beta is sparse
def intBernhX(f, bTheta, beta, Y, prob):
    p = beta.shape[0]
    idxNon0 = torch.nonzero(beta).view(-1)
    s = (beta != 0).sum().to(dtorchdtype).item()
    s = int(s)
    if s != 0:
        Blistmat = torch.tensor(Blist(s))
        beta = beta[idxNon0]
        Blstsum = Blistmat.sum(1)
        sum2 = prob**Blstsum * (1-prob)**(s-Blstsum) # 2^s
        summ = []
        for ii in range(p):
            if ii not in idxNon0:
                sump = (f(Y, bTheta, Blistmat.matmul(beta))* prob * sum2).sum(-1) #n x m 
                summ.append(sump)
            else:
                idx = torch.nonzero(idxNon0 == ii).view(-1)
                Xc = Blistmat[:, idx].squeeze()
                sump = (f(Y, bTheta, Blistmat.matmul(beta))* Xc* sum2).sum(-1) #n x m 
                summ.append(sump)
        return torch.stack(summ, dim=-1) 
    else:
        expX = torch.ones(p) * prob
        return f(Y, bTheta).unsqueeze(-1) * expX

#----------------------------------------------------------------------------------------------------------------


# Compute the value of first derivative of L w.r.t bTheta with MCMC method for any distributions X.
def missdepLpT(bTheta, beta, conDenfs, X, Y, R, sXs):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    sXs: p x N, samples of X_ij to compute the MCMC integration
    """
    n, m, p = X.shape
    _, N = sXs.shape
    f, f2, f22 = conDenfs

    # remove the elements whose corresponding betaK is zero
    idxNon0 = torch.nonzero(beta).view(-1)
    if idxNon0.shape[0] == 0:
        idxNon0 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    beta = beta[idxNon0]
    sXs = sXs[idxNon0]
    X = X[:, :, idxNon0]

    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    itm1 = (f2(Y, TbX)/(f(Y, TbX)+seps))

    bsXs = beta.matmul(sXs)
   #  TbsXs = bTheta.unsqueeze(dim=-1) + bsXs
   #  Ym = Y.unsqueeze(dim=-1) + torch.zeros(N)
    
    itm2den = f(Y, bTheta, bsXs).mean(dim=-1) + seps
    itm2num = f2(Y, bTheta, bsXs).mean(dim=-1)
    itm2 = itm2num/itm2den

    itm = R * (itm1 - itm2)/(m*n)
    return -itm


# the loop version of function missdepLpT. 
# because the missdepLpT use all array computation which need too much memory.
def missdepLpTLoop(bTheta, beta, conDenfs, X, Y, R, sXs):
    n, m, p = X.shape
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    sXs: p x N, samples of X_ij to compute the MCMC integration
    """
    _, N = sXs.shape
    f, f2, f22 = conDenfs
    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    itm1 = (f2(Y, TbX)/(f(Y, TbX)+seps))

    results = itm1.clone()
    for ii in range(n):
        for jj in range(m):
            Yij = Y[ii, jj]
            bThetaij = bTheta[ii, jj]
            incre = torch.zeros(1)
            if R[ii, jj]:
                arg1 = Yij + torch.zeros(N)
                arg2 = bThetaij + beta.matmul(sXs)
                itm2ijden = f(arg1, arg2).mean() + seps
                itm2ijnum = f2(arg1, arg2).mean()
                incre = itm2ijnum/itm2ijden
            results[ii, jj] = R[ii, jj] * results[ii, jj] - incre

    return -results/m/n


# compute the exact value of first derivative of L w.r.t bTheta when X is normal and Y|X is normal
def LpTnormal(bTheta, beta,  X, Y, R, sigmax=0.5, sigma=0.5):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    sigmax: the std of Y|X
    sigma: the std of each entry of X
    """
    sigmax2, sigma2 = sigmax**2, sigma**2
    n, m, p = X.shape
    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    itm1 = (Y-TbX)/sigma2

    Sigma = torch.eye(p) * sigmax2
    SigmaI = torch.eye(p)/sigmax2
    SigmaNI = SigmaI + beta * beta.unsqueeze(-1)/sigma2
    SigmaN = sigmax2*(torch.eye(p) - sigmax2*beta*beta.unsqueeze(-1)/(sigma2+sigmax2*beta.matmul(beta)))
    muN = ((Y - bTheta).unsqueeze(-1) * beta/sigma2).matmul(SigmaN) # n x m x p
    itm2 = Y-bTheta- muN.matmul(beta).squeeze()
    itm2 = itm2/sigma2

    itm = R * (itm1 - itm2)/(m*n)
    return -itm


# compute the exact value of first derivative of L w.r.t bTheta when X is Bernoulli.
def LpTBern(bTheta, beta, conDenfs, X, Y, R, prob=0.5):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    prob: the successful probability of each entry of X
    """
    n, m, p = X.shape
    f, f2, f22 = conDenfs

    itm2den = intBernh(f, bTheta, beta, Y, prob) + seps
    itm2num = intBernh(f2, bTheta, beta, Y, prob)
    itm2 = itm2num/itm2den
    # remove the elements whose corresponding betaK is zero
    idxNon0 = torch.nonzero(beta).view(-1)
    if idxNon0.shape[0] == 0:
        idxNon0 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    beta = beta[idxNon0]
    X = X[:, :, idxNon0]

    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    itm1 = (f2(Y, TbX)/(f(Y, TbX)+seps))
    
    itm = R * (itm1 - itm2)/(m*n)
    return -itm


#----------------------------------------------------------------------------------------------------------------
# Compute the value of second derivative of L w.r.t vec(bTheta) with MCMC method for any distributions X.
def missdepLpTT(bTheta, beta, conDenfs, X, Y, R, sXs):
    """
    Input:
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. 
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    sXs: p x N, samples of X_ij to compute the MCMC integration
    Output:
    itm: n x m
    """
    n, m, p = X.shape
    _, N = sXs.shape
    f, f2, f22 = conDenfs

    # remove the elements whose corresponding betaK is zero
    idxNon0 = torch.nonzero(beta).view(-1)
    if idxNon0.shape[0] == 0:
        idxNon0 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    beta = beta[idxNon0]
    sXs = sXs[idxNon0]
    X = X[:, :, idxNon0]

    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    itm1 = (f22(Y, TbX)/(f(Y, TbX)+seps))
    itm2 = (f2(Y, TbX)**2/(f(Y, TbX)**2+seps))

    bsXs = beta.matmul(sXs)
    
    itm3den = f(Y, bTheta, bsXs).mean(dim=-1) + seps
    itm3num = f22(Y, bTheta, bsXs).mean(dim=-1)
    itm3 = itm3num/itm3den

    itm4den = (f(Y, bTheta, bsXs).mean(dim=-1))**2 + seps
    itm4num = f2(Y, bTheta, bsXs).mean(dim=-1)**2
    itm4 = itm4num/itm4den

    itm = R * (itm1 - itm2- itm3 + itm4)/(m*n)
    return -itm


# compute the exact value of second derivative of L w.r.t vec(bTheta) when X is Bernoulli.
def LpTTBern(bTheta, beta, conDenfs, X, Y, R, prob=0.5):
    """
    Inputs:
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22].
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    prob: the successful probability of each entry of X

    Output:
    itm: n x m
    """
    n, m, p = X.shape
    f, f2, f22 = conDenfs

    itm3den = intBernh(f, bTheta, beta, Y, prob) + seps
    itm3num = intBernh(f22, bTheta, beta, Y, prob)
    itm3 = itm3num/itm3den

    itm4den = intBernh(f, bTheta, beta, Y, prob)**2 + seps
    itm4num = intBernh(f2, bTheta, beta, Y, prob)**2
    itm4 = itm4num/itm4den

    # remove the elements whose corresponding betaK is zero
    idxNon0 = torch.nonzero(beta).view(-1)
    if idxNon0.shape[0] == 0:
        idxNon0 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    beta = beta[idxNon0]
    X = X[:, :, idxNon0]

    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    itm1 = (f22(Y, TbX)/(f(Y, TbX)+seps))
    itm2 = (f2(Y, TbX)**2/(f(Y, TbX)**2+seps))
    
    itm = R * (itm1 - itm2 - itm3 + itm4)/(m*n)
    return -itm

#----------------------------------------------------------------------------------------------------------------
# Compute the value of first derivative of L w.r.t beta with MCMC method for any distributions X.
def missdepLpb(bTheta, beta, conDenfs, X, Y, R, sXs):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    sXs: p x N, samples of X_ij to compute the MCMC integration
    """
    p, N = sXs.shape
    f, f2, _ = conDenfs
    # remove the elements whose corresponding betaK is zero
    idxNon0 = torch.nonzero(beta).view(-1)
    if idxNon0.shape[0] == 0:
        idxNon0 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    betaNon0 = beta[idxNon0]
    sXsNon0 = sXs[idxNon0]
    XNon0 = X[:, :, idxNon0]

    betaX = torch.matmul(XNon0, betaNon0) # n x m
    del XNon0
    TbX = bTheta + betaX # n x m

    itm1 = (f2(Y, TbX)/(f(Y, TbX)+seps)).unsqueeze(dim=2) * X # n x m x p 
    del X

    bsXs = betaNon0.matmul(sXsNon0) # N
    # TbsXs = bTheta.unsqueeze(dim=-1) + bsXs # n x m x N
    # Ym = Y.unsqueeze(dim=-1) + torch.zeros(N) # n x m x N
    
    itm2den = f(Y, bTheta, bsXs).mean(dim=-1) + seps
#    itm2num = (f2(Ym, TbsXs).unsqueeze(dim=-2) * sXs).mean(dim=-1)
    itm2numin =  f2(Y, bTheta, bsXs) # n x m x N 
    itm2num = torch.stack([(itm2numin*sX).mean(dim=-1) for sX in sXs], dim=-1)

    itm2 = itm2num/itm2den.unsqueeze(dim=-1)

    itm = R.unsqueeze(dim=2) * (itm1 - itm2)
    return -itm.mean(dim=[0, 1])


# the loop version of function missdepLpb. 
# because the missdepLpb uses all array computation which needs too much memory.
def missdepLpbLoop(bTheta, beta, conDenfs, X, Y, R, sXs):
    """
    sXs: p x N, samples of X_ij to compute the MCMC integration
    """
    n, m, p = X.shape
    _, N = sXs.shape
    f, f2, _ = conDenfs

    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    itm1 = (f2(Y, TbX)/(f(Y, TbX)+seps)).unsqueeze(dim=2) * X

    results = torch.zeros(p)
    for ii in range(n):
        for jj in range(m):
            Yij = Y[ii, jj]
            bThetaij = bTheta[ii, jj]
            incre = torch.zeros(p)
            if R[ii, jj]:
                itm1ij = itm1[ii, jj]
                arg1 = Yij + torch.zeros(N)
                arg2 = bThetaij + beta.matmul(sXs)
                itm2ijden = f(arg1, arg2).mean() + seps
                itm2ijnum = (f2(arg1, arg2) * sXs).mean(dim=-1)
                itm2ij = itm2ijnum/itm2ijden
                incre = itm1ij - itm2ij
            results = results + incre

    return -results/m/n




# Compute the exact value of first derivative of L w.r.t beta when X is normal and Y|X is normal 
def Lpbnormal(bTheta, beta, X, Y, R, sigma=0.5, sigmax=0.5):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    sigmax: the std of Y|X
    sigma: the std of each entry of X
    """
    sigmax2, sigma2 = sigmax**2, sigma**2
    n, m, p = X.shape
    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    itm1 = (Y-TbX).unsqueeze(dim=2) * X/sigma2

    Sigma = torch.eye(p) * sigmax2
    SigmaI = torch.eye(p)/sigmax2
    SigmaNI = SigmaI + beta * beta.unsqueeze(-1)/sigma2
    SigmaN = sigmax2*(torch.eye(p) - sigmax2*beta*beta.unsqueeze(-1)/(sigma2+sigmax2*beta.matmul(beta)))
    muN = ((Y - bTheta).unsqueeze(-1) * beta/sigma2).matmul(SigmaN) # n x m x p

    itm21 = (Y-bTheta).unsqueeze(-1) * muN
    itm22 = (SigmaN.unsqueeze(0).unsqueeze(0) + muN.unsqueeze(-1) * muN.unsqueeze(-2)).matmul(beta)
    itm2 = (itm21-itm22)/sigma2

    itm = R.unsqueeze(dim=2) * (itm1 - itm2)
    return -itm.mean(dim=[0, 1])


# Compute the exact value of first derivative of L w.r.t beta when X is  Bernoulli
def LpbBern(bTheta, beta, conDenfs, X, Y, R, prob=0.5):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    prob: the successful probability of each entry of X
    """
    _, _, p = X.shape 
    f, f2, _ = conDenfs
    # remove the elements whose corresponding betaK is zero
    idxNon0 = torch.nonzero(beta).view(-1)
    if idxNon0.shape[0] == 0:
        idxNon0 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    betaNon0 = beta[idxNon0]
    XNon0 = X[:, :, idxNon0]

    betaX = torch.matmul(XNon0, betaNon0) # n x m
    TbX = bTheta + betaX # n x m

    itm1 = (f2(Y, TbX)/(f(Y, TbX)+seps)).unsqueeze(dim=2) * X # n x m x p 
    del X
    
    itm2den = intBernh(f, bTheta, beta, Y, prob) + seps
    itm2num = intBernhX(f2, bTheta, beta, Y, prob)

    itm2 = itm2num/itm2den.unsqueeze(dim=-1)

    itm = R.unsqueeze(dim=2) * (itm1 - itm2)
    return -itm.mean(dim=[0, 1])


#----------------------------------------------------------------------------------------------------------------

# Compute the value of second derivative of L w.r.t beta with MCMC method for any distributions X.
# not used.
# Loop version.
def missdepLpbbLoop(bTheta, beta, conDenfs, X, Y, R, sXs):
    n, m, p = X.shape
    _, N = sXs.shape
    f, f2, f22 = conDenfs
    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    itm1 = f22(Y, TbX)/(f(Y, TbX)+seps) - f2(Y, TbX)**2/(f(Y, TbX)**2 + seps)
    itm1 = itm1.unsqueeze(dim=-1).unsqueeze(dim=-1) * (X.unsqueeze(dim=-1) + X.unsqueeze(dim=-2))

    results = torch.zeros(p, p)
    for ii in range(n):
        for jj in range(m):
            Yij = Y[ii, jj]
            bThetaij = bTheta[ii, jj]
            incre = torch.zeros(p, p)
            if R[ii, jj]:
                itm1ij = itm1[ii, jj]
                arg1 = Yij + torch.zeros(N)
                arg2 = bThetaij + beta.matmul(sXs)
                itm2ijdenb = f(arg1, arg2).mean() + seps
                itm2ijnum1 = (f22(arg1, arg2) * (sXs.unsqueeze(dim=0) + sXs.unsqueeze(dim=1))).mean(dim=-1)
                itm2ijnum2 = (f2(arg1, arg2) * sXs).mean(dim=-1)
                itm2ijnum2 = itm2ijnum2.unsqueeze(dim=1) + itm2ijnum2.unsqueeze(dim=0)
                itm2ij1 = itm2ijnum1/itm2ijdenb
                itm2ij2 = itm2ijnum2/itm2ijdenb**2
                itm2ij = itm2ij1 - itm2ij2
                incre = itm1ij - itm2ij
            results = results + incre
    return -results/m/n


#----------------------------------------------------------------------------------------------------------------
# Compute the value of L with MCMC method for any distributions X.
def missdepL(bTheta, beta, f, X, Y, R, sXs):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    f: likelihood function of Y|X
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    sXs: p x N, samples of X_ij to compute the MCMC integration
    """
    _, N = sXs.shape
    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    itm1 = torch.log(f(Y, TbX)+seps)

    bsXs = beta.matmul(sXs)
    # TbsXs = bTheta.unsqueeze(dim=-1) + bsXs
    # Ym = Y.unsqueeze(dim=-1) + torch.zeros(N)
    
    itm2 = torch.log(f(Y, bTheta, bsXs).mean(dim=-1)+seps)

    itm = R * (itm1 - itm2)
    return -itm.mean(dim=[0, 1])


# Loop version of missdepL
def missdepLLoop(bTheta, beta, f, X, Y, R, sXs):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    f: likelihood function of Y|X
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    sXs: p x N, samples of X_ij to compute the MCMC integration
    """
    _, N = sXs.shape
    n, m, p = X.shape
    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    itm1 = torch.log(f(Y, TbX)+seps)

    results = torch.zeros(1)
    for ii in range(n):
        for jj in range(m):
            Yij = Y[ii, jj]
            bThetaij = bTheta[ii, jj]
            incre = torch.zeros(1)
            if R[ii, jj]:
                itm1ij = itm1[ii, jj]
                arg1 = Yij + torch.zeros(N)
                arg2 = bThetaij + beta.matmul(sXs)
                itm2ijexp  = f(arg1, arg2).mean() + seps
                incre = itm1ij - torch.log(itm2ijexp)
            results = results + incre

    return -results/m/n



# Compute the exact value of L when X is normal and  Y|X is normal
def Lnormal(bTheta, beta,  X, Y, R, sigmax=0.5, sigma=0.5):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    sigmax: the std of Y|X
    sigma: the std of each entry of X
    """
    sigmax2, sigma2 = sigmax**2, sigma**2
    n, m, p = X.shape
    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    Sigma = torch.eye(p) * sigmax2
    SigmaI = torch.eye(p)/sigmax2
    SigmaNI = SigmaI + beta * beta.unsqueeze(-1)/sigma2
    SigmaN = sigmax2*(torch.eye(p) - sigmax2*beta*beta.unsqueeze(-1)/(sigma2+sigmax2*beta.matmul(beta)))
    muN = ((Y - bTheta).unsqueeze(-1) * beta/sigma2).matmul(SigmaN) # n x m x p

    C = - 0.5 * torch.log(1 + sigmax2*beta.matmul(beta)/sigma2)
    itm2 = 0.5 * ((Y - bTheta).unsqueeze(-1) * beta/sigma2).unsqueeze(-2).matmul(muN.unsqueeze(-1)).squeeze() # n x m 
    itm1 = betaX * (2*Y - bTheta - TbX)/2/sigma2
    itm = R * (itm1-itm2-C)

    return -itm.mean(dim=[0, 1])


# Compute the exact value of L when X is Bernoulli
def LBern(bTheta, beta, f, X, Y, R, prob=0.5):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    f :  the likelihood function of Y|X
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    prob: the successful probability of each entry of X
    """
    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    itm1 = torch.log(f(Y, TbX)+seps)

    itm2 = torch.log(intBernh(f, bTheta, beta, Y, prob)+seps)

    itm = R * (itm1 - itm2)
    return -itm.mean(dim=[0, 1])

#----------------------------------------------------------------------------------------------------------------

# Just the \rho function in optimization algorithm
def SoftTO(x, a):
    rx = torch.zeros(x.shape)
    idx1 = x > a
    idx2 = x < -a
    rx[idx1] = x[idx1] - a
    rx[idx2] = x[idx2] + a 
    return rx


# To compute the Loss value with penalties. 
# i.e. L + Lambda_T ||\bTheta|| + Lambda_b ||\beta||
def missdepLR(Lv, bTheta, beta, LamT, Lamb):
    itm2 = LamT * torch.norm(bTheta, p="nuc")
    itm3 = Lamb * beta.abs().sum()
    return Lv + itm2 + itm3


# Just the omegat function in optimization algorithm
def omegat(bT, tbTt, R, LamT, LpTv, Rb, tRb, ST):
    num = ((bT-tbTt).t().matmul(LpTv)).trace() + LamT*(Rb-tRb)
    den = ST * ((bT-tbTt)**2 * R).mean() + seps
    itm = num/den
    if itm > 1:
        return torch.tensor([1.0]) 
    else:
        return itm

# To compute R^t_{UB} in optimization algorithm
# i.e. F/Lambda_bTheta
def Rub(Lv, beta, Rb, LamT, Lamb, Lcon=10):
    Fv = Lv + Lamb * beta.abs().sum() + LamT * Rb + Lcon
    return Fv/LamT


# To compute the Lambda_beta
# just constant before the penalty item of beta
def Lambfn(C, n, m):
    rawv = np.sqrt(np.log(m+n))/m/n
    return torch.tensor([C*rawv], dtype=dtorchdtype)


# To compute the Lambda_bTheta
# just constant before the penalty item of bTheta
def LamTfn(C, n, m, p):
    d = np.sqrt(m*n)
    rawvs = [np.sqrt(np.log(d)/d), (np.log(p))**(1/4)/np.sqrt(d)]
    rawv = np.max(rawvs)
    return torch.tensor([C*rawv], dtype=dtorchdtype)


#----------------------------------------------------------------------------------------------------------------
# To generate X (n x m x p) following that NIPS paper. Never used
def genX(n, m, p):
    X = torch.zeros(n, m, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                setv = list(range(5*(k-1)+1, 5*k+1))
                if ((j-1)*(n)+i) in setv:
                    X[i, j, k] = 1
    return X


# To generate X with different distributions.
def genXdis(*args, type="mvnorm", sigmax=0.5, prob=None):
    assert len(args) in [2, 3]
    p, size = args[-1], args[:-1]
    type = type.lower()
    if "mvnorm".startswith(type):
        X = npr.multivariate_normal(np.zeros(p), sigmax**2*np.eye(p), size=size)
    elif "multinomial".startswith(type):
        X = npr.multinomial(3, np.ones(p)/p, size=size)
    elif "uniform".startswith(type):
        X = npr.uniform(size=args) * 2/p - 1/p
    elif "bernoulli".startswith(type):
        assert prob is not None, "You should provide a probability parameter!" 
        X = npr.uniform(0, 1, args)
        idx0, idx1 = X>=prob, X<prob
        X[idx0] = 0 
        X[idx1] = 1
    else:
        raise TypeError("No such type of X!")
    if len(args) == 2:
        X = X.transpose()
    return torch.tensor(X).to(dtorchdtype)


# To generate bTheta_0, (Grand-truth of bTheta)
def genbTheta(n, m, rank=None, sigVs=None):
    bTheta = torch.randn(n, m) * 7
    if rank is None:
        rank = len(sigVs)
    U, S, V = torch.svd(bTheta)
    idx = torch.randperm(S.shape[0])[:rank]
    if sigVs is not None:
        sigVs = torch.tensor(sigVs, dtype=dtorchdtype)
        bTheta = U[:, :rank].matmul(torch.diag(sigVs)).matmul(V[:, :rank].transpose(1, 0))
    else:
        bTheta = U[:, idx].matmul(torch.diag(torch.ones(rank)*16)).matmul(V[:, idx].transpose(1, 0))
    return bTheta 

# To generate beta_0, (Grand-truth of beta), never used
def genbeta(p, sparsity=0.1):
    zeroidx = torch.rand(p) > sparsity
    beta = torch.rand(p)
    beta[zeroidx] = 0
    return beta


# generatae Y when Y|X \sim N(m, sigma**2)
def genYnorm(X, bTheta, beta, sigma=0.1): 
    n, m, _ = X.shape
    M = bTheta + X.matmul(beta)
    Y = torch.randn(n, m)*sigma + M
    return Y


# generatae Y when Y|X  is truncated normal
def genYtnorm(X, bTheta, beta, a, b, sigma=0.1):
    n, m, _ = X.shape
    M = bTheta + X.matmul(beta)
    Marr = M.cpu().numpy()
    a = a/sigma
    b = b/sigma
    Yarr = truncnorm.rvs(a, b, loc=Marr, scale=sigma)
    return torch.tensor(Yarr).to(dtorchdtype)


# generatae Y when Y|X  is logistic
def genYlogit(X, bTheta, beta):
    M = bTheta + X.matmul(beta)
    Marr = M.cpu().numpy()
    probMarr = 1/(1+np.exp(-Marr))
    Yarr = np.random.binomial(1, probMarr)
    return torch.tensor(Yarr).to(dtorchdtype)


# generate missing matrix R under linear and quadratic relation.
def genR(Y, type="Linear", a=2, b=0.4, inp=6.5):
    type = type.lower()
    if "linear".startswith(type):
        Thre = 5*Y  - inp#- 8 #- 1/2 # -  7 # -1/2 #+2
        probs = Normal(0, 1).cdf(Thre)
        ranUnif = torch.rand_like(probs)
        R = probs <= ranUnif
    elif "quadratic".startswith(type):
        Thre = Y**2 + a*Y + b
        probs = Normal(0, 1).cdf(Thre)
        ranUnif = torch.rand_like(probs)
        R = probs <= ranUnif
    elif "fixed".startswith(type):
        probs = torch.zeros(Y.shape)
        probs[Y==1] = 0.05
        probs[Y==0] = 0.65
        ranUnif = torch.rand_like(probs)
        R = probs <= ranUnif
    elif "mar".startswith(type):
        probs = torch.zeros(Y.shape) + 0.25
        ranUnif = torch.rand_like(probs)
        R = probs <= ranUnif
    else:
        raise TypeError("Wrong dependence type!")
    return R.to(dtorchdtype)


#----------------------------------------------------------------------------------------------------------------
# To compute the difference of beta and bTheta in 2 consecutive iterations
# as ai termination criteria
def ParaDiff(Olds, News):
    errs = torch.tensor([torch.norm(Olds[i]-News[i]) for i in range(2)])
    return errs.max()



# function to optimize on realdata 
def RealDataAlg(MaxIters, X, Y, R, sXs, conDenfs, Cb=10, CT=1, log=0, bThetainit=None, betainit=None, tols=None, ErrOpts=0, etab=0.05, etaT=0.05):
    """
    MaxIters: max iteration number.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    sXs: sample of X for MCMC, p x N
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    Cb: the constant of Lambda_beta
    CT: the constant of Lambda_bTheta
    log: Whether output detail training log. 0 not output, 1 output simple training log, 2 output detailed training log.
    bThetainit: initial value of bTheta
    tol: terminate tolerace.
    ErrOpts: whether output errors of beta and bTheta. 0 no, 1 yes
    etabinit: The initial learning rate of beta
    etaTinit: The initial learning rate of btheta
    """
    n, m, p = X.shape
    f, f2, f22 = conDenfs
    tol, tolb, tolT = tols
    Lcon = -10
    # To contain the training errors of bTheta and beta, respectively.
    Likelis = []
    bThetahats = []
    betahats = []

    # Initial the value of beta, bTheta and R_b
    bThetaOld = torch.rand(n, m) if bThetainit is None else bThetainit
    betaOld = torch.rand(p) if betainit is None else betainit
    # the relative change of Loss, i.e. |L_k - L_k+1|/max(|L_k|, |L_k+1|, 1), here the L_k and L_k+1 are with penalty items.
    reCh = 1

    # Under Cb and CT, compute the Lambda_beta and Lambda_bTheta
    LamT = LamTfn(CT, n, m, p)
    Lamb = Lambfn(Cb, n, m)

    # The log output, nothing to do with algorithm.
    if log>=0:
        tb1 = PrettyTable(["Basic Value", "Lamb", "LamT"])
        tb1.add_row(["",  f"{Lamb.item():>5.3g}", f"{LamT.item():>5.3g}"])
        print(tb1)
    # The loss, i.e. L +  Lamdab_bTheta * ||bTheta||
    Losses = []

    # Starting optimizing.
    for t in tqdm(range(MaxIters), desc="MNAR"):
        #--------------------------------------------------------------------------------
        # To get the number of nonzeros entry in betaOld
        NumN0Old = p - (betaOld.abs()==0).sum().to(dtorchdtype)
        #--------------------------------------------------------------------------------
        # compute the loss function (with penalty items) under betaOld and bThetaOld

        # Compute L (without penalty items) 
        LvNow = missdepL(bThetaOld, betaOld, f, X, Y, R, sXs)
        # Add L with penalty items.
        LossNow = missdepLR(LvNow, bThetaOld, betaOld, LamT, Lamb)
        QOld = (LossNow - Lcon)/Lamb
        Losses.append(LossNow.item())

        #--------------------------------------------------------------------------------
        # This block is to update beta.
        LpbvOld = missdepLpb(bThetaOld, betaOld, conDenfs, X, Y, R, sXs)
        # compute the learning rate of beta
        etabOld = etab # 0.05 for linear setting
        betaNewRaw = betaOld - etabOld * LpbvOld
        # Using rho function to soften updated beta
        betaNew = SoftTO(betaNewRaw, etabOld*Lamb)

        #--------------------------------------------------------------------------------
        # To get the number of nonzeros entry in betaNew
        NumN0New = p - (betaNew.abs()==0).sum().to(dtorchdtype)

        #--------------------------------------------------------------------------------
        # Update bTheta 
        LpTvOld = missdepLpT(bThetaOld, betaNew, conDenfs, X, Y, R, sXs)
        LvNew = missdepL(bThetaOld, betaNew, f, X, Y, R, sXs)
        LossNew = missdepLR(LvNew, bThetaOld, betaNew, LamT, Lamb)
        ROld = (LossNew - Lcon)/LamT
        etaTOld = etaT # 0.05 for linear setting
        #tmpmatrix = bThetaOld-LpTvOld*etaTOld
        #tmpmatarr = tmpmatrix.cpu().numpy()
        #U, S, VT = np.linalg.svd(tmpmatarr)
        #U, S, VT = torch.tensor(U), torch.tensor(S), torch.tensor(VT)
        #V = VT.t()
        svdres = torch.svd(bThetaOld-LpTvOld*etaTOld)
        U, S, V =  svdres.U, svdres.S, svdres.V
        softS = (S-LamT*etaTOld).clamp_min(0)
        bThetaNew = U.matmul(torch.diag(softS)).matmul(V.t())

        #--------------------------------------------------------------------------------
        # compute the relative change of Loss
        if t >= 1:
            Lk1 = Losses[-1]
            Lk = Losses[-2]
            reCh = np.abs(Lk1-Lk)/np.max(np.abs((Lk, Lk1, 1))) 

        #--------------------------------------------------------------------------------
        # This block is for log output and Error save, nothing to do with the algorithm
        if ErrOpts:
            Likelis.append(LvNow.item())
            bThetahats.append(bThetaOld.norm().item())
            betahats.append(betaOld.norm().item())
        if log==1:
            tb2 = PrettyTable(["Iteration", "etaT", "Loss"])
            tb2.add_row([f"{t+1:>6}/{MaxIters}", f"{etaTOld:>8.3g}", f"{Losses[-1]:>8.3f}"])
            print(tb2)
        if log==2:
            tb2 = PrettyTable(["Iter", "etaT", "etab", "Loss", "-lkd", "reCh", "betat Norm", "Thetat Norm", "beta diff Norm", "btheta diff Norm", "betat L0 norm"])
            tb2.add_row([f"{t+1:>4}/{MaxIters}", f"{etaTOld:>3.3g}", f"{etabOld:>3.3g}", f"{Losses[-1]:>6.3f}", f"{LvNow.item():>2.1g}", f"{reCh:>6.4g}",  f"{betaNew.norm().item():>2.1f}", f"{bThetaNew.norm().item():>2.1f}", f"{(betaOld-betaNew).norm().item():>6.3g}", f"{(bThetaOld-bThetaNew).norm().item():>6.3g}", f"{NumN0New.item()}"])
            print(tb2)
        #--------------------------------------------------------------------------------
        # if reCh is smaller than tolerance, stop the loop
        if t >= 1:
            if (reCh < tol):
                break
        # if the difference of 2 consecutive bThetahat is smaller than tolerance, stop the loop
        if ((bThetaOld-bThetaNew).norm() < tolT) and ((betaOld-betaNew).norm() < tolb):
            break
        #--------------------------------------------------------------------------------
        # Change New to Old for starting next iteration
        #print(betaOld)
        #print(softS)
        betaOld, bThetaOld = betaNew, bThetaNew 
   #--------------------------------------------------------------------------------
    if ErrOpts:
        return betaOld, bThetaOld, t+1, betahats, bThetahats, Likelis
    else:
        return betaOld, bThetaOld, t+1


def YelpMissing(Yraw, OR=0.22, y1ratio=0.5):
    # get the R
    R = Yraw.copy()
    R[Yraw!=-1] = 1
    R[Yraw==-1] = 0

    rawOR = np.sum(R)/np.prod(R.shape)
    assert OR <= rawOR
    
    numrv = np.prod(R.shape) * (rawOR-OR)
    numrv1 = int(numrv*y1ratio)
    numrv0 = int(numrv) - numrv1

    mask1 = (Yraw > 3.5) & (R==1)
    mask0 = (Yraw < 3.5) & (R==1)
    selidx1 = np.random.choice(int(np.sum(mask1)), numrv1, replace=0)
    selidx0 = np.random.choice(int(np.sum(mask0)), numrv0, replace=0)

    tmp1 = R[mask1] 
    tmp1[selidx1] = 0
    R[mask1] = tmp1

    tmp0 = R[mask0] 
    tmp0[selidx0] = 0
    R[mask0] = tmp0

    return R

    

    
