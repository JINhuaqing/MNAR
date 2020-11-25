import torch
import numpy as np
import numpy.random as npr
from torch.distributions.normal import Normal
from prettytable import PrettyTable
from scipy.stats import truncnorm

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
    sXs = sXs.to_dense()[idxNon0].to_sparse()
    X = X.to_dense()[:, :, idxNon0].to_sparse()


    betaX = torch.matmul(X.to_dense(), beta)
    TbX = bTheta + betaX

    itm1 = (f2(Y, TbX)/(f(Y, TbX)+seps))

    bsXs = beta.matmul(sXs.to_dense())
   #  TbsXs = bTheta.unsqueeze(dim=-1) + bsXs
   #  Ym = Y.unsqueeze(dim=-1) + torch.zeros(N)
    
    torch.cuda.empty_cache()
    itm2den = f(Y, bTheta, bsXs).mean(dim=-1) + seps
    itm2num = f2(Y, bTheta, bsXs).mean(dim=-1)
    itm2 = itm2num/itm2den

    itm = R.to_dense() * (itm1 - itm2)/(m*n)
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
    X = X.to_dense()[:, :, idxNon0].to_sparse()

    betaX = torch.matmul(X.to_dense(), beta)
    TbX = bTheta + betaX

    itm1 = (f2(Y, TbX)/(f(Y, TbX)+seps))
    
    itm = R.to_dense() * (itm1 - itm2)/(m*n)
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
    sXs = sXs.to_dense()[idxNon0].to_sparse()
    X = X.to_dense()[:, :, idxNon0].to_sparse()

    betaX = torch.matmul(X.to_dense(), beta)
    TbX = bTheta + betaX

    itm1 = (f22(Y, TbX)/(f(Y, TbX)+seps))
    itm2 = (f2(Y, TbX)**2/(f(Y, TbX)**2+seps))

    bsXs = beta.matmul(sXs.to_dense())
    
    itm3den = f(Y, bTheta, bsXs).mean(dim=-1) + seps
    itm3num = f22(Y, bTheta, bsXs).mean(dim=-1)
    itm3 = itm3num/itm3den

    itm4den = (f(Y, bTheta, bsXs).mean(dim=-1))**2 + seps
    itm4num = f2(Y, bTheta, bsXs).mean(dim=-1)**2
    itm4 = itm4num/itm4den

    itm = R.to_dense() * (itm1 - itm2- itm3 + itm4)/(m*n)
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
    X = X.to_dense()[:, :, idxNon0].to_sparse()

    betaX = torch.matmul(X.to_dense(), beta)
    TbX = bTheta + betaX

    itm1 = (f22(Y, TbX)/(f(Y, TbX)+seps))
    itm2 = (f2(Y, TbX)**2/(f(Y, TbX)**2+seps))
    
    itm = R.to_dense() * (itm1 - itm2 - itm3 + itm4)/(m*n)
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
    sXsNon0 = sXs.to_dense()[idxNon0].to_sparse()
    XNon0 = X.to_dense()[:, :, idxNon0].to_sparse()

    betaX = torch.matmul(XNon0.to_dense(), betaNon0) # n x m
    del XNon0
    TbX = bTheta + betaX # n x m

    itm1 = ((f2(Y, TbX)/(f(Y, TbX)+seps)).unsqueeze(dim=2) * X.to_dense()).to_sparse() # n x m x p 
    del X

    bsXs = betaNon0.matmul(sXsNon0.to_dense()) # N
    # TbsXs = bTheta.unsqueeze(dim=-1) + bsXs # n x m x N
    # Ym = Y.unsqueeze(dim=-1) + torch.zeros(N) # n x m x N
    
    itm2den = (f(Y, bTheta, bsXs).mean(dim=-1) + seps)
#    itm2num = (f2(Ym, TbsXs).unsqueeze(dim=-2) * sXs).mean(dim=-1)
    torch.cuda.empty_cache()
    itm2numin =  f2(Y, bTheta, bsXs).to_sparse() # n x m x N 
    itm2num = torch.stack([(itm2numin.to_dense()*sX).mean(dim=-1) for sX in sXs.to_dense()], dim=-1).to_sparse() # n x m x p

    itm2 = (itm2num.to_dense()/itm2den.unsqueeze(dim=-1)).to_sparse()

    itm = (R.to_dense().unsqueeze(dim=2) * ((itm1 - itm2).to_dense())).to_sparse()
    return -itm.to_dense().mean(dim=[0, 1])




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
    XNon0 = X.to_dense()[:, :, idxNon0].to_sparse()

    betaX = torch.matmul(XNon0.to_dense(), betaNon0) # n x m
    TbX = bTheta + betaX # n x m


    itm1 = ((f2(Y, TbX)/(f(Y, TbX)+seps)).unsqueeze(dim=2) * X.to_dense()).to_sparse() # n x m x p 
    del X
    
    itm2den = intBernh(f, bTheta, beta, Y, prob) + seps
    itm2num = intBernhX(f2, bTheta, beta, Y, prob).to_sparse()

    itm2 = (itm2num.to_dense()/itm2den.unsqueeze(dim=-1)).to_sparse()

    del itm2den, itm2num, 
    torch.cuda.empty_cache()

    itm = (R.to_dense().unsqueeze(dim=2) * ((itm1 - itm2).to_dense())).to_sparse()
    rv = -itm.to_dense().mean(dim=[0, 1])
    del itm2, itm, itm1
    torch.cuda.empty_cache()
    return rv


#----------------------------------------------------------------------------------------------------------------


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
    betaX = torch.matmul(X.to_dense(), beta)
    TbX = bTheta + betaX

    itm1 = torch.log(f(Y, TbX)+seps)

    bsXs = beta.matmul(sXs.to_dense())
    # TbsXs = bTheta.unsqueeze(dim=-1) + bsXs
    # Ym = Y.unsqueeze(dim=-1) + torch.zeros(N)
    
    itm2 = torch.log(f(Y, bTheta, bsXs).mean(dim=-1)+seps)

    itm = R.to_dense() * (itm1 - itm2)
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
    betaX = torch.matmul(X.to_dense(), beta)
    TbX = bTheta + betaX

    itm1 = torch.log(f(Y, TbX)+seps)

    itm2 = torch.log(intBernh(f, bTheta, beta, Y, prob)+seps)

    itm = R.to_dense() * (itm1 - itm2)
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

# Generate X from Bernoulli distribution
def genXBin(*args, prob=0.1, is_sparse=False):
    assert len(args) in [2, 3]
    p, size = args[-1], args[:-1]
    X = npr.uniform(0, 1, args)
    idx0, idx1 = X>=prob, X<prob
    X[idx0] = 0 
    X[idx1] = 1
    if len(args) == 2:
        X = X.transpose()
    if is_sparse:
        return torch.tensor(X).to(dtorchdtype).to_sparse()
    else:
        return torch.tensor(X).to(dtorchdtype)

# generate missing matrix R under linear and quadratic relation.
def genR(Y, typ="Linear", a=2, b=0.4, inp=6.5, is_sparse=False):
    typ = typ.lower()
    if "linear".startswith(typ):
        Thre = 5*Y - inp
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


# Generatae Y when Y|X \sim N(m, sigma**2)
def genYnorm(X, bTheta, beta, sigma=0.1): 
    n, m, _ = X.shape
    M = bTheta + X.to_dense().matmul(beta)
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


#----------------------------------------------------------------------------------------------------------------
# To compute the difference of beta and bTheta in 2 consecutive iterations
# as ai termination criteria
def ParaDiff(Olds, News):
    errs = torch.tensor([torch.norm(Olds[i]-News[i]) for i in range(2)])
    return errs.max()

#----------------------------------------------------------------------------------------------------------------
# The functions for computing the adaptive learning rates
def Gbetafn(betaOld, wtbetaOld, bThetaOld, LpbOld, Lamb):
    diffbeta = betaOld - wtbetaOld
    itm1 = (LpbOld * diffbeta).sum()
    itm2 = Lamb * (betaOld.abs().sum() - wtbetaOld.abs().sum())
    return itm1 + itm2

def wtbetafn(LpbOld, QOld, Lamb):
    p = LpbOld.shape[0]
    wtbetat = torch.zeros(p)
    itmax = LpbOld.abs().argmax().item()
    if LpbOld.abs().max() > Lamb:
        wtbetat[itmax] = - QOld*torch.sign(LpbOld[itmax]) 
    return wtbetat

# def wtbetafn(LpbOld, QOld, Lamb):
#     p = LpbOld.shape[0]
#     wtbetatm1 = torch.ones(p) * 0.02
#     diff = torch.tensor(10.0)
#     flag = 1
#     while diff > 1e-3:
#         objfv = LpbOld + Lamb * torch.sign(wtbetatm1)
#         itmax = objfv.abs().argmax().item()
#         st = torch.zeros(p)
#         st[itmax] = - QOld*torch.sign(objfv[itmax]) 
#         wtbetat = (1- 1/(flag+2)) * wtbetatm1 + (1/(flag+2)) * st
#         diff = (wtbetat-wtbetatm1).norm()
#         wtbetatm1 = wtbetat
#         flag += 1
#         #print(diff, "------", wtbetat, Lamb * torch.sign(wtbetatm1))
#     #    print(diff)
#     return wtbetat

def etabetat(betaOld, bThetaOld, LpbOld, Lamb, QOld, ajr=0.001):
    wtbetaOld = wtbetafn(LpbOld, QOld, Lamb)
    Gbetav = Gbetafn(betaOld, wtbetaOld, bThetaOld, LpbOld, Lamb)
    etatv = ((betaOld - wtbetaOld)**2).sum()/Gbetav
    return etatv.item() * ajr

def GThetafn(bThetaOld, wtbThetaOld, betaNew, LpTNew, LamT):
    diffTheta = bThetaOld - wtbThetaOld
    itm1 = (LpTNew * diffTheta).sum()
    itm2 = LamT * (bThetaOld.norm(p="nuc") - wtbThetaOld.norm(p="nuc"))
    return itm1 + itm2

def wtThetafn(LpTNew, ROld, LamT):
    svdres = torch.svd(LpTNew)
    n, m = LpTNew.shape
    alpha1, u, v = svdres.S.max(), svdres.U[:, 0].unsqueeze(dim=-1), svdres.V[:, 0].unsqueeze(dim=-1)
    if alpha1 > LamT:
        wtbThetaOld = - torch.min(ROld, LamT) * u.matmul(v.t())
    else:
        wtbThetaOld = torch.zeros(n, m, dtype=dtorchdtype)
    return wtbThetaOld

def etaThetat(betaNew, bThetaOld, LpTNew, LamT, ROld, ajr=0.5):
    wtbThetaOld = wtThetafn(LpTNew, ROld, LamT)
    GThetav = GThetafn(bThetaOld, wtbThetaOld, betaNew, LpTNew, LamT)
    eta1tv = ((wtbThetaOld - bThetaOld)**2).sum()/GThetav
    #print(((wtbThetaOld - bThetaOld)**2).sum(), GThetav, (wtbThetaOld - bThetaOld).norm())
    return ajr*eta1tv.item()
    


#----------------------------------------------------------------------------------------------------------------

# New algorithm  to optimize the bTheta when X is Bernoulli 
def BthetaBern(MaxIters, X, Y, R, conDenfs, TrueParas, CT=1, log=0, bThetainit=None, tol=1e-4, prob=0.5, ErrOpts=0):
    """
    MaxIters: max iteration number.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    Trueparas: True paramter of beta and bTheta, a list like [beta0, bTheta0]
    CT: the constant of Lambda_bTheta
    log: Whether output detail training log. 0 not output, 1 output simple training log, 2 output detailed training log.
    bThetainit: initial value of bTheta
    tol: terminate tolerace.
    prob: sucessful probability of entry of X
    ErrOpts: whether output errors of beta and bTheta. 0 no, 1 yes
    """
    n, m, p = X.shape
    f, f2, f22 = conDenfs
    Lcon = -20
    # To contain the training errors of bTheta, respectively.
    Terrs = []
    Likelis = []
    bThetahats = []
    etaTs = []

    # The true parameters
    beta0, bTheta0 = TrueParas
    # Initial the value of beta, bTheta and R_b
    bThetaOld = torch.rand(n, m) if bThetainit is None else bThetainit
    # the relative change of Loss, i.e. |L_k - L_k+1|/max(|L_k|, |L_k+1|, 1), here the L_k and L_k+1 are with penalty items.
    reCh = 1

    # Under Cb and CT, compute the Lambda_beta and Lambda_bTheta
    LamT = LamTfn(CT, n, m, p)

    # The log output, nothing to do with algorithm.
    if log>=0:
        tb1 = PrettyTable(["Basic Value", "LamT",  "norm of bTheta0"])
        tb1.add_row(["",  f"{LamT.item():>5.3g}", f"{bTheta0.norm().item():>8.3g}"])
        print(tb1)
    # The loss, i.e. L +  Lamdab_bTheta * ||bTheta||
    Losses = []

    # Starting optimizing.
    for t in range(MaxIters):
        #--------------------------------------------------------------------------------
        LvNow = LBern(bThetaOld, beta0,  f, X, Y, R, prob)
        # Add L with penalty items.
        LossNow = missdepLR(LvNow, bThetaOld, beta0, LamT, 0)
        Losses.append(LossNow.item())

        #--------------------------------------------------------------------------------
        # Update bTheta 
        LpTvOld = LpTBern(bThetaOld, beta0, conDenfs, X, Y, R, prob)
        ROld = (LossNow - Lcon)/LamT
        etaTOld = etaThetat(beta0, bThetaOld, LpTvOld, LamT, ROld)
        etaTOld = 100
        svdres = torch.svd(bThetaOld-LpTvOld*etaTOld)
        #print(LpTvOld.norm())
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
            Terrs.append((bTheta0-bThetaOld).norm().item())
            Likelis.append(LvNow.item())
            bThetahats.append(bThetaOld.norm().item())
            etaTs.append(etaTOld)
        if log==1:
            tb2 = PrettyTable(["Iteration", "etaT", "Loss", "Error of Theta"])
            tb2.add_row([f"{t+1:>6}/{MaxIters}", f"{etaTOld:>8.3g}", f"{Losses[-1]:>8.3f}", f"{torch.norm(bTheta0-bThetaNew).item():>8.3f}"])
            print(tb2)
        if log==2:
            tb2 = PrettyTable(["Iteration", "etaT", "Loss", "-likelihood",  "Error of Theta", "reCh", "Norm of Thetat", "Norm of difference"])
            tb2.add_row([f"{t+1:>4}/{MaxIters}", f"{etaTOld:>8.3g}", f"{Losses[-1]:>8.3f}", f"{LvNow.item():>8.6f}", f"{torch.norm(bTheta0-bThetaNew).item():>8.3f}",
                f"{reCh:>8.4g}",  f"{bThetaNew.norm().item():>8.3f}", f"{(bThetaOld-bThetaNew).norm().item():>8.3g}"])
            print(tb2)
        #--------------------------------------------------------------------------------
        # if reCh is smaller than tolerance, stop the loop
        #if t >= 1:
        #    if (reCh < tol):
        #        break
        # if the difference of 2 consecutive bThetahat is smaller than tolerance, stop the loop
        if (bThetaOld-bThetaNew).norm() < tol:
            break
        #--------------------------------------------------------------------------------
        #print(softS[:100])
        # Change New to Old for starting next iteration
        bThetaOld = bThetaNew 
   #--------------------------------------------------------------------------------
    if ErrOpts:
        return bThetaOld, t+1, Terrs, Likelis, bThetahats, etaTs
    else:
        return bThetaOld, t+1

# New algorithm  to optimize the bTheta when X is Bernoulli 
def BetaBern(MaxIters, X, Y, R, sXs, conDenfs, TrueParas, Cb=1, log=0, betainit=None, tol=1e-4, prob=0.1, ErrOpts=0):
    """
    MaxIters: max iteration number.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    Trueparas: True paramter of beta and bTheta, a list like [beta0, bTheta0]
    Cb: the constant of Lambda_beta
    log: Whether output detail training log. 0 not output, 1 output simple training log, 2 output detailed training log.
    betainit: initial value of bTheta
    tol: terminate tolerace.
    prob: sucessful probability of entry of X
    ErrOpts: whether output errors of beta and bTheta. 0 no, 1 yes
    """
    n, m, p = X.shape
    f, f2, _ = conDenfs
    Lcon = -100
    numExact = 14
    # To contain the training errors of bTheta, respectively.
    Berrs = []
    Likelis = []
    betahats = []
    etabs = []

    # The true parameters
    beta0, bTheta0 = TrueParas
    # the relative change of Loss, i.e. |L_k - L_k+1|/max(|L_k|, |L_k+1|, 1), here the L_k and L_k+1 are with penalty items.
    reCh = 1

    betaOld = torch.rand(p) if betainit is None else betainit

    # Under Cb, compute the Lambda_beta 
    Lamb = Lambfn(Cb, n, m)

    # The log output, nothing to do with algorithm.
    if log>=0:
        tb1 = PrettyTable(["Basic Value", "Lamb", "norm of beta0"])
        tb1.add_row(["",  f"{Lamb.item():>5.3g}", f"{beta0.norm().item():>5.3g}"])
        print(tb1)
    # The loss, i.e. L +  Lamdab_beta * ||beta||
    Losses = []

    # Starting optimizing.
    for t in range(MaxIters):
        #--------------------------------------------------------------------------------
        # To get the number of nonzeros entry in betaOld
        NumN0Old = p - (betaOld.abs()==0).sum().to(dtorchdtype)
        #--------------------------------------------------------------------------------
        # compute the loss function (with penalty items) under betaOld and bThetaOld

        # Compute L (without penalty items) 
        # If betaOld is truly sparse, compute exact integration, otherwise use MCMC
        if NumN0Old > numExact:
            LvNow = missdepL(bTheta0, betaOld, f, X, Y, R, sXs)
        else:
            LvNow = LBern(bTheta0, betaOld, f, X, Y, R, prob)
        # Add L with penalty items.
        LossNow = missdepLR(LvNow, bTheta0, betaOld, 0, Lamb)
        QOld = (LossNow - Lcon)/Lamb
        print("F", LossNow-Lcon)
        Losses.append(LossNow.item())

        #--------------------------------------------------------------------------------
        # This block is to update beta.
        # If betaOld is truly sparse, compute exact integration, otherwise use MCMC
        if NumN0Old > numExact:
            LpbvOld = missdepLpb(bTheta0, betaOld, conDenfs, X, Y, R, sXs)
        else:
            LpbvOld = LpbBern(bTheta0, betaOld, conDenfs, X, Y, R, prob)
        # compute the learning rate of beta
        etabOld = etabetat(betaOld, bTheta0, LpbvOld, Lamb, QOld)
        etabOld = 100
        if len(etabs) >= 1 and etabOld >= etabs[-1]:
            etabOld = etabs[-1]
        betaNewRaw = betaOld - etabOld * LpbvOld
        # Using rho function to soften updated beta
        betaNew = SoftTO(betaNewRaw, etabOld*Lamb)
        #--------------------------------------------------------------------------------
        # compute the relative change of Loss
        if t >= 1:
            Lk1 = Losses[-1]
            Lk = Losses[-2]
            reCh = np.abs(Lk1-Lk)/np.max(np.abs((Lk, Lk1, 1))) 
        #--------------------------------------------------------------------------------
        # This block is for log output and Error save, nothing to do with the algorithm
        if ErrOpts:
            Berrs.append((beta0-betaOld).norm().item())
            Likelis.append(LvNow.item())
            betahats.append(betaOld.norm().item())
            etabs.append(etabOld)
        if log==1:
            tb2 = PrettyTable(["Iteration", "etab", "Loss", "Error of beta"])
            tb2.add_row([f"{t+1:>6}/{MaxIters}", f"{etabOld:>8.3g}", f"{Losses[-1]:>8.3f}", f"{torch.norm(beta0-betaNew).item():>8.3f}"])
            print(tb2)
        if log==2:
            tb2 = PrettyTable(["Iteration", "etab", "Loss", "-likelihood",  "Error of beta", "reCh", "Norm of betat", "Norm of difference"])
            tb2.add_row([f"{t+1:>4}/{MaxIters}", f"{etabOld:>8.3g}", f"{Losses[-1]:>8.3f}", f"{LvNow.item():>8.5g}", f"{torch.norm(beta0-betaNew).item():>8.6f}",
                f"{reCh:>8.4g}",  f"{betaNew.norm().item():>8.3f}", f"{(betaOld-betaNew).norm().item():>8.5g}"])
            print(tb2)
        #--------------------------------------------------------------------------------
        # if reCh is smaller than tolerance, stop the loop
        if t >= 1:
            if (reCh < tol):
                break
        # if the difference of 2 consecutive bThetahat is smaller than tolerance, stop the loop
        if (betaOld-betaNew).norm() < tol:
            break
        #--------------------------------------------------------------------------------
        # Change New to Old for starting next iteration
        print(betaOld)
        betaOld = betaNew 
   #--------------------------------------------------------------------------------
    if ErrOpts:
        return betaOld, t+1, Berrs, Likelis, betahats, etabs
    else:
        return betaOld, t+1


# New algorithm  to optimize the bTheta and beta when X is Bernoulli 
def NewBern(MaxIters, X, Y, R, sXs, conDenfs, TrueParas, Cb=10, CT=1, log=0, bThetainit=None, betainit=None, tols=None, prob=0.5, ErrOpts=0, etab=0.05, etaT=0.05):
    """
    MaxIters: max iteration number.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    sXs: sample of X for MCMC, p x N
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    Trueparas: True paramter of beta and bTheta, a list like [beta0, bTheta0]
    Cb: the constant of Lambda_beta
    CT: the constant of Lambda_bTheta
    log: Whether output detail training log. 0 not output, 1 output simple training log, 2 output detailed training log.
    bThetainit: initial value of bTheta
    tol: terminate tolerace.
    prob: sucessful probability of entry of X
    ErrOpts: whether output errors of beta and bTheta. 0 no, 1 yes
    etabinit: The initial learning rate of beta
    etaTinit: The initial learning rate of btheta
    """
    n, m, p = X.shape
    f, f2, f22 = conDenfs
    tol, tolb, tolT = tols
    numExact = 14
    Lcon = -10
    # To contain the training errors of bTheta and beta, respectively.
    Terrs = []
    Berrs = []
    Likelis = []
    bThetahats = []
    betahats = []
    etass = [[100, 1e10]]

    # The true parameters
    beta0, bTheta0 = TrueParas
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
        tb1 = PrettyTable(["Basic Value", "Lamb", "LamT", "norm of beta0", "norm of bTheta0"])
        tb1.add_row(["",  f"{Lamb.item():>5.3g}", f"{LamT.item():>5.3g}", f"{beta0.norm().item():>5.3g}", f"{bTheta0.norm().item():>5.3g}"])
        print(tb1)
    # The loss, i.e. L +  Lamdab_bTheta * ||bTheta||
    Losses = []

    # Starting optimizing.
    for t in range(MaxIters):
        #--------------------------------------------------------------------------------
        # To get the number of nonzeros entry in betaOld
        NumN0Old = p - (betaOld.abs()==0).sum().to(dtorchdtype)
        #--------------------------------------------------------------------------------
        # compute the loss function (with penalty items) under betaOld and bThetaOld

        # Compute L (without penalty items) 
        # If betaNew is truly sparse, compute exact integration, otherwise use MCMC

        if NumN0Old > numExact:
            LvNow = missdepL(bThetaOld, betaOld, f, X, Y, R, sXs)
        else:
            LvNow = LBern(bThetaOld, betaOld, f, X, Y, R, prob)
        # Add L with penalty items.
        LossNow = missdepLR(LvNow, bThetaOld, betaOld, LamT, Lamb)
        QOld = (LossNow - Lcon)/Lamb
        Losses.append(LossNow.item())

        #--------------------------------------------------------------------------------
        # This block is to update beta.
        # If betaOld is truly sparse, compute exact integration, otherwise use MCMC
        if NumN0Old > numExact:
            LpbvOld = missdepLpb(bThetaOld, betaOld, conDenfs, X, Y, R, sXs)
        else:
            LpbvOld = LpbBern(bThetaOld, betaOld, conDenfs, X, Y, R, prob)
        # compute the learning rate of beta
        etabOld = etabetat(betaOld, bThetaOld, LpbvOld, Lamb, QOld)
        etabOld = etab # 0.05 for linear setting
        betaNewRaw = betaOld - etabOld * LpbvOld
        # Using rho function to soften updated beta
        betaNew = SoftTO(betaNewRaw, etabOld*Lamb)

        #print(torch.cuda.memory_cached(0)/1e9, torch.cuda.memory_allocated(0)/1e9)
        #--------------------------------------------------------------------------------
        # To get the number of nonzeros entry in betaNew
        NumN0New = p - (betaNew.abs()==0).sum().to(dtorchdtype)
        torch.cuda.empty_cache()

        #--------------------------------------------------------------------------------
        # Update bTheta 
        if NumN0New > numExact:
            LpTvOld = missdepLpT(bThetaOld, betaNew, conDenfs, X, Y, R, sXs)
            LvNew = missdepL(bThetaOld, betaNew, f, X, Y, R, sXs)
        else:
            LpTvOld = LpTBern(bThetaOld, betaNew, conDenfs, X, Y, R, prob)
            LvNew = LBern(bThetaOld, betaNew, f, X, Y, R, prob)
        torch.cuda.empty_cache()
        LossNew = missdepLR(LvNew, bThetaOld, betaNew, LamT, Lamb)
        ROld = (LossNew - Lcon)/LamT
        etaTOld = etaThetat(betaNew, bThetaOld, LpTvOld, LamT, ROld)
        etaTOld = etaT # 0.05 for linear setting
        if len(etass) >= 1 and etaTOld >= etass[-1][-1]:
            etaTOld = etass[-1][-1]
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
            Terrs.append((bTheta0-bThetaOld).norm().item())
            Berrs.append((beta0-betaOld).norm().item())
            Likelis.append(LvNow.item())
            bThetahats.append(bThetaOld.norm().item())
            betahats.append(betaOld.norm().item())
            etass.append([etabOld, etaTOld])
        if t % 10 == 0:
            if log==1:
                tb2 = PrettyTable(["Iteration", "etaT", "Loss", "Error of beta", "Error of Theta"])
                tb2.add_row([f"{t+1:>6}/{MaxIters}", f"{etaTOld:>8.3g}", f"{Losses[-1]:>8.3f}", f"{torch.norm(beta0-betaNew).item():>8.3f}", f"{torch.norm(bTheta0-bThetaNew).item():>8.3f}"])
                print(tb2)
            if log==2:
                tb2 = PrettyTable(["Iter", "etaT", "etab", "Loss", "-lkd", "bErr", "TErr", "reCh", "bNorm", "TNorm", "bdiff", "Tdiff", "betat L0"])
                tb2.add_row([f"{t+1:>4}/{MaxIters}", f"{etaTOld:>3.3g}", f"{etabOld:>3.3g}", f"{Losses[-1]:>6.3f}", f"{LvNow.item():>2.1g}",  f"{torch.norm(beta0-betaNew).item():>6.3f}", f"{torch.norm(bTheta0-bThetaNew).item():>6.3f}",
                    f"{reCh:>6.4g}",  f"{betaNew.norm().item():>2.1f}", f"{bThetaNew.norm().item():>2.1f}", f"{(betaOld-betaNew).norm().item():>6.3g}", f"{(bThetaOld-bThetaNew).norm().item():>6.3g}", f"{NumN0New.item()}"])
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
        return betaOld, bThetaOld, t+1, Berrs, Terrs, betahats, bThetahats, Likelis, etass
    else:
        return betaOld, bThetaOld, t+1
