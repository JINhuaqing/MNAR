import torch
import numpy as np
import numpy.random as npr
from torch.distributions.normal import Normal
from prettytable import PrettyTable
from scipy.stats import truncnorm
import time

# ----------------------------------------------------------------------------------------------------------------

# This file contains the functions for main simulation.
#
#

# ----------------------------------------------------------------------------------------------------------------

# seps: small number to avoid zero in log funciton and denominator. 
seps = 1e-15
# dtorchdtype and dnpdtype are default data types used for torch package and numpy package, respectively.
dtorchdtype = torch.float64
dnpdtype = np.float64

# ----------------------------------------------------------------------------------------------------------------


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

# ----------------------------------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------------------------------


# Compute the value of first derivative of L w.r.t bTheta with MCMC method for any distributions X.
def missdepLpT(bTheta, beta, conDenfs, X, Y, R, fct=10):
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
    
    n, m, p = X.shape
    sXs = (X.reshape(-1, p).t()[:, :10000])
    _, N = sXs.shape
    f, f2, f22 = conDenfs

    # remove the elements whose corresponding betaK is zero
    idxNon0 = torch.nonzero(beta).view(-1)
    if idxNon0.shape[0] == 0:
        idxNon0 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    beta = beta[idxNon0] # p0 x 1
    sXs = sXs[idxNon0] # p0 x N
    X = X[:, :, idxNon0] # n x m x p0

    betaX = torch.matmul(X, beta) # n x m
    TbX = bTheta + betaX # n x m
    itm1 = (f2(Y, TbX)/(f(Y, TbX)+seps))

    # Integration part 
    bsXs = beta.matmul(sXs) # 1 x N
    torch.cuda.empty_cache()

    lenSeg = int(np.ceil(m/fct))
    itm2 = torch.zeros((n, m))
    for i in np.arange(0, m, lenSeg):
        lower, upper = i, i+lenSeg
        YPart = Y[:, lower:upper]
        bThetaPart = bTheta[:, lower:upper]
        itm2denPart = f(YPart, bThetaPart, bsXs).mean(dim=-1) + seps
        itm2numPart = f2(YPart, bThetaPart, bsXs).mean(dim=-1)
        itm2Part = itm2numPart / itm2denPart
        itm2[:, lower:upper] = itm2Part

    itm = R.to_dense() * (itm1 - itm2)/(m*n)
    return -itm




# Compute the value of first derivative of L w.r.t beta with MCMC method for any distributions X.
def missdepLpb(bTheta, beta, conDenfs, X, Y, R, fct=10):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    """
    n, m, p = X.shape
    sXs = (X.reshape(-1, p).t()[:, :10000])
    _, N = sXs.shape

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

    lenSeg = int(np.ceil(m/fct))
    bsXs = betaNon0.matmul(sXsNon0) # N
    sumRes = torch.zeros(p)
    for i in np.arange(0, m, lenSeg):
        lower, upper = i, i+lenSeg
        TbXPart = TbX[:, lower:upper]
        YPart = Y[:, lower:upper]
        bThetaPart = bTheta[:, lower:upper]
        XPart = (X[:, lower:upper, :])
        RPart = (R.to_dense()[:, lower:upper]).to_sparse()

        itm1Part = ((f2(YPart, TbXPart)/(f(YPart, TbXPart)+seps)).unsqueeze(dim=2) * XPart) # n x m x p 

        itm2denPart = (f(YPart, bThetaPart, bsXs).mean(dim=-1) + seps)
        itm2numinPart =  f2(YPart, bThetaPart, bsXs) # ni x mi x N 
        itm2numPart = torch.stack([(itm2numinPart*sX).mean(dim=-1) for sX in sXs], dim=-1) # ni x mi x p
        itm2Part = (itm2numPart/itm2denPart.unsqueeze(dim=-1))
        itmPart = (RPart.to_dense().unsqueeze(dim=2) * (itm1Part - itm2Part))
        sumRes += itmPart.sum(dim=[0, 1])
        torch.cuda.empty_cache()
    return -sumRes/n/m



# ----------------------------------------------------------------------------------------------------------------

# Compute the value of L with MCMC method for any distributions X.
def missdepL(bTheta, beta, f, X, Y, R, fct=10, is_logf=False):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    f: likelihood function of Y|X
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
    """
    n, m, p = X.shape
    sXs = (X.reshape(-1, p).t()[:, :10000])
    _, N = sXs.shape
    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    if is_logf:
        itm1 = f(Y, TbX)
    else:
        itm1 = torch.log(f(Y, TbX)+seps)

    bsXs = beta.matmul(sXs)
    # TbsXs = bTheta.unsqueeze(dim=-1) + bsXs
    # Ym = Y.unsqueeze(dim=-1) + torch.zeros(N)

    lenSeg = int(np.ceil(m/fct))
    itm2 = torch.zeros((n, m))
    for i in np.arange(0, m, lenSeg):
        lower, upper = i, i+lenSeg
        YPart = Y[:, lower:upper]
        bThetaPart = bTheta[:, lower:upper]
        if is_logf:
            itm2Part = torch.log(torch.exp(f(YPart, bThetaPart, bsXs)).mean(dim=-1)+seps)
        else:
            itm2Part = torch.log(f(YPart, bThetaPart, bsXs).mean(dim=-1)+seps)
        itm2[:, lower:upper] = itm2Part
    

    itm = R.to_dense() * (itm1 - itm2)
    return -itm.mean(dim=[0, 1])




# ----------------------------------------------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------------------------------------------

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
        return torch.tensor(X, device="cpu").to(dtorchdtype).cuda()
    else:
        return torch.tensor(X).to(dtorchdtype)

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


#----------------------------------------------------------------------------------------------------------------
# To compute the difference of beta and bTheta in 2 consecutive iterations
# as ai termination criteria
def ParaDiff(Olds, News):
    errs = torch.tensor([torch.norm(Olds[i]-News[i]) for i in range(2)])
    return errs.max()



# ----------------------------------------------------------------------------------------------------------------

# function to optimize on realdata 
def RealDataAlg(MaxIters, X, Y, R, conDenfs, Cb=10, CT=1, log=0, bThetainit=None, betainit=None, tols=None,  ErrOpts=0, etab=0.05, etaT=0.05, fct=10):
    """
    MaxIters: max iteration number.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    R: the Missing matrix, n x m
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
    """
    n, m, p = X.shape
    if len(conDenfs) == 4:
        is_logf = True
        f = conDenfs[-1]
    else:
        is_logf = False
        f = conDenfs[0]
    tol, tolb, tolT = tols
    Lcon = -10
    # To contain the training errors of bTheta and beta, respectively.
    Terrs = []
    Berrs = []
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
    #NT = 100
    # Starting optimizing.
    t00 = time.time()
    for t in range(MaxIters):
        t0 = time.time()
        #if t==200:
        #    LamT = LamTfn(CT/100, n, m, p)
        #    Lamb = Lambfn(Cb/100, n, m)
        #    etab = etab/100.
        #    etaT = etaT/100.

        #--------------------------------------------------------------------------------
        # To get the number of nonzeros entry in betaOld
        NumN0Old = p - (betaOld.abs()==0).sum().to(dtorchdtype)
        #--------------------------------------------------------------------------------
        # compute the loss function (with penalty items) under betaOld and bThetaOld

        # Compute L (without penalty items) 
        LvNow = missdepL(bThetaOld, betaOld, f, X, Y, R, fct=fct, is_logf=is_logf)
        # Add L with penalty items.
        LossNow = missdepLR(LvNow, bThetaOld, betaOld, LamT, Lamb)
        QOld = (LossNow - Lcon)/Lamb
        Losses.append(LossNow.item())

        #--------------------------------------------------------------------------------
        # This block is to update beta.
        LpbvOld = missdepLpb(bThetaOld, betaOld, conDenfs[:3], X, Y, R, fct=fct)

        etabOld = etab 
        betaNewRaw = betaOld - etabOld * LpbvOld
        # Using rho function to soften updated beta
        betaNew = SoftTO(betaNewRaw, etabOld*Lamb)

        #--------------------------------------------------------------------------------
        # To get the number of nonzeros entry in betaNew
        NumN0New = p - (betaNew.abs()==0).sum().to(dtorchdtype)
        torch.cuda.empty_cache()

        #--------------------------------------------------------------------------------
        # Update bTheta 
        LpTvOld = missdepLpT(bThetaOld, betaNew, conDenfs[:3], X, Y, R, fct=fct)
        LvNew = missdepL(bThetaOld, betaNew, f, X, Y, R, fct=fct, is_logf=is_logf)
        torch.cuda.empty_cache()
        LossNew = missdepLR(LvNew, bThetaOld, betaNew, LamT, Lamb)
        ROld = (LossNew - Lcon)/LamT
        etaTOld = etaT # 0.05 for linear setting

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
        if t % 10 == 0:
            if log==1:
                tb2 = PrettyTable(["Iteration", "etaT", "Loss"])
                tb2.add_row([f"{t+1:>6}/{MaxIters}", f"{etaTOld:>8.3g}", f"{Losses[-1]:>8.3f}"])
                print(tb2)
            if log==2:
                tb2 = PrettyTable(["Iter", "etaT", "etab", "Loss", "-lkd", "reCh", "bNorm", "TNorm", "bdiff", "Tdiff", "betat L0"])
                tb2.add_row([f"{t+1:>4}/{MaxIters}", f"{etaTOld:>3.3g}", f"{etabOld:>3.3g}", f"{Losses[-1]:>6.3f}", f"{LvNow.item():>2.1g}",  f"{reCh:>6.4g}",  f"{betaNew.norm().item():>2.1f}", f"{bThetaNew.norm().item():>2.1f}", f"{(betaOld-betaNew).norm().item():>6.3g}", f"{(bThetaOld-bThetaNew).norm().item():>6.3g}", f"{NumN0New.item()}"])
                print(tb2)
                t1 = time.time()
                print(f"The time for current iteration is {t1-t0:.3f}s ")
                print(f"The average time for each iteration is {(t1-t00)/(t+1):.3f}s ")
                print(f"The rank of estimate Theta is {torch.matrix_rank(bThetaNew)}.")
                print(betaNew)
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
        etaT = etaT * 0.995
        etab = etab * 0.995
   #--------------------------------------------------------------------------------
    if ErrOpts:
        return betaOld, bThetaOld, t+1, betahats, bThetahats, Likelis
    else:
        return betaOld, bThetaOld, t+1



def YelpMissing(Yraw, OR=0.22, y1ratio=0.5):
    """
    OR: observed ratio.
    y1ratio: num of 1 / (num of 1 and 0) in the removed part
    """
    # get the R
    R = Yraw.copy()
    R[Yraw!=-1] = 1
    R[Yraw==-1] = 0

    rawOR = np.sum(R)/np.prod(R.shape)
    assert OR <= rawOR
    
    numrv = np.prod(R.shape) * (rawOR-OR) # number of samples to remove
    numrv1 = int(numrv*y1ratio) # number of 1 to remove
    numrv0 = int(numrv) - numrv1 # number of 0 to remove

    mask1 = (Yraw > 3.5) & (R==1) # matrix with observed 1 as True
    mask0 = (Yraw < 3.5) & (R==1) # matrix with observed 0 as True
    selidx1 = np.random.choice(int(np.sum(mask1)), numrv1, replace=0)
    selidx0 = np.random.choice(int(np.sum(mask0)), numrv0, replace=0)

    tmp1 = R[mask1] 
    tmp1[selidx1] = 0
    R[mask1] = tmp1

    tmp0 = R[mask0] 
    tmp0[selidx0] = 0
    R[mask0] = tmp0

    return R

