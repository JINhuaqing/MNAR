import torch
import numpy as np
import numpy.random as npr
from torch.distributions.normal import Normal
from prettytable import PrettyTable
from scipy.stats import truncnorm

__all__  = [
    "missdepL", "missdepLLoop", "missdepLpT", "missdepLpTLoop", "missdepLpbbLoop",
    "missdepLpb", "missdepLpbLoop", "missdepLR", "SoftTO", "MCGD", "Lnormal", 
    "genXdis", "genX", "genR", "genbTheta", "genYnorm", "genbeta", "MCGDnormal", 
    "omegat" , "Rub", "ParaDiff", "LamTfn", "Lambfn", "LpTnormal", "Lpbnormal",
    "LBern", "LpTBern", "LpbBern", "MCGDBern", "Dshlowerfnorm", "genYtnorm"
]


seps = 1e-15


def H2fnorm(y, m, sigma=0.5):
    n, m = y.shape
    return -torch.ones(n, m) /sigma**2

def S2fnorm(y, m, sigma):
    return (y-m)/sigma**2


def Dshlowerfnorm(Y, X, beta, bTheta, sigma=0.5):
    m = bTheta + X.matmul(beta)
    H2v = H2fnorm(Y, m, sigma)
    S2v = S2fnorm(Y, m, sigma)
    Ds2 = S2v.abs().max().item()
    Dh2 = H2v.abs().max().item()
    return Ds2, Dh2


def Blist(s):
    slist = list(range(2**s))
    mat = []
    for i in slist:
        strbi = bin(i)
        strbi = strbi.split("b")[-1]
        strbi = (s-len(strbi)) * "0" + strbi
        mat.append(list(strbi))
    matarr = np.array(mat) 
    return matarr.astype(np.float32)

def intBernh(f, bTheta, beta, Y, prob):
    idxNon0 = torch.nonzero(beta).view(-1)
    s = (beta != 0).sum().float().item()
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


def intBernhX(f, bTheta, beta, Y, prob):
    p = beta.shape[0]
    idxNon0 = torch.nonzero(beta).view(-1)
    s = (beta != 0).sum().float().item()
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

    


def LpTnormal(bTheta, beta,  X, Y, R, sigmax=0.5, sigma=0.5):
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


def Lpbnormal(bTheta, beta, X, Y, R, sigma=0.5, sigmax=0.5):
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


def missdepLpb(bTheta, beta, conDenfs, X, Y, R, sXs):
    """
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


def LpbBern(bTheta, beta, conDenfs, X, Y, R, prob=0.5):
    """
    sXs: p x N, samples of X_ij to compute the MCMC integration
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

def LpbSpark(bTheta, beta, conDenfs, X, Y, R, sXs, sc):
    """
    sXs: p x N, samples of X_ij to compute the MCMC integration
    """
    n, m, p = X.shape
    _, N = sXs.shape
    f, f2, _ = conDenfs
    betaX = torch.matmul(X, beta) # n x m
    TbX = bTheta + betaX # n x m

    itm1 = (f2(Y, TbX)/(f(Y, TbX)+seps)).unsqueeze(dim=2) * X # n x m x p 

    
    Ybc = sc.broadcast(Y)
    itm1bc = sc.broadcast(itm1)
    sXsbc = sc.broadcast(sXs)
    bThetabc = sc.broadcast(bTheta)

    def func(x):
        ii, jj = x
        Yij = Ybc.value[ii, jj]
        bThetaij = bThetabc.value[ii, jj]
        incre = torch.zeros(p)
        if R[ii, jj]:
            itm1ij = itm1bc.value[ii, jj]
            arg1 = Yij + torch.zeros(N)
            arg2 = bThetaij + beta.matmul(sXsbc.value)
            itm2ijden = f(arg1, arg2).mean() + seps
            itm2ijnum = (f2(arg1, arg2) * sXsbc.value).mean(dim=-1)
            itm2ij = itm2ijnum/itm2ijden
            incre = itm1ij - itm2ij
        return incre
    RDDidx = sc.range(n).cartesian(sc.range(m)).filter(lambda x: R[x[0], x[1]])
    res = RDDidx.map(func)
    results = res.sum()
    return -results/m/n


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



def SoftTO(x, a):
    rx = torch.zeros(x.shape)
    idx1 = x > a
    idx2 = x < -a
    rx[idx1] = x[idx1] - a
    rx[idx2] = x[idx2] + a 
    return rx


def missdepLpT(bTheta, beta, conDenfs, X, Y, R, sXs):
    """
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


def LpTBern(bTheta, beta, conDenfs, X, Y, R, prob=0.5):
    """
    sXs: p x N, samples of X_ij to compute the MCMC integration
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


def missdepLpTLoop(bTheta, beta, conDenfs, X, Y, R, sXs):
    """
    sXs: p x N, samples of X_ij to compute the MCMC integration
    """
    n, m, p = X.shape
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


def Lnormal(bTheta, beta,  X, Y, R, sigmax=0.5, sigma=0.5):
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

def missdepL(bTheta, beta, f, X, Y, R, sXs):
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


def LBern(bTheta, beta, f, X, Y, R, prob=0.5):
    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    itm1 = torch.log(f(Y, TbX)+seps)

    # TbsXs = bTheta.unsqueeze(dim=-1) + bsXs
    # Ym = Y.unsqueeze(dim=-1) + torch.zeros(N)
    
    itm2 = torch.log(intBernh(f, bTheta, beta, Y, prob)+seps)

    itm = R * (itm1 - itm2)
    return -itm.mean(dim=[0, 1])

def missdepLLoop(bTheta, beta, f, X, Y, R, sXs):
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


def missdepLR(Lv, bTheta, beta, LamT, Lamb):
    itm2 = LamT * torch.norm(bTheta, p="nuc")
    itm3 = Lamb * beta.abs().sum()
    return Lv + itm2 + itm3


def omegat(bT, tbTt, R, LamT, LpTv, Rb, tRb, ST):
    num = ((bT-tbTt).t().matmul(LpTv)).trace() + LamT*(Rb-tRb)
    den = ST * ((bT-tbTt)**2 * R).mean() + seps
    itm = num/den
    if itm > 1:
        return torch.tensor([1.0]) 
#     elif itm < -1:
#         return torch.tensor([-1.0]) 
    else:
        return itm

def Rub(Lv, beta, Rb, LamT, Lamb):
    L = 50 
    if (Lv + L) > 0:
        Fv = Lv + Lamb * beta.abs().sum() + LamT * Rb + L
    else:
        Fv = Lamb * beta.abs().sum() + LamT * Rb 
    return Fv/LamT


def Lambfn(C, n, m):
    rawv = np.sqrt(np.log(m+n))/m/n
    return torch.tensor([C*rawv], dtype=torch.float)

def LamTfn(C, n, m, p, sp=0.1):
    d = np.sqrt(m*n)
    rawvs = [np.sqrt(np.log(d)/d), np.sqrt(sp*p*np.log(p))/d, (np.log(p))**(1/4)/np.sqrt(d)]
    rawv = np.max(rawvs)
    return torch.tensor([C*rawv], dtype=torch.float)


def genX(n, m, p):
    X = torch.zeros(n, m, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                setv = list(range(5*(k-1)+1, 5*k+1))
                if ((j-1)*(n)+i) in setv:
                    X[i, j, k] = 1
    return X

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
    return torch.tensor(X).float()


def genbTheta(n, m, rank=4):
    bTheta = torch.rand(n, m)
    U, S, V = torch.svd(bTheta)
    bTheta = U[:, :rank].matmul(torch.diag(S[:rank])).matmul(V[:, :rank].transpose(1, 0))
    return bTheta


def genbeta(p, sparsity=0.1):
    zeroidx = torch.rand(p) > sparsity
    beta = torch.rand(p)
    beta[zeroidx] = 0
    return beta


# Y|X \sim N(m, m**2/2)
# def genYnorm(X, bTheta, beta):
#     n, m, _ = X.shape
#     M = bTheta + X.matmul(beta)
#     Y = torch.randn(n, m)*((M**2/2).sqrt()) + M
#     return Y

def genYnorm(X, bTheta, beta, sigma=0.1):
    n, m, _ = X.shape
    M = bTheta + X.matmul(beta)
    Y = torch.randn(n, m)*sigma + M
    return Y


def genYtnorm(X, bTheta, beta, a, b, sigma=0.1):
    n, m, _ = X.shape
    M = bTheta + X.matmul(beta)
    Marr = M.cpu().numpy()
    a = a/sigma
    b = b/sigma
    Yarr = truncnorm.rvs(a, b, loc=Marr, scale=sigma)
    return torch.tensor(Yarr).float()


def genR(Y, type="Linear", inp=6.5):
    type = type.lower()
    if "linear".startswith(type):
        Thre = Y  - inp#- 8 #- 1/2 # -  7 # -1/2 #+2
        probs = Normal(0, 1).cdf(Thre)
        ranUnif = torch.rand_like(probs)
        R = probs <= ranUnif
    elif "quadratic".startswith(type):
        Thre = Y**2 - 2*Y - 0.4
        probs = Normal(0, 1).cdf(Thre)
        ranUnif = torch.rand_like(probs)
        R = probs >= ranUnif
    else:
        raise TypeError("Wrong type of independence!")
    return R.float()


def ParaDiff(Olds, News):
    errs = torch.tensor([torch.norm(Olds[i]-News[i]) for i in range(2)])
    return errs.max()


def MCGDnormal(MaxIters, X, Y, R, TrueParas, eta=0.001, Cb=5, CT=0.01, log=0,
        betainit=None, bThetainit=None, Rbinit=None, tol=1e-4, sigma=0.5, sigmax=0.3, ST=10000):
    n, m, p = X.shape
    beta0, bTheta0 = TrueParas
    betaOld = torch.rand(p) if betainit is None else betainit
    RbOld = torch.rand(1) if Rbinit is None else Rbinit
    bThetaOld = torch.rand(n, m) if bThetainit is None else bThetainit

    Lamb = Lambfn(Cb, n, m)
    LamT = LamTfn(CT, n, m, p, 0.05)
    if log>=1:
        tb1 = PrettyTable(["Basic Value", "Lamb", "LamT", "eta"])
        tb1.add_row(["", f"{Lamb.item():>5.3g}", f"{LamT.item():>5.3g}", f"{eta:>5.3g}"])
        print(tb1)
    Losses = []

    for t in range(MaxIters):
        # update beta
        betaNewRaw = betaOld - eta * Lpbnormal(bThetaOld, betaOld, X, Y, R, sigma=sigma, sigmax=sigmax)
        betaNew = SoftTO(betaNewRaw, eta*Lamb)
        #print((betaNew.abs()<=1e-2 ).sum().float()/p, betaNew.abs().min())

        # compute the loss function
        LvOld = Lnormal(bThetaOld, betaNew, X, Y, R, sigma=sigma, sigmax=sigmax)
        LossOld = missdepLR(LvOld, bThetaOld, betaOld, LamT, Lamb)
        Losses.append(LossOld.item())

        RubNew = Rub(LvOld, betaNew, RbOld, LamT, Lamb)
        LpTvOld = LpTnormal(bThetaOld, betaNew, X, Y, R, sigma=sigma, sigmax=sigmax)
        svdres = torch.svd(LpTvOld)
        alpha1, u, v = svdres.S.max(), svdres.U[:, 0].unsqueeze(dim=-1), svdres.V[:, 0].unsqueeze(dim=-1)
        

        if LamT >= alpha1:
            tbTNew, tRbNew = torch.zeros(n, m, dtype=torch.float), torch.tensor([0.0])
        else:
            tbTNew, tRbNew =  -RubNew * u.matmul(v.t()), RubNew

        omeganew = omegat(bThetaOld, tbTNew, R, LamT, LpTvOld, RbOld, tRbNew, ST)

        # Update bTheta and Rb
        bThetaNew = bThetaOld + omeganew * (tbTNew-bThetaOld)
        RbNew = RbOld + omeganew * (tRbNew-RbOld)

        paradiff = ParaDiff([betaOld, bThetaOld], [betaNew, bThetaNew])
        if t >= 1:
            if (paradiff < tol) or (np.abs(Losses[-1]-Losses[-2]) < tol):
                break
        if log==1:
            tb2 = PrettyTable(["Iteration", "Loss", "Error of Beta", "Error of Theta"])
            tb2.add_row([f"{t+1:>6}/{MaxIters}", f"{Losses[-1]:>8.3f}", f"{torch.norm(beta0-betaNew).item():>8.3f}", f"{torch.norm(bTheta0-bThetaNew).item():>8.3f}"])
            print(tb2)
        if log==2:
            tb2 = PrettyTable(["Iteration", "Loss", "Error of Beta", "Error of Theta", "LamT", "Alpha", "Omegat", "Rb", "tildeRb", "Rub", "Norm of Betat", "Norm of Thetat"])
            tb2.add_row([f"{t+1:>4}/{MaxIters}", f"{Losses[-1]:>8.3f}", f"{torch.norm(beta0-betaNew).item():>8.3f}", f"{torch.norm(bTheta0-bThetaNew).item():>8.3f}",
                f"{LamT.item():>8.3g}", f"{alpha1.item():>8.3g}", f"{omeganew.item():>8.3g}",f"{RbNew.item():>8.3g}",
                f"{tRbNew.item():>8.3g}", f"{RubNew.item():>8.3g}", f"{betaNew.norm().item():>8.3f}", f"{bThetaNew.norm().item():>8.3f}"])
            print(tb2)
        # update the Beta and bTheta
        betaOld, bThetaOld, RbOld = betaNew, bThetaNew, RbNew
    return betaOld, bThetaOld, RbOld, t+1


def MCGD(MaxIters, X, Y, R, sXs, conDenfs, TrueParas, eta=0.001, Cb=5, CT=0.01, log=0,
         betainit=None, bThetainit=None, Rbinit=None, tol=1e-4,
         missdepL=missdepL, missdepLpb=missdepLpb, missdepLpT=missdepLpT, ST=10000):
    n, m, p = X.shape
    f, f2, _ = conDenfs
    beta0, bTheta0 = TrueParas
    betaOld = torch.rand(p) if betainit is None else betainit
    RbOld = torch.rand(1) if Rbinit is None else Rbinit
    bThetaOld = torch.rand(n, m) if bThetainit is None else bThetainit

    Lamb = Lambfn(Cb, n, m)
    LamT = LamTfn(CT, n, m, p, 0.05)
    if log>=1:
        tb1 = PrettyTable(["Basic Value", "Lamb", "LamT", "eta"])
        tb1.add_row(["", f"{Lamb.item():>5.3g}", f"{LamT.item():>5.3g}", f"{eta:>5.3g}"])
        print(tb1)
    Losses = []

    for t in range(MaxIters):
        # update beta
        betaNewRaw = betaOld - eta * missdepLpb(bThetaOld, betaOld, conDenfs, X, Y, R, sXs)
        betaNew = SoftTO(betaNewRaw, eta*Lamb)
        #print((betaNew.abs()==0 ).sum().float()/p, betaNew.abs().min())

        # compute the loss function
        LvOld = missdepL(bThetaOld, betaNew, f, X, Y, R, sXs)
        LossOld = missdepLR(LvOld, bThetaOld, betaOld, LamT, Lamb)
        Losses.append(LossOld.item())

        RubNew = Rub(LvOld, betaNew, RbOld, LamT, Lamb)
        LpTvOld = missdepLpT(bThetaOld, betaNew, conDenfs, X, Y, R, sXs)
        svdres = torch.svd(LpTvOld)
        alpha1, u, v = svdres.S.max(), svdres.U[:, 0].unsqueeze(dim=-1), svdres.V[:, 0].unsqueeze(dim=-1)
        

        if LamT >= alpha1:
            tbTNew, tRbNew = torch.zeros(n, m, dtype=torch.float), torch.tensor([0.0])
        else:
            tbTNew, tRbNew =  -RubNew * u.matmul(v.t()), RubNew

        omeganew = omegat(bThetaOld, tbTNew, R, LamT, LpTvOld, RbOld, tRbNew, ST=ST)

        # Update bTheta and Rb
        bThetaNew = bThetaOld + omeganew * (tbTNew-bThetaOld)
        RbNew = RbOld + omeganew * (tRbNew-RbOld)

        paradiff = ParaDiff([betaOld, bThetaOld], [betaNew, bThetaNew])
        if t >= 1:
            if (paradiff < tol) or (np.abs(Losses[-1]-Losses[-2]) < tol):
                break
        if log==1:
            tb2 = PrettyTable(["Iteration", "Loss", "Error of Beta", "Error of Theta"])
            tb2.add_row([f"{t+1:>6}/{MaxIters}", f"{Losses[-1]:>8.3f}", f"{torch.norm(beta0-betaNew).item():>8.3f}", f"{torch.norm(bTheta0-bThetaNew).item():>8.3f}"])
            print(tb2)
        if log==2:
            tb2 = PrettyTable(["Iteration", "Loss", "Error of Beta", "Error of Theta", "LamT", "Alpha", "Omegat", "Rb", "tildeRb", "Rub", "Norm of Betat", "Norm of Thetat"])
            tb2.add_row([f"{t+1:>4}/{MaxIters}", f"{Losses[-1]:>8.3f}", f"{torch.norm(beta0-betaNew).item():>8.3f}", f"{torch.norm(bTheta0-bThetaNew).item():>8.3f}",
                f"{LamT.item():>8.3g}", f"{alpha1.item():>8.3g}", f"{omeganew.item():>8.3g}",f"{RbNew.item():>8.3g}",
                f"{tRbNew.item():>8.3g}", f"{RubNew.item():>8.3g}", f"{betaNew.norm().item():>8.3f}", f"{bThetaNew.norm().item():>8.3f}"])
            print(tb2)
        # update the Beta and bTheta
        betaOld, bThetaOld, RbOld = betaNew, bThetaNew, RbNew
    return betaOld, bThetaOld, RbOld, t+1


def MCGDBern(MaxIters, X, Y, R, sXs, conDenfs, TrueParas, eta=0.001, Cb=5, CT=0.01, log=0, betainit=None, bThetainit=None, Rbinit=None, tol=1e-4, ST=10000, prob=0.5, ErrOpts=0, sps=0.05):
    n, m, p = X.shape
    f, f2, _ = conDenfs
    Berrs = []
    Terrs = []
    numExact = 12
    beta0, bTheta0 = TrueParas
    betaOld = torch.rand(p) if betainit is None else betainit
    RbOld = torch.rand(1) if Rbinit is None else Rbinit
    bThetaOld = torch.rand(n, m) if bThetainit is None else bThetainit

    Lamb = Lambfn(Cb, n, m)
    LamT = LamTfn(CT, n, m, p, sps)
    if log>=1:
        tb1 = PrettyTable(["Basic Value", "Lamb", "LamT", "eta"])
        tb1.add_row(["", f"{Lamb.item():>5.3g}", f"{LamT.item():>5.3g}", f"{eta:>5.3g}"])
        print(tb1)
    Losses = []

    for t in range(MaxIters):
        # update beta
        NumN0Old = p - (betaOld.abs()==0).sum().float()
        if NumN0Old > numExact:
            betaNewRaw = betaOld - eta * missdepLpb(bThetaOld, betaOld, conDenfs, X, Y, R, sXs)
        else:
            betaNewRaw = betaOld - eta * LpbBern(bThetaOld, betaOld, conDenfs, X, Y, R, prob)
        betaNew = SoftTO(betaNewRaw, eta*Lamb)
        NumN0New = p - (betaNew.abs()==0).sum().float()
        #print(NumN0New, NumN0Old)
        #print((betaNew.abs()==0 ).sum().float()/p, betaNew.abs().min())

        # compute the loss function
        if NumN0New > numExact:
            LvOld = missdepL(bThetaOld, betaNew, f, X, Y, R, sXs)
        else:
            LvOld = LBern(bThetaOld, betaNew, f, X, Y, R, prob)
        LossOld = missdepLR(LvOld, bThetaOld, betaOld, LamT, Lamb)
        Losses.append(LossOld.item())

        RubNew = Rub(LvOld, betaNew, RbOld, LamT, Lamb)
        if NumN0New > numExact:
            LpTvOld = missdepLpT(bThetaOld, betaNew, conDenfs, X, Y, R, sXs)
        else:
            LpTvOld = LpTBern(bThetaOld, betaNew, conDenfs, X, Y, R, prob)

        svdres = torch.svd(LpTvOld)
        alpha1, u, v = svdres.S.max(), svdres.U[:, 0].unsqueeze(dim=-1), svdres.V[:, 0].unsqueeze(dim=-1)
        

        if LamT >= alpha1:
            tbTNew, tRbNew = torch.zeros(n, m, dtype=torch.float), torch.tensor([0.0])
        else:
            tbTNew, tRbNew =  -RubNew * u.matmul(v.t()), RubNew

        omeganew = omegat(bThetaOld, tbTNew, R, LamT, LpTvOld, RbOld, tRbNew, ST=ST)

        # Update bTheta and Rb
        bThetaNew = bThetaOld + omeganew * (tbTNew-bThetaOld)
        RbNew = RbOld + omeganew * (tRbNew-RbOld)

        #print(betaNew)
        #paradiff = ParaDiff([betaOld, bThetaOld], [betaNew, bThetaNew])
        if ErrOpts:
            Berrs.append((beta0-betaOld).norm().item())        
            Terrs.append((bTheta0-bThetaOld).norm().item())
        if t >= 1:
            Lk1 = losses[-1]
            Lk = losses[-2]
            if (np.abs(Lk1-Lk)/np.max(np.abs((Lk, Lk1, 1))) < tol):
                break
        if log==1:
            tb2 = PrettyTable(["Iteration", "Loss", "Error of Beta", "Error of Theta"])
            tb2.add_row([f"{t+1:>6}/{MaxIters}", f"{Losses[-1]:>8.3f}", f"{torch.norm(beta0-betaNew).item():>8.3f}", f"{torch.norm(bTheta0-bThetaNew).item():>8.3f}"])
            print(tb2)
        if log==2:
            tb2 = PrettyTable(["Iteration", "Loss", "Error of Beta", "Error of Theta", "LamT", "Alpha", "Omegat", "Rb", "tildeRb", "Rub", "Norm of Betat", "Norm of Thetat"])
            tb2.add_row([f"{t+1:>4}/{MaxIters}", f"{Losses[-1]:>8.3f}", f"{torch.norm(beta0-betaNew).item():>8.3f}", f"{torch.norm(bTheta0-bThetaNew).item():>8.3f}",
                f"{LamT.item():>8.3g}", f"{alpha1.item():>8.3g}", f"{omeganew.item():>8.3g}",f"{RbNew.item():>8.3g}",
                f"{tRbNew.item():>8.3g}", f"{RubNew.item():>8.3g}", f"{betaNew.norm().item():>8.3f}", f"{bThetaNew.norm().item():>8.3f}"])
            print(tb2)
        # update the Beta and bTheta
        betaOld, bThetaOld, RbOld = betaNew, bThetaNew, RbNew
    if ErrOpts:
        return betaOld, bThetaOld, RbOld, t+1, Berrs, Terrs
    else:
        return betaOld, bThetaOld, RbOld, t+1
