import torch
from utils import *
import numpy as np
import numpy.random as npr
from torch.distributions.normal import Normal
from prettytable import PrettyTable
from scipy.stats import truncnorm

#----------------------------------------------------------------------------------------------------------------

# This file contains the functions for main simulation for MAR settting
#
#

#----------------------------------------------------------------------------------------------------------------

# seps: small number to avoid zero in log funciton and denominator. 
seps = 1e-15
# dtorchdtype and dnpdtype are default data types used for torch package and numpy package, respectively.
dtorchdtype = torch.float64
dnpdtype = np.float64


#----------------------------------------------------------------------------------------------------------------


# Compute the value of first derivative of L w.r.t bTheta for any distributions X under EM method
def EMLpT(bTheta, beta, conDenfs, X, Y):
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

    # remove the elements whose corresponding betaK is zero
    idxNon0 = torch.nonzero(beta).view(-1)
    if idxNon0.shape[0] == 0:
        idxNon0 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    beta = beta[idxNon0]
    X = X[:, :, idxNon0]

    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    itm1 = (f2(Y, TbX)/(f(Y, TbX)+seps))

    itm = itm1/(m*n)
    return -itm


#----------------------------------------------------------------------------------------------------------------
# Compute the value of second derivative of L w.r.t vec(bTheta) for any distributions X.
def EMLpTT(bTheta, beta, conDenfs, X, Y):
    """
    Input:
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. 
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    Output:
    itm: n x m
    """
    n, m, p = X.shape
    f, f2, f22 = conDenfs

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

    itm = (itm1 - itm2)/(m*n)
    return -itm



#----------------------------------------------------------------------------------------------------------------
# Compute the value of first derivative of L w.r.t beta for any distributions X under EM setting
def EMLpb(bTheta, beta, conDenfs, X, Y):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    conDenfs: a list to contain the likelihood function of Y|X, and its fisrt derivative and second derivative w.r.t second argument.
             [f, f2, f22]. In fact, f22 is not used.
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    """
    f, f2, _ = conDenfs
    # remove the elements whose corresponding betaK is zero
    idxNon0 = torch.nonzero(beta).view(-1)
    if idxNon0.shape[0] == 0:
        idxNon0 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    betaNon0 = beta[idxNon0]
    XNon0 = X[:, :, idxNon0]

    betaX = torch.matmul(XNon0, betaNon0) # n x m
    del XNon0
    TbX = bTheta + betaX # n x m

    itm = (f2(Y, TbX)/(f(Y, TbX)+seps)).unsqueeze(dim=2) * X # n x m x p 
    del X

    return -itm.mean(dim=[0, 1])


#----------------------------------------------------------------------------------------------------------------
# Compute the value of L w.r.t beta for any distributions X under EM setting
def EML(bTheta, beta, f, X, Y):
    """
    bTheta: the matrix parameter, n x m
    beta: the vector parameter, p 
    f: likelihood function of Y|X
    X: the covariate matrix, n x m x p
    Y: the response matrix, n x m
    """
    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    itm1 = torch.log(f(Y, TbX)+seps)

    itm = itm1
    return -itm.mean(dim=[0, 1])


#----------------------------------------------------------------------------------------------------------------


# New algorithm  to optimize the bTheta and beta when X is Bernoulli under EM and Y is linear
def EMNewBern(MaxIters, X, Y, R, conDenfs, TrueParas, Cb=10, CT=1, etab=1e-3, etaT=1, log=0, bThetainit=None, betainit=None, tols=None, prob=0.5, ErrOpts=0):
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
    f, f2, f22 = conDenfs
    tol, tolb, tolT = tols
    # To contain the training errors of bTheta and beta, respectively.
    Terrs = []
    Berrs = []
    Likelis = []
    bThetahats = []
    betahats = []

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
    t00 = time.time()
    for t in range(MaxIters):
        t0 = time.time()

        # The estimated Y
        betaXOld = torch.matmul(X, betaOld) # n x m
        TbXOld = bThetaOld + betaXOld # n x m
        Rb = R.bool()
        Y[Rb] = TbXOld[Rb]

        #--------------------------------------------------------------------------------
        # To get the number of nonzeros entry in betaOld
        NumN0Old = p - (betaOld.abs()==0).sum().to(dtorchdtype)
        #--------------------------------------------------------------------------------
        # compute the loss function (with penalty items) under betaOld and bThetaOld

        # Compute L (without penalty items) 
        # If betaNew is truly sparse, compute exact integration, otherwise use MCMC
        LvNow = EML(bThetaOld, betaOld, f, X, Y)
        # Add L with penalty items.
        LossNow = missdepLR(LvNow, bThetaOld, betaOld, LamT, Lamb)
        Losses.append(LossNow.item())

        #--------------------------------------------------------------------------------
        # This block is to update beta.
        # If betaOld is truly sparse, compute exact integration, otherwise use MCMC
        betaNewRaw = betaOld-etab * EMLpb(bThetaOld, betaOld, conDenfs, X, Y)
        # Using rho function to soften updated beta
        betaNew = SoftTO(betaNewRaw, etab*Lamb)

        #--------------------------------------------------------------------------------
        # To get the number of nonzeros entry in betaOld
        NumN0New = p - (betaNew.abs()==0).sum().to(dtorchdtype)

        #--------------------------------------------------------------------------------
        # Update bTheta 
        LpTvOld = EMLpT(bThetaOld, betaNew, conDenfs, X, Y)
        svdres = torch.svd(bThetaOld-LpTvOld*etaT)
        U, S, V =  svdres.U, svdres.S, svdres.V
        softS = (S-LamT*etaT).clamp_min(0)
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
        if t % 10 == 0:
            if log==1:
                tb2 = PrettyTable(["Iteration", "etaT", "Loss", "BErr", "TErr"])
                tb2.add_row([f"{t+1:>6}/{MaxIters}", f"{etaT:>8.3g}", f"{Losses[-1]:>8.3f}", f"{torch.norm(beta0-betaNew).item():>8.3f}", f"{torch.norm(bTheta0-bThetaNew).item():>8.3f}"])
                print(tb2)
            if log==2:
                tb2 = PrettyTable(["Iter", "etaT", "etab", "Loss", "-lkd", "BErr", "TErr", "reCh", "BNorm ", "TNorm ", "Bdiff", "Tdiff", "bL0norm"])
                tb2.add_row([f"{t+1:>4}/{MaxIters}", f"{etaT:>4.2g}", f"{etab:>4.3g}", f"{Losses[-1]:>4.3f}", f"{LvNow.item():>4.2g}",  f"{torch.norm(beta0-betaNew).item():>5.3f}", f"{torch.norm(bTheta0-bThetaNew).item():>5.3f}",
                    f"{reCh:>4.2g}",  f"{betaNew.norm().item():>5.3f}", f"{bThetaNew.norm().item():>5.3f}", f"{(betaOld-betaNew).norm().item():>5.3g}", f"{(bThetaOld-bThetaNew).norm().item():>5.3g}", f"{NumN0New.item()}"])
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
        betaOld, bThetaOld = betaNew, bThetaNew 
   #--------------------------------------------------------------------------------
    if ErrOpts:
        return betaOld, bThetaOld, t+1, Berrs, Terrs, betahats, bThetahats, Likelis
    else:
        return betaOld, bThetaOld, t+1

