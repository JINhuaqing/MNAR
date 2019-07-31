import pyspark
from pyspark import SparkContext
from utitlities import *



def LpbSpark(bTheta, beta, conDenfs, X, Y, R, sXs):
    """
    sXs: p x N, samples of X_ij to compute the MCMC integration
    """
    _, N = sXs.shape
    f, f2, _ = conDenfs
    betaX = torch.matmul(X, beta)
    TbX = bTheta + betaX

    itm1 = (f2(Y, TbX)/(f(Y, TbX)+seps)).unsqueeze(dim=2) * X

    bsXs = beta.matmul(sXs)
    TbsXs = bTheta.unsqueeze(dim=-1) + bsXs
    Ym = Y.unsqueeze(dim=-1) + torch.zeros(N)
    
    itm2den = f(Ym, TbsXs).mean(dim=-1) + seps
    itm2num = (f2(Ym, TbsXs).unsqueeze(dim=-2) * sXs).mean(dim=-1)
    itm2 = itm2num/itm2den.unsqueeze(dim=-1)

    itm = R.unsqueeze(dim=2) * (itm1 - itm2)
    return -itm.mean(dim=[0, 1])
