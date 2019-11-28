import numpy as np
import torch
from torch.distributions.normal import Normal

dtorchdtype = torch.float32

def Alpha_0b_exp1(Y, X, bTheta, beta):
    _, _, p = X.shape
    M = bTheta + X.matmul(beta) 
    S2sq = (torch.exp(M)/(1+torch.exp(M)) - Y)**2
    allvs = S2sq.unsqueeze(-1).unsqueeze(-1) * (X.unsqueeze(-1) * X.unsqueeze(-2))
    vs1 = allvs[Y==1].reshape(-1, p, p)
    vs0 = allvs[Y==0].reshape(-1, p, p)
    return [vs0.mean(0), vs1.mean(0)]

def Alpha_0b_exp2(Y, X, bTheta, beta):
    _, _, p = X.shape
    M = bTheta + X.matmul(beta) 
    S2 = (torch.exp(M)/(1+torch.exp(M)) - Y)
    allvs = S2.unsqueeze(-1) * X
    vs1 = allvs[Y==1].reshape(-1, p)
    vs0 = allvs[Y==0].reshape(-1, p)
    return [vs0.mean(0), vs1.mean(0)]

def Alpha_0b(Y, X, bTheta, beta):
    cond10, cond11 = Alpha_0b_exp1(Y, X, bTheta, beta)
    cond20, cond21 = Alpha_0b_exp2(Y, X, bTheta, beta)
    f0 = cond10 - cond20.unsqueeze(-1) * cond20.unsqueeze(0)
    f1 = cond11 - cond21.unsqueeze(-1) * cond21.unsqueeze(0)
    #pref0, pref1 = 1-Normal(0, 1).cdf(0-inp), 1-Normal(0, 1).cdf(1-inp)
    pref0, pref1 = 1 - 0.65, 1 - 0.05
    p0, p1 = (Y==0).sum().float()/Y.numel(), (Y==1).sum().float()/Y.numel()
    mat = p0*pref0*f0 + p1*pref1*f1
    svdres = torch.svd(mat)
    alphamin = svdres.S.min().item()
    return alphamin/4

def Sigma_1F(Y, X, bTheta, beta):
    M = bTheta + X.matmul(beta) 
    a = np.max([bTheta.abs().max().item(), beta.abs().max().item()])
    c0 = X.abs().sum(-1).max().item()
    c0 = 1
    S2 = (torch.exp(M)/(1+torch.exp(M)) - Y)
    H2 = torch.exp(M)/(1+torch.exp(M))**2
    ds2 = S2.abs().max().item()
    dh2 = H2.abs().max().item()
    ds2, dh2 = 1, 1
    return 32*c0**2*a**2*(dh2+ds2**2)


def Alpha_0T_exp1(Y, X, bTheta, beta):
    _, _, p = X.shape
    M = bTheta + X.matmul(beta) 
    S2sq = (torch.exp(M)/(1+torch.exp(M)) - Y)**2
    num1, num0 = (Y==1).sum().float(), (Y==0).sum().float()
    S2sq[Y==1] = S2sq[Y==1]/num1
    S2sq[Y==0] = S2sq[Y==0]/num0
    return S2sq 

def Alpha_0T_exp2(Y, X, bTheta, beta):
    _, _, p = X.shape
    M = bTheta + X.matmul(beta) 
    S2 = (torch.exp(M)/(1+torch.exp(M)) - Y)
    num1, num0 = (Y==1).sum().float(), (Y==0).sum().float()
    S2[Y==1] = S2[Y==1]/num1
    S2[Y==0] = S2[Y==0]/num0
    return S2


def Alpha_0T(Y, X, bTheta, beta):
    n, m = Y.shape
    Condexp1 = Alpha_0T_exp1(Y, X, bTheta, beta)
    Condexp2 = Alpha_0T_exp2(Y, X, bTheta, beta)
    Fv = Condexp1 - Condexp2**2
    mat = torch.zeros(n, m)
    #pref0, pref1 = 1-Normal(0, 1).cdf(0-inp), 1-Normal(0, 1).cdf(1-inp)
    pref0, pref1 = 1 - 0.65, 1 - 0.05
    p0, p1 = (Y==0).sum().float()/Y.numel(), (Y==1).sum().float()/Y.numel()
    mat[Y==0] = p0*pref0*Fv[Y==0]
    mat[Y==1] = p1*pref1*Fv[Y==1]
    alphamin = mat.abs().min().item()
    return alphamin/4


def Sigma_dF(Y, X, bTheta, beta):
    M = bTheta + X.matmul(beta) 
    a = np.max([bTheta.abs().max().item(), beta.abs().max().item()])
    S2 = (torch.exp(M)/(1+torch.exp(M)) - Y)
    H2 = torch.exp(M)/(1+torch.exp(M))**2
    ds2 = S2.abs().max().item()
    dh2 = H2.abs().max().item()
    ds2, dh2 = 1, 1
    return 32*a**2*(dh2+ds2**2)
