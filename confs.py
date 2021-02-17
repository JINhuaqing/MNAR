import numpy as np
from scipy.stats import norm as STN
import torch

sigma = 0.1
a, b = -10, 10 

def torchstnpdf(y):
    pi = torch.tensor([np.pi])
    prefix = torch.sqrt(1/pi/2)
    return prefix*torch.exp(-y**2/2)


def ftn(y, m, bsXs=None, sigma=sigma, a=a, b=b):
    # y     : n x m
    # m     : n x m 
    # bsXs  : N
    Z = (STN.cdf(b/sigma) - STN.cdf(a/sigma)) * sigma
    if bsXs is not None:
        v = (y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))/sigma
        zidx = (y.unsqueeze(-1) > b + (m.unsqueeze(-1) + bsXs)) | (y.unsqueeze(-1) < a + (m.unsqueeze(-1) + bsXs))
    else:
        v = (y-m)/sigma
        zidx = (y > b + m) | (y < a+m)
    tv = torchstnpdf(v)/Z 
    tv[zidx] = 0
    return tv


def fn(y, m, bsXs=None, sigma=sigma):
    # y     : n x m
    # m     : n x m 
    # bsXs  : N
    pi = torch.tensor([np.pi])
    prefix = torch.sqrt(1/pi/2/sigma**2)
    if bsXs is not None:
        v = torch.exp(-(y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))**2/2/sigma**2)
    else:
        v = torch.exp(-(y-m)**2/2/sigma**2)
    return prefix*v

def Logistic(m):
    return torch.exp(m)/(1+torch.exp(m))

def fln(y, m, bsXs=None):
    if bsXs is not None:
        return torch.exp(y.unsqueeze(-1) * (m.unsqueeze(-1) + bsXs))/(1 + torch.exp(m.unsqueeze(-1) + bsXs))
    else:
        return torch.exp(y*m)/(1+torch.exp(m))

def fln2(y, m , bsXs=None):
    if bsXs is not None:
        return (2*y-1).unsqueeze(-1) * torch.exp((m.unsqueeze(-1) + bsXs))/ (1+torch.exp((m.unsqueeze(-1) + bsXs)))**2
    else:
        return (2*y-1) * torch.exp(m)/ (1+torch.exp(m))**2


def fln22(y, m, bsXs=None):
    if bsXs is not None:
        return (2*y-1).unsqueeze(-1)*torch.exp((m.unsqueeze(-1) + bsXs))*(1-torch.exp((m.unsqueeze(-1) + bsXs)))/(1+torch.exp((m.unsqueeze(-1) + bsXs)))**3
    else:
        return (2*y-1)*torch.exp(m)*(1-torch.exp(m))/(1+torch.exp(m))**3


def ftn2(y, m, bsXs=None, sigma=sigma, a=a, b=b):
    sigma2 = sigma**2
    Z = (STN.cdf(b/sigma) - STN.cdf(a/sigma)) * sigma
    if bsXs is not None:
        v = (y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))/sigma
        v2 = (y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))/sigma2
        zidx = (y.unsqueeze(-1) > b + (m.unsqueeze(-1) + bsXs)) | (y.unsqueeze(-1) < a + (m.unsqueeze(-1) + bsXs))
    else:
        v = (y-m)/sigma
        v2 = (y-m)/sigma2
        zidx = (y > b + m) | (y < a+m)
    tv = torchstnpdf(v)*v2/Z
    tv[zidx] = 0
    return tv


def fn2(y, m, bsXs=None, sigma=sigma):
    pi = torch.tensor([np.pi])
    sigma2 = sigma**2
    prefix = torch.sqrt(1/pi/2/sigma2)
    #v1 = torch.exp(-(y-m)**2/2/sigma2)
    #v2 =  (y-m)/sigma2
    if bsXs is not None:
        return prefix*torch.exp(-(y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))**2/2/sigma2)*(y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))/sigma2
    else:
        return prefix*torch.exp(-(y-m)**2/2/sigma2)*(y-m)/sigma2


def ftn22(y, m, bsXs=None, sigma=sigma, a=a, b=b):
    sigma2 = sigma**2
    Z = (STN.cdf(b/sigma) - STN.cdf(a/sigma)) * sigma
    if bsXs is not None:
        v = (y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))/sigma
        v2 = ((y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))/sigma2)**2 - 1/sigma2
        zidx = (y.unsqueeze(-1) > b + (m.unsqueeze(-1) + bsXs)) | (y.unsqueeze(-1) < a + (m.unsqueeze(-1) + bsXs))
    else:
        v = (y-m)/sigma
        v2 = ((y-m)/sigma2)**2 - 1/sigma2
        zidx = (y > b + m) | (y < a+m)
    tv = torchstnpdf(v)*v2/Z
    tv[zidx] = 0
    return tv



def fn22(y, m, sigma=sigma):
    pi = torch.tensor([np.pi])
    sigma2 = sigma**2
    prefix = torch.sqrt(1/pi/2/sigma2)
    expitm = torch.exp(-(y-m)**2/2/sigma2)
    v =  (y-m)/sigma2
    linitm = v**2-1/sigma2
    return prefix*linitm*expitm

