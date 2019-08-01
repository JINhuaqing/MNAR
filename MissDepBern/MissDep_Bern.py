from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit
import time

torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
np.random.seed(0) #numpy
random.seed(0) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

cuda = torch.cuda.is_available()
cuda = False
if cuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
sigma = 0.5


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


def fn22(y, m, sigma=sigma):
    pi = torch.tensor([np.pi])
    sigma2 = sigma**2
    prefix = torch.sqrt(1/pi/2/sigma2)
    expitm = torch.exp(-(y-m)**2/2/sigma2)
    v =  (y-m)/sigma2
    linitm = v**2-1/sigma2
    return prefix*linitm*expitm


n = 100
m = 100
p = 100
N = 20000
STm = np.sqrt(n*m/10000)

prob = 0.1
X = genXdis(n, m, p, type="Bern", prob=prob) 
#X = torch.tensor(np.random.randint(0, 2, (m, n, p))).float()
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, 3, 4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m) * 7 
M = bTheta0 + X.matmul(beta0)
Y = genYnorm(X, bTheta0, beta0, sigma=sigma)
R = genR(Y)
# print(X.matmul(beta0).abs().mean(), bTheta0.abs().mean())
print(R.sum()/R.numel())
sXs = genXdis(N, p, type="Bern", prob=prob) 
conDenfs = [fn, fn2, fn22]



numRG = 100
# eta = 1/(5*0.75*m*p)
eta = 0.01 
tol = 1e-4
TrueParas = [beta0, bTheta0]
results = [{"beta0":beta0, "bTheta0":bTheta0, "eta":eta, "tol": tol}]
betainit = torch.rand(p)
idxs = torch.randperm(p)[:p-8]
betainit[idxs] = 0
betainit = beta0* 1.1
bThetainit = bTheta0 * 1.1

Cb, CT, ST = 465.9, 0.766, 102.3*STm
betahat, bThetahat, _, numI, Berrs, Terrs = MCGDBern(1000, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, eta=eta, Cb=Cb, CT=CT, tol=tol, log=2, ST=ST, prob=prob, betainit=betainit, bThetainit=bThetainit, ErrOpts=1)
errb = torch.norm(beta0-betahat)
errT = torch.norm(bTheta0-bThetahat)
results.append((numI, Cb, errb.item(), betahat.norm().item(), CT, errT.item(), bThetahat.norm().item(), ST))
print(
    f"The Iteration number is {numI}, "
    f"The error of beta is {errb.item():.3f}, "
    f"The error of bTheta is {errT.item():.3f}."
)

outres = {"Cb":Cb, "CT": CT, "ST":ST, "Berrs": Berrs, "Terrs": Terrs}
f = open(f"./outputs/Bern_{Cb:.4g}_{CT:.4f}_{ST:.0f}.pkl", "wb")
pickle.dump(outres, f)
f.close()
