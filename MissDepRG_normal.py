from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit
import time
from scipy.stats import norm as STN

torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
np.random.seed(0) #numpy
random.seed(0) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

cuda = torch.cuda.is_available()
# cuda = False
if cuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
sigma = 0.5


def torchstnpdf(y):
    pi = torch.tensor([np.pi])
    prefix = torch.sqrt(1/pi/2)
    return prefix*torch.exp(-y**2/2)

# def fn(y, m, bsXs=None, sigma=sigma):
#     # y     : n x m
#     # m     : n x m 
#     # bsXs  : N
#     pi = torch.tensor([np.pi])
#     prefix = torch.sqrt(1/pi/2/sigma**2)
#     if bsXs is not None:
#         v = torch.exp(-(y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))**2/2/sigma**2)
#     else:
#         v = torch.exp(-(y-m)**2/2/sigma**2)
#     return prefix*v
# 
# 
# def fn2(y, m, bsXs=None, sigma=sigma):
#     pi = torch.tensor([np.pi])
#     sigma2 = sigma**2
#     prefix = torch.sqrt(1/pi/2/sigma2)
#     #v1 = torch.exp(-(y-m)**2/2/sigma2)
#     #v2 =  (y-m)/sigma2
#     if bsXs is not None:
#         return prefix*torch.exp(-(y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))**2/2/sigma2)*(y.unsqueeze(-1)- (m.unsqueeze(-1) + bsXs))/sigma2
#     else:
#         return prefix*torch.exp(-(y-m)**2/2/sigma2)*(y-m)/sigma2
# 
# 
# def fn22(y, m, sigma=sigma):
#     pi = torch.tensor([np.pi])
#     sigma2 = sigma**2
#     prefix = torch.sqrt(1/pi/2/sigma2)
#     expitm = torch.exp(-(y-m)**2/2/sigma2)
#     v =  (y-m)/sigma2
#     linitm = v**2-1/sigma2
#     return prefix*linitm*expitm


def ftn(y, m, bsXs=None, sigma=sigma, a=-100, b=100):
    # y     : n x m
    # m     : n x m 
    # bsXs  : N
    print(bsXs.shape, m.shape)
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


def ftn2(y, m, bsXs=None, sigma=sigma, a=-100, b=100):
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


def ftn22(y, m, bsXs=None, sigma=sigma, a=-100, b=100):
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


n = 100
m = 100
p = 100
N = 10000
sigmax = np.sqrt(1/3)

X = genXdis(n, m, p, type="mvnorm", sigmax=sigmax) 
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, 3, 4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m) * 7
M = bTheta0 + X.matmul(beta0)
Y = genYnorm(X, bTheta0, beta0, sigma=sigma)
#Y = genYtnorm(X, bTheta0, beta0, -100, 100,sigma=sigma)
R = genR(Y)
# print(X.matmul(beta0).abs().mean(), bTheta0.abs().mean())
print(R.sum()/R.numel())
sda
sXs = genXdis(N, p, type="mvnorm", sigmax=sigmax) 
conDenfs = [fn, fn2, fn22]
# etascaleinv = missdepLpbbLoop(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# etascale = etascaleinv.inverse()
# U, S, V = etascale.svd()
# print(S.max())

# Lv = missdepL(bTheta0, beta0, fn, X, Y, R, sXs)
# Lvn = Lnormal(bTheta0, beta0, X, Y, R, sigmax=sigmax, sigma=sigma)
# print(Lv-Lvn, Lvn)
# Lv2 = missdepLLoop(bTheta0, beta0, fn, X, Y, R, sXs)
# t0 = time.time()
# LpTv = missdepLpT(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# LpTvn = LpTnormal(bTheta0, beta0, X, Y, R, sigmax=sigmax, sigma=sigma)
# #LpTv2 = missdepLpTLoop(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# print((LpTv-LpTvn).norm(), LpTvn.norm())
# Lpbv = missdepLpb(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# Lpbvn = Lpbnormal(bTheta0, beta0, X, Y, R, sigma=sigma, sigmax=sigmax)
# # Lpbv2 = missdepLpbLoop(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# print((Lpbv - Lpbvn).norm(), Lpbvn.norm())
# t1 = time.time()
# print(t1-t0)
# raise SystemExit




Cbpool = np.concatenate([np.arange(0.1, 1, 0.1), np.arange(1, 100, 4), np.arange(200, 1e4, 500)])
CTpool = Cbpool/100
STpool = np.arange(1000, 1e6, 500)

numRG = 200
# eta = 1/(5*0.75*m*p)
eta = 0.01 
tol = 1e-4
TrueParas = [beta0, bTheta0]
results = [{"beta0":beta0, "bTheta0":bTheta0, "eta":eta, "tol": tol}]
print(results)
for i in range(numRG):
    len1, len2, len3  = len(Cbpool), len(CTpool), len(STpool)
    idx1, idx2, idx3 = np.random.randint(0, len1), np.random.randint(0, len2), np.random.randint(0, len3)
    Cb, CT, ST = Cbpool[idx1], CTpool[idx2], STpool[idx3]
    print(Cb, CT, ST)
    try:
        #betahat, bThetahat, _, numI = MCGD(100, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, eta=eta, Cb=Cb, CT=CT, tol=tol, log=2, ST=2500)
#                           missdepL=missdepLLoop, missdepLpb=missdepLpbLoop, missdepLpT=missdepLpTLoop)
         betahat, bThetahat, _, numI = MCGDnormal(300, X, Y, R, TrueParas=TrueParas, eta=eta, Cb=Cb, CT=CT, tol=tol, log=2, sigmax=sigmax,sigma=sigma, ST=10000, betainit=torch.cat((torch.tensor([0, 1.0, 0, 5, 0, 0, 0, 10]), torch.zeros(p-8))))
    except RuntimeError:
        results.append((-100, Cb, -100, -100,  CT, -100, -100, ST))
        print(
            f"The {i+1}th/{numRG},"
            f"Iteration Fails!"
        )
    else:
        errb = torch.norm(beta0-betahat)
        errT = torch.norm(bTheta0-bThetahat)
        results.append((numI, Cb, errb.item(), betahat.norm().item(), CT, errT.item(), bThetahat.norm().item(), ST))
        print(
            f"The {i+1}th/{numRG},"
            f"The Iteration number is {numI}, "
            f"The error of beta is {errb.item():.3f}, "
            f"The error of bTheta is {errT.item():.3f}."
        )

f = open("./RandGridinit.pkl", "wb")
pickle.dump(results, f)
f.close()
# betahat, bThetahat, _ = MCGD(1000, X, Y, R, sXs, conDenfs, eta=1e-1, debug=0, Cb=10, CT=0.8, tol=1e-4, log=1)
# print(torch.norm(beta0-betahat))
# print(torch.norm(beta0))
# print(torch.norm(bTheta0-bThetahat))
# print(torch.norm(bTheta0))
# print(betahat)
# betahat, bThetahat, _ = MCGD(1000, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, eta=eta, Cb=10, CT=0.1, tol=tol, log=2, sigmax=sigmax, sigma=sigma)
