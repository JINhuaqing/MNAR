from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit
from pyspark import SparkContext
from timeit import timeit, repeat  
import time

torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
np.random.seed(0) #numpy
random.seed(0) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

# Create SparkContext
sc = SparkContext("local[20]", "MissDep")

cuda = torch.cuda.is_available()
cuda = False
if cuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
sigma = 0.5


def fn(y, m, sigma=sigma):
    pi = torch.tensor([np.pi])
    prefix = torch.sqrt(1/pi/2/sigma**2)
    v = torch.exp(-(y-m)**2/2/sigma**2)
    return prefix*v


def fn2(y, m, sigma=sigma):
    pi = torch.tensor([np.pi])
    sigma2 = sigma**2
    prefix = torch.sqrt(1/pi/2/sigma2)
    v1 = torch.exp(-(y-m)**2/2/sigma2)
    v2 =  (y-m)/sigma2
    return prefix*v1*v2


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
N = 1000
sigmax = np.sqrt(1/3)

X = genXdis(n, m, p, type="mvnorm", sigmax=sigmax) 
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, 3, 4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m) * 7
M = bTheta0 + X.matmul(beta0)
Y = genYnorm(X, bTheta0, beta0, sigma=sigma)
R = genR(Y)
# print(X.matmul(beta0).abs().mean(), bTheta0.abs().mean())
print(R.sum()/R.numel())
sXs = genXdis(N, p, type="mvnorm", sigmax=sigmax) 
conDenfs = [fn, fn2, fn22]
# etascaleinv = missdepLpbbLoop(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# etascale = etascaleinv.inverse()
# U, S, V = etascale.svd()
# print(S.max())

Lv = missdepL(bTheta0, beta0, fn, X, Y, R, sXs)
Lvn = Lnormal(bTheta0, beta0, X, Y, R, sigmax=sigmax, sigma=sigma)
print((Lv-Lvn))

# Lv2 = missdepLLoop(bTheta0, beta0, fn, X, Y, R, sXs)
# LpTv = missdepLpT(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# LpTv2 = missdepLpTLoop(bTheta0, beta0, conDenfs, X, Y, R, sXs)

# t0 = time.time()
# for i in range(1):
#     Lpbv = missdepLpb(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# t1 = time.time()
# for i in range(1):
#     Lpbv1 = LpbSpark(bTheta0, beta0, conDenfs, X, Y, R, sXs, sc = sc)
# t2 = time.time()
# for i in range(1):
#     LpTv2 = missdepLpTLoop(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# t3 = time.time()
# print(t1-t0, t2-t1, t3 - t2)
raise SystemExit
# 


Cbpool = np.concatenate([np.arange(0.1, 1, 0.1), np.arange(1, 100, 4), np.arange(200, 1e4, 500)])
CTpool = Cbpool/100
STpool = np.arange(1000, 1e6, 500)

lenCb, lenCT, lenST  = len(Cbpool), len(CTpool), len(STpool)
numRG = 20
idxs = [np.random.randint(0, lenx, numRG) for lenx in [lenCb, lenCT, lenST]]
hyperparams = np.array([Cbpool[idxs[0]], CTpool[idxs[1]], STpool[idxs[2]]]).transpose(1, 0)
RDDhypers = sc.parallelize(hyperparams)

# eta = 1/(5*0.75*m*p)
eta = 0.01 
tol = 1e-4
TrueParas = [beta0, bTheta0]

def MCGDSpark(x):
    Cb, CT, ST = x
    try:
        betahat, bThetahat, _, numI = MCGD(10, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, eta=eta, Cb=Cb, CT=CT, tol=tol, log=0, ST=ST)
    except RuntimeError:
        return (-100, Cb, -100, -100,  CT, -100, -100, ST)
    else:
        errb = torch.norm(beta0-betahat)
        errT = torch.norm(bTheta0-bThetahat)
        return (numI, Cb, errb.item(), betahat.norm().item(), CT, errT.item(), bThetahat.norm().item(), ST)


t0 = time.time()
RDDres = RDDhypers.map(MCGDSpark)
res = RDDres.collect()
t1 = time.time()
print(res)
t2 = time.time()
results = [{"beta0":beta0, "bTheta0":bTheta0, "eta":eta, "tol": tol}]
# print(results)
for i in range(numRG):
    idx1, idx2, idx3 = np.random.randint(0, lenCb), np.random.randint(0, lenCT), np.random.randint(0, lenST)
    Cb, CT, ST = Cbpool[idx1], CTpool[idx2], STpool[idx3]
    try:
        betahat, bThetahat, _, numI = MCGD(10, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, eta=eta, Cb=Cb, CT=CT, tol=tol, log=0, ST=ST)
#                          missdepL=missdepLLoop, missdepLpb=missdepLpbLoop, missdepLpT=missdepLpTLoop)
        #betahat, bThetahat, _, numI = MCGDnormal(1000, X, Y, R, TrueParas=TrueParas, eta=eta, Cb=Cb, CT=CT, tol=tol, log=0, sigmax=sigmax,sigma=sigma, ST=ST)
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

t3 = time.time()
print(t3-t2, t1-t0)
# f = open("./RandGridinit.pkl", "wb")
# pickle.dump(results, f)
# f.close()
# betahat, bThetahat, _ = MCGD(1000, X, Y, R, sXs, conDenfs, eta=1e-1, debug=0, Cb=10, CT=0.8, tol=1e-4, log=1)
# print(torch.norm(beta0-betahat))
# print(torch.norm(beta0))
# print(torch.norm(bTheta0-bThetahat))
# print(torch.norm(bTheta0))
# print(betahat)
# betahat, bThetahat, _ = MCGD(1000, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, eta=eta, Cb=10, CT=0.1, tol=tol, log=2, sigmax=sigmax, sigma=sigma)
