from utilities import *
import random
import numpy as np
import torch
import pickle
import timeit

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
N = 5000

conDenfs = [fn, fn2, fn22]
# etascaleinv = missdepLpbbLoop(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# etascale = etascaleinv.inverse()
# U, S, V = etascale.svd()
# print(S.max())

# Lv = missdepL(bTheta0, beta0, fn, X, Y, R, sXs)
# Lv2 = missdepLLoop(bTheta0, beta0, fn, X, Y, R, sXs)
# LpTv = missdepLpT(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# LpTv2 = missdepLpTLoop(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# Lpbv = missdepLpb(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# Lpbv2 = missdepLpbLoop(bTheta0, beta0, conDenfs, X, Y, R, sXs)



Cb = 69
CT = 0.05

beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, 3, 4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m) * 2
numRG = 100
eta = 1/(5*0.75*m*p)
tol = 1e-4
TrueParas = [beta0, bTheta0]
results = [{"beta0":beta0, "bTheta0":bTheta0, "eta":eta, "tol": tol}]
print(results)
for i in range(numRG):
    X = genXdis(n, m, p, type="unif") * 100
    Y = genYnorm(X, bTheta0, beta0, sigma=sigma)
    M = bTheta0 + X.matmul(beta0)
    R = genR(Y)
    keeprate = R.sum()/R.numel()
    sXs = genXdis(N, p, type="unif") * 100
    print(missdepL(torch.zeros(n, m), torch.zeros(p), fn, X, Y, R, sXs))
    # betahat, bThetahat, _ = MCGD(1000, X, Y, R, sXs, conDenfs, eta=1e-1, debug=1, Cb=Cb, CT=CT, tol=1e-4, log=1)
    betahat, bThetahat, _ = MCGD(1000, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, eta=eta, Cb=Cb, CT=CT, tol=tol, log=1, betainit=beta0, bThetainit=bTheta0)
    print(betahat)
                         # missdepL=missdepLLoop, missdepLpb=missdepLpbLoop, missdepLpT=missdepLpTLoop)
    errb = torch.norm(beta0-betahat)
    errT = torch.norm(bTheta0-bThetahat)
    results.append((errb.item(), betahat.norm().item(), errT.item(), bThetahat.norm().item()))
    print(
        f"The {i+1:>4}th/{numRG} ",
        f"Obs rate is {keeprate:.3f} ",
        f"The error of beta is {errb.item():>8.3f}, "
        f"The error of bTheta is {errT.item():>8.3f}."
    )

f = open("./simutest1.pkl", "wb")
pickle.dump(results, f)
f.close()
# betahat, bThetahat, _ = MCGD(1000, X, Y, R, sXs, conDenfs, eta=1e-1, debug=0, Cb=10, CT=0.8, tol=1e-4, log=1)
# print(torch.norm(beta0-betahat))
# print(torch.norm(beta0))
# print(torch.norm(bTheta0-bThetahat))
# print(torch.norm(bTheta0))
# print(betahat)
