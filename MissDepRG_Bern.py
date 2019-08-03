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
# cuda = False
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

prob = 0.1
X = genXdis(n, m, p, type="Bern", prob=prob) 
#X = torch.tensor(np.random.randint(0, 2, (m, n, p))).float()
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, 3, 4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m) * 7 
M = bTheta0 + X.matmul(beta0)
Y = genYnorm(X, bTheta0, beta0, sigma=sigma)
R = genR(Y)
# print(X.matmul(beta0).abs().mean(), bTheta0.abs().mean())
#Ds2, Dh2 = Dshlowerfnorm(Y, X, beta0, bTheta0, sigma)
#print(2*Dh2+2*Ds2**2)
print(R.sum()/R.numel())
sXs = genXdis(N, p, type="Bern", prob=prob) 
conDenfs = [fn, fn2, fn22]
# etascaleinv = missdepLpbbLoop(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# etascale = etascaleinv.inverse()
# U, S, V = etascale.svd()
# print(S.max())

# Lv = missdepL(bTheta0, beta0, fn, X, Y, R, sXs)
# Lvb = LBern(bTheta0, beta0, fn, X, Y, R, prob=prob)
# print(Lv-Lvb, Lvb)
# t0 = time.time()
# LpTv = missdepLpT(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# LpTvb = LpTBern(bTheta0, beta0, conDenfs, X, Y, R, prob=prob)
# print((LpTv-LpTvb).norm(), LpTvb.norm())
# Lpbv = missdepLpb(bTheta0, beta0, conDenfs, X, Y, R, sXs)
# Lpbvb = LpbBern(bTheta0, beta0, conDenfs, X, Y, R, prob=prob)
# print((Lpbv - Lpbvb).norm(), Lpbvb.norm())
# t1 = time.time()
# print(t1-t0)
# raise SystemExit




Cbpool = np.exp(np.linspace(np.log(0.1), np.log(1e4), 200))
CTpool = Cbpool/10
STpool = np.exp(np.linspace(np.log(100), np.log(1e4), 200))
np.random.shuffle(Cbpool)
np.random.shuffle(CTpool)
np.random.shuffle(STpool)

numRG = 100
# eta = 1/(5*0.75*m*p)
eta = 0.01 
tol = 1e-4
TrueParas = [beta0, bTheta0]
results = [{"beta0":beta0.cpu(), "bTheta0":bTheta0.cpu(), "eta":eta, "tol": tol}]
betainit = torch.rand(p)
idxs = torch.randperm(p)[:p-8]
betainit[idxs] = 0
betainit = beta0* 1.5
bThetainit = bTheta0 * 1.5

print(results)
Errs = []
for i in range(numRG):
    len1, len2, len3  = len(Cbpool), len(CTpool), len(STpool)
    idx1, idx2, idx3 = np.random.randint(0, len1), np.random.randint(0, len2), np.random.randint(0, len3)
    Cb, CT, ST = Cbpool[idx1], CTpool[idx2], STpool[idx3]
    print(f"The {i+1}/{numRG}, Cb is {Cb:>8.4g}, CT is {CT:>8.4g}, ST is {ST:>8.4g}")
    try:
       betahat, bThetahat, _, numI, Berrs, Terrs = MCGDBern(1000, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, eta=eta, Cb=Cb, CT=CT, tol=tol, log=0, ST=ST, prob=prob, betainit=betainit, bThetainit=bThetainit, ErrOpts=1)
    except RuntimeError:
        results.append((-100, Cb, -100, -100,  CT, -100, -100, ST))
        Errs.append([])
        print(
            f"The {i+1}th/{numRG},"
            f"Iteration Fails!"
        )
    else:
        errb = torch.norm(beta0-betahat)
        errT = torch.norm(bTheta0-bThetahat)
        results.append((numI, Cb, errb.item(), betahat.norm().item(), CT, errT.item(), bThetahat.norm().item(), ST))
        Errs.append([Berrs, Terrs])
        print(
            f"The {i+1}th/{numRG},"
            f"The Iteration number is {numI}, "
            f"The error of beta is {errb.item():.3f}, "
            f"The error of bTheta is {errT.item():.3f}."
        )

f = open("./outputs/RandGrid_Bern_2w_01_001_errs_init15.pkl", "wb")
pickle.dump([results, Errs], f)
f.close()
# betahat, bThetahat, _ = MCGD(1000, X, Y, R, sXs, conDenfs, eta=1e-1, debug=0, Cb=10, CT=0.8, tol=1e-4, log=1)
# print(torch.norm(beta0-betahat))
# print(torch.norm(beta0))
# print(torch.norm(bTheta0-bThetahat))
# print(torch.norm(bTheta0))
# print(betahat)
# betahat, bThetahat, _ = MCGD(1000, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, eta=eta, Cb=10, CT=0.1, tol=tol, log=2, sigmax=sigmax, sigma=sigma)
