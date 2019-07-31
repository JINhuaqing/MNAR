from utilities import *
from BayOptCode import BayOptMCGD 
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
N = 1000
sigmax = np.sqrt(1/3)

X = genXdis(n, m, p, type="mvnorm", sigmax=sigmax) 
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, 3, 4, 5]), torch.zeros(p-7)))
bTheta0 = genbTheta(n, m) * 7
M = bTheta0 + X.matmul(beta0)
Y = genYnorm(X, bTheta0, beta0, sigma=sigma)
R = genR(Y)
print(R.sum()/R.numel())
sXs = genXdis(N, p, type="mvnorm", sigmax=sigmax) 
conDenfs = [fn, fn2, fn22]

f = open("./RandGridinit.pkl", "rb") 
data = pickle.load(f)
f.close()
dat = np.array(data[1:])
idxrm = dat != 1000
idxrm =  idxrm.all(axis=-1)
dat = dat[idxrm]
idxrm = dat[:, -2] != 0
dat = dat[idxrm]



numInit = 4
train_Y = dat[:numInit, (2, -3)]
train_X = dat[:numInit, (1, 4, -1)]
num, _ = train_X.shape
train_X = torch.tensor(train_X).float()
train_Y = torch.tensor(train_Y).float()
train_Y = train_Y * torch.cat([torch.ones(num, 1), torch.ones(num, 1)*0.01], dim=1)
train_Y = train_Y.sum(dim=1)
meanY = train_Y.mean()
sdY = train_Y.var().sqrt()
train_Y = (train_Y - meanY)/sdY




bounds = torch.stack([torch.tensor([0.1, 1e-3, 1000]),
    torch.tensor([10000, 1e4, 1e6])]) # Cb, CT, sigmabTheta

numRG = 100
# eta = 1/(5*0.75*m*p)
eta = 0.01 
tol = 1e-4
TrueParas = [beta0, bTheta0]
results = [{"beta0":beta0, "bTheta0":bTheta0, "eta":eta, "tol": tol}]
print(results)
for i in range(numRG):
   candidate = BayOptMCGD(train_X, train_Y, bounds, beta=0.01)
   Cb, CT, ST = candidate.squeeze()
   print(
           f"The candidate point is ({Cb:.3f}, {CT:.3f}, {ST:.3f})"
           )
   betahat, bThetahat, _, numI = MCGDnormal(1000, X, Y, R, TrueParas=TrueParas, eta=eta, Cb=Cb, CT=CT, tol=tol, log=0, ST=ST, sigmax=sigmax,sigma=sigma)
   errb = torch.norm(beta0-betahat)
   errT = torch.norm(bTheta0-bThetahat)
   newY = (errb + errT*0.01).unsqueeze(0)  * (((numI==1000)+1)**5)
   train_X = torch.cat([train_X, candidate], dim=0)
   train_Y = torch.cat([train_Y, newY])
   results.append((Cb, CT, ST, newY.item(), errb.item(), errT.item()))
   print(
       f"The {i+1}th/{numRG},"
       f"The Iteration number is {numI}, "
       f"The error of beta is {errb.item():.3f}, "
       f"The error of bTheta is {errT.item():.3f}."
       f"The newY is {newY.item():.3f}"
   )

f = open("./BayOpt.pkl", "wb")
pickle.dump(results, f)
f.close()
# betahat, bThetahat, _ = MCGD(1000, X, Y, R, sXs, conDenfs, eta=1e-1, debug=0, Cb=10, CT=0.8, tol=1e-4, log=1)
# print(torch.norm(beta0-betahat))
# print(torch.norm(beta0))
# print(torch.norm(bTheta0-bThetahat))
# print(torch.norm(bTheta0))
# print(betahat)
# betahat, bThetahat, _ = MCGD(1000, X, Y, R, sXs, conDenfs, TrueParas=TrueParas, eta=eta, Cb=10, CT=0.1, tol=tol, log=2, sigmax=sigmax, sigma=sigma)
