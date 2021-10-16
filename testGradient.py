import torch
import numpy as np
from utils import *
from confs import *
from utilsNew import * 

import numpy.random as npr
from torch.distributions.normal import Normal
from prettytable import PrettyTable
from scipy.stats import truncnorm
import time
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
np.random.seed(0) #numpy
torch.backends.cudnn.deterministic=True # cudnn
cudaid = 2
torch.cuda.set_device(cudaid)
torch.set_default_tensor_type(torch.cuda.DoubleTensor)

# +
n = m = 50
p = 100
prob = 0.2

sigmaY = 1.0

# generate the parameters
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7))) 
bTheta0 = genbTheta(n, m, sigVs=np.array([10, 9, 8, 7, 6])*10) 
initbetapref = 1 + (torch.rand(p)-1/2)/40  #[0.875, 1.125]
initthetapref = 1 + (torch.rand(n, m)-1/2)/4
betainit = beta0 * initbetapref
bThetainit = bTheta0 * initthetapref
#betainit = torch.cat((torch.tensor([0.8, 0, 1.5, 0, -2.5, -4.3, 4.8]), torch.zeros(p-7))) * 10
#bThetainit = genbTheta(n, m, sigVs=np.array([15, 10, 7, 5, 5])/2) 

#X = genXUnif(n, m, p) 
X = genXBin(n, m, p, prob=prob) 
Y = genYnorm(X, bTheta0, beta0, sigmaY)
# around 95% missing rate
inps = {}
inps[50] = - 8.0
inps[100] = - 5.93
inps[200] = -5.30
inps[400] = -5.2
inps[800] = -5.15
inps[1600] = -5.1
R = genR(Y, "linear", inp=inps[n], slop=1)
R.to_dense().mean()
# -

f = fn
f2 = fn2
f22 = fn2
conDenfs = [f, f2, f22]

beta = beta0 + 0
bTheta = bTheta0 + 0

# ## Under MNAR

# Gradient for $\beta$

torch.norm(missdepLpb(bTheta, beta, conDenfs, X, Y, R, fct=10)- lossLpb(bTheta, beta, conDenfs, X, Y, R, fct=10))

exGra = lossLpb(bTheta, beta, conDenfs, X, Y, R, fct=10)
torch.norm(exGra)

h = 1e-5
numGra = torch.zeros(p)
flag = 0
for i in tqdm(range(p)):
    beta1 = beta.clone()
    beta2 = beta.clone()
    beta1[i] = beta1[i] + h
    beta2[i] = beta2[i] - h
    fv1 = lossL(bTheta, beta1, f, X, Y, R, fct=10, is_logf=False, N=10000)
    fv2 = lossL(bTheta, beta2, f, X, Y, R, fct=10, is_logf=False, N=10000)
    numGra[i] = (fv1-fv2)/2/h

torch.norm(numGra)

torch.norm(numGra-exGra)

# Gradient for $\Theta$

torch.norm(missdepLpT(bTheta, beta, conDenfs, X, Y, R, fct=10)- lossLpT(bTheta, beta, conDenfs, X, Y, R, fct=10))

exGra = lossLpT(bTheta, beta, conDenfs, X, Y, R, fct=10)
torch.norm(exGra)

h = 1e-5
numGra = torch.zeros((n, m))
flag = 0
for i in tqdm(range(n)):
    for j in range(m):
        cR = R.to_dense()[i, j]
        if cR == 1:
            flag += 1
            bTheta1 = bTheta.clone()
            bTheta2 = bTheta.clone()
            bTheta1[i, j] = bTheta1[i, j] + h
            bTheta2[i, j] = bTheta2[i, j] - h
            fv1 = lossL(bTheta1, beta, f, X, Y, R, fct=1, is_logf=False, N=10000)
            fv2 = lossL(bTheta2, beta, f, X, Y, R, fct=1, is_logf=False, N=10000)
            numGra[i, j] = (fv1-fv2)/2/h

torch.norm(numGra)

torch.norm(numGra-exGra)



# ### Under MAR

# Gradient for $\beta$

exGra = marLossLpb(bTheta, beta, conDenfs, X, Y, R)
torch.norm(exGra)

h = 1e-5
numGra = torch.zeros(p)
flag = 0
for i in tqdm(range(p)):
    beta1 = beta.clone()
    beta2 = beta.clone()
    beta1[i] = beta1[i] + h
    beta2[i] = beta2[i] - h
    fv1 = marLossL(bTheta, beta1, f, X, Y, R, is_logf=False,)
    fv2 = marLossL(bTheta, beta2, f, X, Y, R, is_logf=False,)
    numGra[i] = (fv1-fv2)/2/h

torch.norm(numGra)

torch.norm(numGra-exGra)

# Gradient for $\Theta$

exGra = marLossLpT(bTheta, beta, conDenfs, X, Y, R)
torch.norm(exGra)

h = 1e-5
numGra = torch.zeros((n, m))
flag = 0
for i in tqdm(range(n)):
    for j in range(m):
        cR = R.to_dense()[i, j]
        if cR == 1:
            flag += 1
            bTheta1 = bTheta.clone()
            bTheta2 = bTheta.clone()
            bTheta1[i, j] = bTheta1[i, j] + h
            bTheta2[i, j] = bTheta2[i, j] - h
            fv1 = marLossL(bTheta1, beta, f, X, Y, R, is_logf=False)
            fv2 = marLossL(bTheta2, beta, f, X, Y, R, is_logf=False)
            numGra[i, j] = (fv1-fv2)/2/h

torch.norm(numGra)

torch.norm(numGra-exGra)



# ### Under EM

# Gradient for $\beta$

exGra = emLossLpb(bTheta, beta, conDenfs, X, Y)
torch.norm(exGra)

h = 1e-5
numGra = torch.zeros(p)
flag = 0
for i in tqdm(range(p)):
    beta1 = beta.clone()
    beta2 = beta.clone()
    beta1[i] = beta1[i] + h
    beta2[i] = beta2[i] - h
    fv1 = emLossL(bTheta, beta1, f, X, Y, is_logf=False)
    fv2 = emLossL(bTheta, beta2, f, X, Y, is_logf=False)
    numGra[i] = (fv1-fv2)/2/h

torch.norm(numGra)

torch.norm(numGra-exGra)

# Gradient for $\Theta$

exGra = emLossLpT(bTheta, beta, conDenfs, X, Y)
torch.norm(exGra)

h = 1e-5
numGra = torch.zeros((n, m))
flag = 0
for i in tqdm(range(n)):
    for j in range(m):
        flag += 1
        bTheta1 = bTheta.clone()
        bTheta2 = bTheta.clone()
        bTheta1[i, j] = bTheta1[i, j] + h
        bTheta2[i, j] = bTheta2[i, j] - h
        fv1 = emLossL(bTheta1, beta, f, X, Y, is_logf=False)
        fv2 = emLossL(bTheta2, beta, f, X, Y, is_logf=False)
        numGra[i, j] = (fv1-fv2)/2/h

torch.norm(numGra)

torch.norm(numGra-exGra)


