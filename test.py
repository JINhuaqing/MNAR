import torch
from confs import fln, fln2, fln2_raw
from utils import *
import numpy as np

torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
np.random.seed(0) #numpy
torch.backends.cudnn.deterministic=True # cudnn
cudaid = 3
torch.cuda.set_device(cudaid)

#------------------------------------------------------------------------------------
# Whether GPU is available, 
torch.set_default_tensor_type(torch.cuda.DoubleTensor)
n = m = 50
p = 50
beta0 = torch.cat((torch.tensor([1.0, 0, 2, 0, -3, -4, 5]), torch.zeros(p-7))) * 10
bTheta0 = genbTheta(n, m, sigVs=np.array([10, 9, 8, 7, 6])/2) 
X = genXBin(n, m, p, prob=0.1) 
Y = genYnorm(X, bTheta0, beta0, 0.1)
Y = (Y>0).type(torch.cuda.DoubleTensor)
X = X.to_dense()

mat1 = fln2(Y, X, beta0)
mat2 = fln2_raw(Y, X, beta0)
print(torch.abs(mat1-mat2).sum())
