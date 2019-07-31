import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import joint_optimize

def BayOptMCGD(train_X, train_Y, bounds, beta=0.1):
    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    
    UCB = UpperConfidenceBound(gp, beta=beta)
    
    candidate = joint_optimize(UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
    return candidate

# train_X = torch.rand(10, 2)
# Y = 1 - torch.norm(train_X - 0.5, dim=-1) + 0.1 * torch.rand(10)
# train_Y = (Y - Y.mean()) / Y.std()
# for i in range(10):
#     bounds = torch.stack([torch.zeros(2), torch.ones(2)]) 
#     
#     
#     candidate = BayOptMCGD(train_X, train_Y, bounds)
# 
#     train_X = torch.cat([train_X, candidate], dim=0)
#     lens, _ = train_X.shape
#     Y = 1 - torch.norm(train_X - 0.5, dim=-1) + 0.1 * torch.rand(lens)
#     train_Y = (Y - Y.mean()) / Y.std()
# 


