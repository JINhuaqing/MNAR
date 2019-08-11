import numpy as np
from pickle import load
from prettytable import PrettyTable
import pickle

with open("./RandGrid_Bern_lg_5w_01_r5s5_09_100_tol5_eta2.pkl", "rb") as f:
    dataall = load(f)

data, errss = dataall[0], dataall[1]

# data[0] = data[0][1:]
# with open("./RandGrid_Bern_2w_01_001_errs_cpu.pkl", "wb") as f:
#     pickle.dump(data, f)
# 
# rar

truepara = data[0]
print(
        f"The norm of true Beta is {truepara['beta0'].norm():>8.5g}, "
        f"The norm of true bTheta is {truepara['bTheta0'].norm():>8.5g}, "
        f"The eta is {truepara['eta']:.5g}, "
        f"The tolerace is {truepara['tol']:.5g}, "
        )

data = data[1:]
dataarr = np.array(data)
errssarr = np.array(errss)
tb = PrettyTable([" ", "Iteration times", "Cb", "CT", "ST", 
    "error of beta", "norm of betahat", "error of bTheta", "norm of bThetahat"])
# remove the results which didn't converge
kpidx = (dataarr != -100).all(axis=1)
rmddata = dataarr[kpidx]
rmderrss = errssarr[kpidx]

idxnew = [0, 1, 4, 7, 2, 3, 5, 6]
formatstr = "{:>8}, " +  "{:>5.0f}, "+ "{:>10.5f}, " * 2 + "{:>12.1f}, " + "{:>10.5f}, " * 4
# find hyperparameters of the minimal errors of beta and bTheta, respectively
minidx = np.argmin(rmddata, axis=0)
betaidx, bThetaidx = minidx[2], minidx[-3]

bres = rmddata[betaidx][idxnew]
Tres = rmddata[bThetaidx][idxnew]
bres = formatstr.format("minB", *list(bres)).split(",")[:-1]
Tres = formatstr.format("minT", *list(Tres)).split(",")[:-1]

# remove the results which reach the maxmal iteration number 
kpidx2 = rmddata[:, 0] != 1000
rmddata2 = rmddata[kpidx2]
#print(np.round(rmddata2, 3)[:, [0, 2, 5]])
rmderrss2 = rmderrss[kpidx2]
minidx2 = np.argmin(rmddata2, axis=0)
betaidx2, bThetaidx2 = minidx2[2], minidx2[-3]

bres2 = rmddata2[betaidx2][idxnew]
Tres2 = rmddata2[bThetaidx2][idxnew]
bres2 = formatstr.format("minBCon", *list(bres2)).split(",")[:-1]
Tres2 = formatstr.format("minTCon", *list(Tres2)).split(",")[:-1]

tb.add_row(bres)
tb.add_row(bres2)
tb.add_row(Tres)
tb.add_row(Tres2)
print(tb)
