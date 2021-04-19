import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import metrics
from pathlib import Path
from collections import defaultdict as ddict
from easydict import EasyDict as edict
# %matplotlib inline

# +
root = Path("./")
alls = list(root.glob("MNARx*.pkl"))

sortf = lambda x: int(x.stem.split("_")[-1])
alls = sorted(alls, key=sortf)
# -

alls

with open(alls[0], "rb") as f:
    dat = pickle.load(f)


# +
def path2AUCs(path, typ="MNARres"):
    aucs = []
    with open(path, "rb") as f:
        ress = pickle.load(f)
    ress = ress[typ]
    for res in ress:
        prob, gt = res
        prob, gt = prob, gt
        aucs.append(metrics.roc_auc_score(gt, prob))
    return aucs

def paths2AUCss(mp, typ="MNARres"):
    aucss = ddict(list)
    for p in mp:
        idx = sortf(p)
        aucss[idx] = path2AUCs(p, typ)
    
    return aucss


# -

resdic = {}
resdic["MNARres"] = None
resdic["EMres"] = None
resdic["MARres"] = None

for key in resdic.keys():
    resdic[key] = paths2AUCss(alls, key)


# +
# This two functions are for obtain the difference between results
def DiffFun(res1, res2):
    outRes = {}
    for key in res1.keys():
        diff = np.array(res1[key])  - np.array(res2[key])
        outRes[key] = diff
    return outRes

def res2diff(resdic, base="MNARres"):
    outRes = {}
    baseRes = resdic[base]
    for key in resdic.keys():
        if key != base:
            keyN = key + "x" + base
            cRes = resdic[key]
            outRes[keyN] = DiffFun(cRes, baseRes)
    return outRes


# -

diffRes = res2diff(resdic)

# +
typs = ["bo", "r*", "yh", "g+", "c.", "kH"]

plt.figure(figsize=[15, 5])
for ii, OR in enumerate([501, 50, 65, 80]):
    plt.subplot(1, 4, ii+1)
    idx = 0
    for nKey in diffRes.keys():
        aucs = diffRes[nKey]
        val = aucs[OR]
        plt.plot(val, typs[idx], label=str(OR)+ " " + nKey)
        idx += 1
    plt.ylim([-0.03, 0])
    plt.legend()
# -


