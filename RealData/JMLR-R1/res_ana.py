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
import pandas as pd
# %matplotlib inline

# +
root = Path("./")
alls = list(root.glob("MNARx*.pkl"))

sortf = lambda x: int(x.stem.split("_")[-1])
alls = sorted(alls, key=sortf)
# -

alls


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

mimiRes = pd.read_csv("mimi.csv")

# +
mimiResDict = {}
kys = [5, 6, 7, 8]
for ky in kys:
    key = "OR"+str(ky)
    mimiResDict[10*ky] = list(mimiRes[key])
    
resdic["MIMIres"] = mimiResDict


# -

def res2mean(resdic):
    mRes = {}
    for key in resdic.keys():
        cRes = resdic[key]
        tV1 = []
        tV2 = []
        for ky, v in cRes.items():
            tV1.append(np.mean(v))
            tV2.append(ky)
        mRes[key] = {}
        mRes[key]["v"] = np.array(tV1)
        mRes[key]["idx"] = np.array(tV2)
    return mRes


resM = res2mean(resdic)

typs = ["b-", "r--", "y-.", "g:", "c.", "kH"]
flag = 0
plt.figure(figsize=[6, 6], dpi=200)
for ky, val in resM.items():
    plt.plot(1 - val['idx']/1000, val['v'], typs[flag], label=ky[:-3])
    flag += 1
plt.ylabel("AUC")
plt.xlabel("Missing rate")
plt.ylim([0.56, 0.75])
plt.legend()
plt.savefig("./realdata_auc.jpg", bbox_inches="tight")


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
            keyN = base[:-3] + "-" + key[:-3]
            cRes = resdic[key]
            outRes[keyN] = DiffFun(baseRes, cRes)
    return outRes


# -

diffRes = res2diff(resdic)

# +
typs = ["b+", "r*", "yo", "gh", "c.", "kH"]

plt.figure(figsize=[24, 6], dpi=200)
plt.subplots_adjust(wspace=0.3, hspace =1)
for ii, OR in enumerate([80, 70, 60, 50]):
    plt.subplot(1, 4, ii+1)
    idx = 0
    for nKey in diffRes.keys():
        aucs = diffRes[nKey]
        val = aucs[OR]
        plt.title("Missing rate " + str(100- OR/10)+ "%")
        plt.plot(val, typs[idx], label= nKey)
        idx += 1
    plt.ylim([0, 0.14])
    plt.ylabel("AUC difference")
    plt.xlabel("Repetition index")
    plt.legend(loc=7)
plt.savefig("./realdata_auc_diff.jpg", bbox_inches="tight")
# -

