import pickle
import json
from collections import defaultdict as ddict 
from collections import Counter 
import os.path as osp
import numpy.random as npr
import numpy as np
from pprint import pprint

CityDic = ddict(set)
target = "business_id"
sfile = "CityRest.pkl"
lfile = "business.json"

if not osp.isfile(sfile):
    for idx, oneobs in enumerate(open(lfile, "r")):
        print(f"Current index is {idx}")
        oneobs = json.loads(oneobs)
        CityDic[oneobs["city"]].add(oneobs[target])
        if idx == 1000:
            pass#break
    
    with open(sfile, "wb") as f:
        pickle.dump(CityDic, f)

with open(sfile, "rb") as f:
    CityDic = pickle.load(f)

CityC = Counter()

for key, dat in CityDic.items():
    CityC[key] = len(dat)    

print(CityC.most_common(140))

