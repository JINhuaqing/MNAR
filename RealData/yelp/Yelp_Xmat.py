import pickle
from collections import defaultdict as ddict
import json
from pprint import pprint
import os.path as osp


with open("RestDicCnt.pkl", "rb") as f:
    RestDicCnt = pickle.load(f)
with open("custrest.pkl", "rb") as f:
    custrest = pickle.load(f)

allRestsdic, _ = RestDicCnt
selcusts = set(custrest["customers"])
selrests = set(custrest["restaurants"])

if not osp.isfile("Y.pkl"):
    Y = ddict(dict)
    for idx, oneobs in enumerate(open("review.json", "r")):
        print(f"Current index is {idx}")
        oneobs = json.loads(oneobs)
        crest = oneobs["business_id"]
        ccust = oneobs["user_id"] 
        if (crest in selrests) and (ccust in selcusts):
            Y[crest][ccust] = oneobs["stars"]
    
    with open("Y.pkl", "wb") as f:
        pickle.dump(Y, f)

with open("Y.pkl", "rb") as f:
    Y = pickle.load(f)
#tsum = 0
#for key, item in Y.items():
#    tsum += len(item)
#    print(len(item))


if not osp.isfile("Xrest.pkl"):
    Xrest = {}
    for idx1, oneobs in enumerate(open("business.json", "r")):
        print(f"Current restaurant index is {idx1}")
        oneobs = json.loads(oneobs)
        crest = oneobs["business_id"]
        if crest in selrests:
            Xrest[crest] = oneobs
    with open("Xrest.pkl", "wb") as f:
        pickle.dump(Xrest, f)

with open("Xrest.pkl", "rb") as f:
    Xrest = pickle.load(f)

if not osp.isfile("Xuser.pkl"):
    Xuser = {}
    for idx2, userobs in enumerate(open("user.json", "r")):
        print(f"Current user index is {idx2}")
        userobs = json.loads(userobs)
        ccust = userobs["user_id"]
        if ccust in selcusts:
            Xuser[ccust] = userobs
    with open("Xuser.pkl", "wb") as f:
        pickle.dump(Xuser, f)

with open("Xuser.pkl", "rb") as f:
    Xuser = pickle.load( f)

if not osp.isfile("X.pkl"):
    X = ddict(dict)
    for crest in selrests:
        X[crest]["data"] = Xrest[crest]
        for ccust in Y[crest].keys():
            X[crest][ccust] = Xuser[ccust]
    with open("X.pkl", "wb") as f:
        pickle.dump(X, f)
with open("X.pkl", "rb") as f:
    X = pickle.load(f)

tsum = 0
for key, item in X.items():
    tsum += len(item)
print(tsum)
pprint(item)
    

