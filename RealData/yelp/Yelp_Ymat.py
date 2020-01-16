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
        if idx == 1000:
            pass #break
    
    with open("Y.pkl", "wb") as f:
        pickle.dump(Y, f)

with open("Y.pkl", "rb") as f:
    Y = pickle.load(f)
#tsum = 0
#for key, item in Y.items():
#    tsum += len(item)



