import pickle
import json
from collections import defaultdict as ddict 
from collections import Counter as Cntr
import os.path as osp
import numpy.random as npr
import numpy as np


Restdic = ddict(set)
RestCnt = Cntr()

if not osp.isfile("RestDicCnt.pkl"):
    for idx, oneobs in enumerate(open("review.json", "r")):
        print(f"Current index is {idx}")
        oneobs = json.loads(oneobs)
        Restdic[oneobs["business_id"]].add(oneobs["user_id"])
        RestCnt[oneobs["business_id"]]  += 1
        if idx == 1000:
            pass#break
    
    with open("RestDicCnt.pkl", "wb") as f:
        pickle.dump([Restdic, RestCnt], f)

with open("RestDicCnt.pkl", "rb") as f:
    RestDicCnt = pickle.load(f)

with open("CityRest.pkl", "rb") as f:
    CityDic = pickle.load(f)

CityC = Cntr()

for key, dat in CityDic.items():
    CityC[key] = len(dat)    

cities = CityC.most_common(140)
# choose the city
ccity = cities[0][0]

Restdic, RestCnt = RestDicCnt
n = 100 # number of restaurants
m = 100 # number of customers 
numtt = n * m # total number of reviews 
num = m # number of non-missing reviews


def Randmfn(ttCusts, m=100):
    idxs = list(range(len(ttCusts)))
    idxs = npr.permutation(idxs)
    return np.array(ttCusts)[idxs[:m]]

# select the n most frequent restaurants
subRests = Cntr()
for key, nn in RestCnt.items():
    if key in CityDic[ccity]:
        subRests[key] = nn

Rests = subRests.most_common(n)
#Rests = RestCnt.most_common(n)
currentRest = Rests[0][0]
ttCusts = list(Restdic[currentRest])
Custs = ttCusts
#Custs = Randmfn(ttCusts,m=m)
Custs = set(Custs)

res = Cntr()
for Cust in Custs:
    for rest, _ in Rests:
        custofrest = Restdic[rest]
        if Cust in custofrest:
            res[Cust] += 1


ress = res.most_common(100)
numlist = [i[1] for i in ress]
num = np.sum(numlist) 
print(1-num/numtt)


Customers = [i[0] for i in ress]
Restaurants = [i[0] for i in Rests]
print(Customers)
print(Restaurants)
if not osp.isfile("custrest.pkl"):
    with open("custrest.pkl", "wb") as fi:
        pickle.dump({"customers":Customers, "restaurants":Restaurants}, fi)


