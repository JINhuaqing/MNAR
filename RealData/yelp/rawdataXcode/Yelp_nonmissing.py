import pickle
import json
from collections import defaultdict as ddict 
from collections import Counter as Cntr
import os.path as osp


Restdic = ddict(set)
RestCnt = Cntr()
threshold = 20

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

Restdic, RestCnt = RestDicCnt

sortedCnt = RestCnt.most_common()
MostKey = sortedCnt[0][0]

RestCnt[MostKey] = 0
sortedCnt = RestCnt.most_common()
userCollect = Restdic[MostKey]

keepedKeys = [MostKey]
curMaxNum = 10000

while (curMaxNum >= threshold) and (len(keepedKeys)<= 100):
    numCnt = Cntr()
    for key, num in sortedCnt:
        curSet = Restdic[key] 
        curNum = len(userCollect.intersection(curSet))
        numCnt[key] = curNum
        if num <= threshold:
            break
    
    #print(numCnt.most_common(10))
    curMaxkey, curMaxNum = numCnt.most_common(1)[0]
    RestCnt[curMaxkey] = 0 
    sortedCnt = RestCnt.most_common()
    
    # record current resturant
    keepedKeys.append(curMaxkey)
    
    
    # update current user collect
    userCollect.intersection_update(Restdic[curMaxkey])
    print(f"The number of elements in userCollect is {curMaxNum}."
          f"The number of recorded restaurants is {len(keepedKeys)}."
          )



