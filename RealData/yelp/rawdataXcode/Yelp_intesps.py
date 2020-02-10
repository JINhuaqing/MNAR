import pickle
from datetime import datetime
import numpy as np
from collections import defaultdict as ddict
from pprint import pprint
import os.path as osp
#import ast
import json


userdict = ddict(dict)
bussdict = ddict(dict) 
if not osp.isfile("userdict.pkl"):
    for idx, oneobs in enumerate(open("user.json", "r")):
        print(f"Users, Current index is {idx}")
        oneobs = json.loads(oneobs)
        userdict[oneobs["user_id"]] = oneobs

    with open("userdict.pkl", "wb") as f:
        pickle.dump(userdict, f)

with open("userdict.pkl", "rb") as f:
    userdict = pickle.load(f)

if not osp.isfile("bussdict.pkl"):
    for idx, oneobs in enumerate(open("business.json", "r")):
        print(f"Business, Current index is {idx}")
        oneobs = json.loads(oneobs)
        bussdict[oneobs["business_id"]] = oneobs

    with open("bussdict.pkl", "wb") as f:
        pickle.dump(bussdict, f)

with open("bussdict.pkl", "rb") as f:
    bussdict = pickle.load(f)


def date2weekday(datestring):
    date = datetime.strptime(datestring, "%Y-%m-%d")
    weekday = date.weekday()
    if weekday >= 5:
        return 1
    else:
        return 0


N = 200000
p = 26
if not osp.isfile("Xsps.pkl"):
    Xsps = np.zeros((N, p))
    for idx, oneobs in enumerate(open("review.json", "r")):
        if idx == N:
            break
        print(f"current sample id is {idx}.")
        oneobs = json.loads(oneobs)
        user_id = oneobs["user_id"]
        buss_id = oneobs["business_id"]
        cuser = userdict[user_id]
        crest = bussdict[buss_id]

        datestring = oneobs["date"]
        datestring = datestring.split(" ")[0]
        Xsps[idx, 0] = date2weekday(datestring)

        Xsps[idx, 1] = oneobs["useful"]
        Xsps[idx, 2] = oneobs["funny"]
        Xsps[idx, 3] = oneobs["cool"]
        Xsps[idx, 4] = cuser["review_count"]
        Xsps[idx, 5] = len(cuser["friends"])
        Xsps[idx, 6] = cuser["useful"]
        Xsps[idx, 7] = cuser["funny"]
        Xsps[idx, 8] = cuser["cool"]
        Xsps[idx, 9] = cuser["fans"]
        Xsps[idx, 10] = len(cuser["elite"])
        Xsps[idx, 11] = cuser["average_stars"]
        Xsps[idx, 12] = cuser["compliment_hot"]
        Xsps[idx, 13] = cuser["compliment_more"]
        Xsps[idx, 14] = cuser["compliment_profile"]
        Xsps[idx, 15] = cuser["compliment_cute"]
        Xsps[idx, 16] = cuser["compliment_list"]
        Xsps[idx, 17] = cuser["compliment_note"]
        Xsps[idx, 18] = cuser["compliment_plain"]
        Xsps[idx, 19] = cuser["compliment_cool"]
        Xsps[idx, 20] = cuser["compliment_funny"]
        Xsps[idx, 21] = cuser["compliment_writer"]
        Xsps[idx, 22] = cuser["compliment_photos"]
        Xsps[idx, 23] = crest["stars"]
        Xsps[idx, 24] = crest["review_count"]
        Xsps[idx, 25] = crest["is_open"]


    with open("Xsps.pkl", "wb") as f:
        pickle.dump(Xsps, f)
