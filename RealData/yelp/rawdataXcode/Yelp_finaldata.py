import pickle
from datetime import datetime
import numpy as np
from collections import defaultdict as ddict
from pprint import pprint
import os.path as osp
import ast

# columns: restuarants, m 
# rows: customers, n
# covariates:
# Reviews: [weekend, Ruseful, Rfunny, Rcool]
# users: [review_count, yelping_since, num_of_friends, Uuseful, Ufunny, Ucool,
#         fans, num_of_elite, av_stars, c_hot, c_more, c_p, c_cute, c_list, c_note, c_plain, c_cool, 
#         c_funny, c_writer, c_photos]
# bussiness: [stars, review_count, is_open, RestTakeOut, BPgarage, BPstreet, BPvalidated, BPlot, BPvalet, hrs_weekdays, hrs_weekend] 

with open("Y.pkl", "rb") as f:
    Y = pickle.load(f)

with open("X.pkl", "rb") as f:
    X = pickle.load(f)

with open("custrest.pkl", "rb") as f:
    custrest = pickle.load(f)
    
customers = custrest["customers"]
restaurants = custrest["restaurants"]

#xkeys = list(X.keys())
#xskeys = list(X[xkeys[0]].keys())
#print(X[xkeys[0]][xskeys[1]].keys())
def date2weekday(datestring):
    date = datetime.strptime(datestring, "%Y-%m-%d")
    weekday = date.weekday()
    if weekday >= 5:
        return 1
    else:
        return 0

def hours2num(hours):
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekends = ["Sunday", "Saturday"]
    def int2del(intstring):
        begint, endt  = intstring.split("-")[0], intstring.split("-")[1]
        beginth, begintm = begint.split(":")
        endth, endtm = endt.split(":")
        hrs = float(endth)-float(beginth) + (float(endtm) - float(begintm))/60
        if hrs == 0:
            return 24
        else:
            return hrs
    weekdayhours = np.sum([int2del(hours[weekday]) for weekday in weekdays if weekday in hours.keys()])
    weekendhours = np.sum([int2del(hours[weekend]) for weekend in weekends if weekend in hours.keys()])
    return weekdayhours, weekendhours

# sort the users and restaurants
customers = sorted(customers)
restaurants = sorted(restaurants)

m = 100
n = 100
p = 26
if not osp.isfile("Ymat.pkl"):
    Ymat = np.zeros((n, m)) - 1
    for idxcol, restaurant in enumerate(restaurants):
        datarest = Y[restaurant]
        userkeys = datarest.keys()
        for idxrow, customer in enumerate(customers):
            if customer in userkeys:
                Ymat[idxrow, idxcol] = datarest[customer]["stars"]
    with open("Ymat.pkl", "wb") as f:
        pickle.dump(Ymat, f)
with open("Ymat.pkl", "rb") as f:
    Ymat = pickle.load(f)


if not osp.isfile("Xmat.pkl"):
    Xmat = np.zeros((n, m, p)) - 1
    for idxcol, restaurant in enumerate(restaurants):
        datarest = Y[restaurant]
        userkeys = datarest.keys()
        datacust = X[restaurant]
        for idxrow, customer in enumerate(customers):
            if customer in userkeys:
                datestring = datarest[customer]["date"]
                datestring = datestring.split(" ")[0]
                Xmat[idxrow, idxcol, 0] = date2weekday(datestring)
                Xmat[idxrow, idxcol, 1] = datarest[customer]["UFC"][0]
                Xmat[idxrow, idxcol, 2] = datarest[customer]["UFC"][1]
                Xmat[idxrow, idxcol, 3] = datarest[customer]["UFC"][2]
                Xmat[idxrow, idxcol, 4] = datacust[customer]["review_count"]
                Xmat[idxrow, idxcol, 5] = len(datacust[customer]["friends"])
                Xmat[idxrow, idxcol, 6] = datacust[customer]["useful"]
                Xmat[idxrow, idxcol, 7] = datacust[customer]["funny"]
                Xmat[idxrow, idxcol, 8] = datacust[customer]["cool"]
                Xmat[idxrow, idxcol, 9] = datacust[customer]["fans"]
                Xmat[idxrow, idxcol, 10] = len(datacust[customer]["elite"])
                Xmat[idxrow, idxcol, 11] = datacust[customer]["average_stars"]
                Xmat[idxrow, idxcol, 12] = datacust[customer]["compliment_hot"]
                Xmat[idxrow, idxcol, 13] = datacust[customer]["compliment_more"]
                Xmat[idxrow, idxcol, 14] = datacust[customer]["compliment_profile"]
                Xmat[idxrow, idxcol, 15] = datacust[customer]["compliment_cute"]
                Xmat[idxrow, idxcol, 16] = datacust[customer]["compliment_list"]
                Xmat[idxrow, idxcol, 17] = datacust[customer]["compliment_note"]
                Xmat[idxrow, idxcol, 18] = datacust[customer]["compliment_plain"]
                Xmat[idxrow, idxcol, 19] = datacust[customer]["compliment_cool"]
                Xmat[idxrow, idxcol, 20] = datacust[customer]["compliment_funny"]
                Xmat[idxrow, idxcol, 21] = datacust[customer]["compliment_writer"]
                Xmat[idxrow, idxcol, 22] = datacust[customer]["compliment_photos"]
                Xmat[idxrow, idxcol, 23] = datacust["data"]["stars"]
                Xmat[idxrow, idxcol, 24] = datacust["data"]["review_count"]
                Xmat[idxrow, idxcol, 25] = datacust["data"]["is_open"]
                #Xmat[idxrow, idxcol, 26] = int(bool(datacust["data"]["attributes"]["RestaurantsTakeOut"]))
                #busparks = datacust["data"]["attributes"]["BusinessParking"]
                #busparks = ast.literal_eval(busparks)
                #Xmat[idxrow, idxcol, 27] = int(bool(busparks["garage"]))
                #Xmat[idxrow, idxcol, 28] = int(bool(busparks["street"]))
                #Xmat[idxrow, idxcol, 29] = int(bool(busparks["validated"]))
                #Xmat[idxrow, idxcol, 30] = int(bool(busparks["lot"]))
                #Xmat[idxrow, idxcol, 31] = int(bool(busparks["valet"]))
                #Xmat[idxrow, idxcol, 26] = hours2num(datacust["data"]["hours"])[0]
                #Xmat[idxrow, idxcol, 27] = hours2num(datacust["data"]["hours"])[1]
    with open("Xmat.pkl", "wb") as f:
        pickle.dump(Xmat, f)

with open("Xmat.pkl", "rb") as f:
    Xmat = pickle.load(f)
