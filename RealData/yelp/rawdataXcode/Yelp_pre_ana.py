# this script is to clean the yelp data
import numpy as np
import json
import pprint
from collections import Counter

# n: number of users
# m: number of restaurants
# I use Y contains stars 

# I use following variables:

## restaurant-related variables:
# state (dummy variables, 34 dim), binary
# stars (average stars), float
# attributes ( 6 variables), binary
# review_count, integer
# is_open, binary
# categories (cannot be used)

## user-related variables:
# review_count, integer
# useful, integer
# funny, integer 
# cool, integer
# fans, integer
# average_stars, float
# compliment_xx (11 variables), integers

## user-restaurant-related variables:
# useful, integer
# funny, integer
# cool, integer
# number_day_in_yelp (data-yelping_since), integer


# function to load json data
def Load_json(jsonfile, keys=None):
    if keys is None:
        return  [json.loads(oneobs) for oneobs in open(jsonfile, "r")]
    else:
        return  [json.loads(oneobs)[keys] for oneobs in open(jsonfile, "r")]

#bussdata = Load_json("business.json", "business_id")
busCnt = Load_json("review.json", "business_id")
userCnt = Load_json("review.json", "user_id")
#userdata = Load_json("user.json", "user_id")
#checkindata = Load_json("checkin.json")

buscnt = Counter(busCnt)
usercnt = Counter(userCnt)
print(buscnt.most_common(100))
print(usercnt.most_common(100))





