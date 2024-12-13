import json
import os

# set working directory
directory = "path to working directory"

# get the test results
res = os.path.join(directory, "_test_reflect.info")

# parse the test result from json format per line 
tmp_data = {}
cnt = 0
tot = 0
with open(res, "r") as f:
    for line in f:
        tot += 1
        tmp = json.loads(line)

        # count the case if answer is among candidates, "exact match" == 1
        if tmp["em"] == 1:
            cnt += 1

# calculate the improvement percentage
print("improve rate: " + str(round((cnt / tot) * 100, 2)))
