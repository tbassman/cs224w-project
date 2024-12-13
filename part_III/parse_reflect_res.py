import json
import os

directory = "/Users/vctrlin/data/stanford_online/winter_quater/CS224W/final_project/GNN_RAG/cs224w"

res = os.path.join(directory, "_test_reflect.info")


tmp_data = {}
cnt = 0
tot = 0
with open(res, "r") as f:
    for line in f:
        tot += 1
        tmp = json.loads(line)
        if tmp["em"] == 1:
            cnt += 1



print("improve rate: " + str(round((cnt / tot) * 100, 2)))
