import json
import os

directory = "/Users/vctrlin/data/stanford_online/winter_quater/CS224W/final_project/GNN_RAG/cs224w"

dataset_file = os.path.join(directory, "test.json")
shortest_cases = os.path.join(directory, "shortest_case.txt")

read_list = []
entity_list = []
with open(shortest_cases, "r") as f:
    for line in f:
        sample_id, entity = line.strip().split(",")
        read_list.append(sample_id)
        entity_list.append(entity)

num_list = len(entity_list)

tmp_data = {}
order_list = []
with open(dataset_file, "r") as f:
    #dataset = json.load(f)

    for line in f:
        line = json.loads(line)

        tmp_data[line["id"]] = line
        order_list.append(line["id"])

# swap order list
cnt = 0
for elm in read_list:
    tmp = order_list[cnt]
    idx = order_list.index(elm)
    order_list[cnt] = elm
    order_list[idx] = tmp
    cnt += 1

with open("permute_case.json", "w") as f:
    #for k, v in tmp_data.items():
    cnt = 0
    for idx in order_list:
        v = tmp_data[idx]
        if cnt <= num_list -1:
            v['entities'] = [int(entity_list[cnt])]
            cnt += 1
        json.dump(v, f)
        f.write("\n")

tmp = 0
