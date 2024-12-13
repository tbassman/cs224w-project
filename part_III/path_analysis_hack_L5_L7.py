import json
import os
from typing import Dict, List
import networkx as nx
import pandas as pd

# set working directory
directory = "working directory"

dataset_file = os.path.join(directory, "test_reformat.json")
prediction_file = os.path.join(directory, "_test.json")
prediction_file1 = os.path.join(directory, "_test_L7.info")

idx2entity_file = os.path.join(directory, "entities.txt")
entity2text_file = os.path.join(directory, "CWQ_entity2text_COMBINED.json")
idx2relation_file = os.path.join(directory, "relations.txt")
entity_name_file = os.path.join(directory, "entities_names.json")


def build_graph(tuples: List[List[int]], idx2ent: List[str], ent2text: Dict[str, str]):
    G = nx.Graph()
    for tuple in tuples:
        h, r, t = tuple
        h = idx2ent[h] 
        r = idx2rel[r]
        t = idx2ent[t] 
        G.add_edge(h, t, relation=r.strip())
    return G


with open(dataset_file, "r") as f:
    dataset = json.load(f)

with open(prediction_file, "r") as f:
    preds = json.load(f)

with open(prediction_file1, "r") as f:
    preds0 = f.readlines()

with open(idx2entity_file, "r") as f:
    idx2ent = f.read().split("\n")

ent2idx = {}
for i, elm in enumerate(idx2ent):
    ent2idx[elm] = i

with open(entity2text_file, "r") as f:
    ent2text = json.load(f)

with open(idx2relation_file, "r") as f:
    idx2rel = f.read().split("\n")

tmp_data = {}
with open(entity_name_file, "r") as f:
    entity_name = json.load(f)

case_a = pd.DataFrame(columns=["sample id", "candidate", "paths", "shortest_path"])
counter = 0

cnt = 0
cases = {}
cases_entity = {}
for sample_id, sample in preds.items():

    data_sample = dataset[sample_id]
    sample = json.loads(preds0[cnt].strip())
    cnt += 1
    true_answers = [ans["kb_id"] for ans in data_sample["answers"]]

    if sample["em"] < 1:

        # build the subgraph
        subgraph = build_graph(
            data_sample["subgraph"]["tuples"],
            idx2ent=idx2ent,
            ent2text=ent2text,
        )

        # for each candidate answer
        for candidate in sample["cand"]:
            cand = candidate[0]
            paths = {}
            shortest_path = 99
            
            for ans in true_answers:
                try:
                    shortest_paths = nx.all_shortest_paths(subgraph, cand, ans)
                    paths[ans] = [len(p) - 1 for p in shortest_paths][0]
                    shortest_path = min(shortest_path, paths[ans])
                except:
                    paths[ans] = 99
                    continue

            case_a = pd.concat(
                [
                    case_a,
                    pd.DataFrame(
                        {
                            "sample id": sample_id,
                            "candidate": str([cand, ent2text.get(cand, cand)]),
                            "paths": str(paths),
                            "shortest": shortest_path,
                        },
                        index=[counter],
                    ),
                ]
            )
            counter += 1

            if candidate[0] in entity_name:
                if shortest_path <= 5:
                    try:
                        if shortest_path < cases[sample_id]:
                            cases[sample_id] = shortest_path
                            cases_entity[sample_id] = ent2idx[candidate[0]]
                    except:
                        cases[sample_id] = shortest_path
                        cases_entity[sample_id] = ent2idx[candidate[0]]

cases_sorted = sorted(cases.items(), key=lambda x: x[1])

# output the targeting test cases
with open("shortest_case.txt", "w") as file:
    for elm in cases_sorted:
        file.write(f"{elm[0]},{cases_entity[elm[0]]}\n")
