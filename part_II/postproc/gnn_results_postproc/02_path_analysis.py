# For reference, each sample in the CWQ dataset starts out looking like this
# {
#   "id": "WebQTrn-9",
#   "question": "how old is sacha baron cohen",
#   "entities": [99399],
#   "answers": [{ "kb_id": "1971-10-13", "text": null }],
#   "subgraph": {
#     "tuples": [
#       [1084387, 54, 1128807],
#       [545190, 4, 2520],
#       [145210, 133, 1128808]
#     ],
#     "entities": [2520, 1070971, 1129126, 543949, 12661]
#   }
# }

# In this script we want to:
# -- build the subgraph
# -- identify the query entities
# -- identify the true answers and the candidate entities
# -- figure out how far are the shortest paths from the candidate entities to the true answers
# -- these results will inform our reflection studies later on in Part III

# NOTE: the relevant files referenced/imported in this script are mentioned/created in 01_analyze_results_by_sample.py

import json
import os
from typing import Dict, List
import networkx as nx
import pandas as pd

# NOTE change for your local directory
directory = "GNN-RAG/llm/cs224w"

dataset_file = os.path.join(directory, "test_reformat.json")
prediction_file = os.path.join(directory, "_test.json")

idx2entity_file = os.path.join(directory, "entities.txt")
entity2text_file = os.path.join(directory, "entities_names.json")
idx2relation_file = os.path.join(directory, "relations.txt")


def build_graph(tuples: List[List[int]], idx2ent: List[str], ent2text: Dict[str, str]):
    G = nx.Graph()
    for tuple in tuples:
        h, r, t = tuple
        h = ent2text.get(idx2ent[h], idx2ent[h])
        r = idx2rel[r]
        t = ent2text.get(idx2ent[t], idx2ent[t])
        G.add_edge(h, t, relation=r.strip())
    return G


with open(dataset_file, "r") as f:
    dataset = json.load(f)

with open(prediction_file, "r") as f:
    preds = json.load(f)

with open(idx2entity_file, "r") as f:
    idx2ent = f.read().split("\n")

with open(entity2text_file, "r") as f:
    ent2text = json.load(f)

with open(idx2relation_file, "r") as f:
    idx2rel = f.read().split("\n")

# for cases in which the correct answer(s) is/are not returned by the GNN
# case a: the shortest path from the incorrect prediction to a true answer
case_a = pd.DataFrame(columns=["sample id", "candidate", "paths", "shortest_path"])
counter = 0

for sample_id, sample in preds.items():

    data_sample = dataset[sample_id]
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
            # for each true answer, find the
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

case_a.to_csv(os.path.join(directory, "path_analysis.csv"))
