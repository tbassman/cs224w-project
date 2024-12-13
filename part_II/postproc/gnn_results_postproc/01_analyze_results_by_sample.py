import json
import os
from typing import Dict, List
import pandas as pd
import networkx as nx

# NOTES ON RUNNING
# 1. 'entities.txt', 'test.json', 'relations.txt' can be obtained from 'part_II/gnn/data/CWQ'
# 2. 'entities_names.json' can be obtained from the original authors' data download at https://drive.google.com/drive/folders/1ifgVHQDnvFEunP9hmVYT07Y3rvcpIfQp
# 3. folder_name is of your choice and should be where relevant files are stored
# 4. F_PREFIX and other file pathways herein should be adjusted for your local file directory
# 5. The '_test.info' file is produced by any GNN run of the ReaRev model per Part II. You can obtain it from our Google Drive link https://drive.google.com/drive/folders/1ADnEac18M-AdXFZ8btx1h76bT0FWL3ow?usp=sharing by renaming the *.info file in any experiment folder to '_test.info'
# 6. O

mode = "gnn"
dataset = "cwq"

F_PREFIX = "./"

folder_name = "experiment_1S"

idx2entity_file = os.path.join(folder_name, "entities.txt")
entity2text_file = os.path.join(folder_name, "entities_names.json")
idx2relation_file = os.path.join(folder_name, "relations.txt")
with open(idx2entity_file, "r") as f:
    idx2ent = f.read().split("\n")
with open(entity2text_file, "r") as f:
    ent2text = json.load(f)
with open(idx2relation_file, "r") as f:
    idx2rel = f.read().split("\n")


def build_graph_eid(
    tuples: List[List[int]], idx2ent: List[str], ent2text: Dict[str, str]
):
    G = nx.Graph()
    for tuple in tuples:
        h, r, t = tuple
        h = idx2ent[h]
        r = idx2rel[r]
        t = idx2ent[t]
        G.add_edge(h, t, relation=r.strip())
    return G


def build_graph_text(
    tuples: List[List[int]], idx2ent: List[str], ent2text: Dict[str, str]
):
    G = nx.Graph()
    for tuple in tuples:
        h, r, t = tuple
        h = ent2text.get(idx2ent[h], idx2ent[h])
        r = idx2rel[r]
        t = ent2text.get(idx2ent[t], idx2ent[t])
        G.add_edge(h, t, relation=r.strip())
    return G


def id2text(id: str, entity_map: dict):
    return entity_map.get(id, "*tbd*")


def answer_id2text(id: str, dataset: dict):

    answers_list = dataset[id]["answers"]

    return [answer["text"] for answer in answers_list]


if mode == "gnn" and dataset == "cwq":

    ### 1 - Additional Inputs ###

    ####### f'{folder_name}' #######

    pred_inputfile = os.path.join(
        F_PREFIX,
        f"GNN-RAG/gnn/checkpoint/pretrain/test_results/{folder_name}/_test.info",
    )
    pred_outputfile = os.path.join(
        F_PREFIX,
        f"GNN-RAG/gnn/checkpoint/pretrain/test_results/{folder_name}/_test.json",
    )
    incorrect_pred_outputcsvfile = os.path.join(
        F_PREFIX,
        f"GNN-RAG/gnn/checkpoint/pretrain/test_results/{folder_name}/_test_allpred_with_extra_info.csv",
    )

    dataset_inputfile = os.path.join(F_PREFIX, "GNN-RAG/gnn/data/CWQ/test.json")
    dataset_outputfile = os.path.join(
        F_PREFIX, "GNN-RAG/gnn/data/CWQ/test_reformat.json"
    )

    ################################

    # NOTE change these flags to false if you have already run the script once with them as True
    update_dataset_file = False
    # update_pred_file = True
    update_pred_file = False

    ### 2 - Process (test) dataset file to be readable as json ###

    if update_dataset_file:

        dataset = {}

        with open(dataset_inputfile, "r") as f_in:
            for line in f_in:
                sample = json.loads(line)
                dataset[sample["id"]] = {
                    key: sample[key]
                    for key in ["answers", "question", "entities", "subgraph"]
                }

        with open(dataset_outputfile, "w") as f_out:
            json.dump(dataset, f_out)
        print(f"Reformatted test dataset json written to file - {dataset_outputfile}")

    else:

        with open(dataset_outputfile, "r") as f:
            dataset = json.load(f)

    ### 3 - Process prediction file to be readable as json ###

    if update_pred_file:

        preds = {}

        with open(pred_inputfile, "r") as f_in:
            with open(dataset_inputfile, "r") as f_in_dataset:
                for _, (line, line_dataset) in enumerate(zip(f_in, f_in_dataset)):
                    sample = json.loads(line)
                    dataset_sample = json.loads(line_dataset)
                    preds[dataset_sample["id"]] = {
                        key: sample[key]
                        for key in [
                            "question",
                            # "0",
                            # "1", # seems to be produced for each iteration 1 to T
                            "answers",
                            "precison",
                            "recall",
                            "f1",
                            "hit",
                            "em",
                            "cand",
                        ]
                    }

        with open(pred_outputfile, "w") as f_out:
            json.dump(preds, f_out)
        print(f"Reformatted test prediction file written - {pred_outputfile}")

    else:

        with open(pred_outputfile, "r") as f:
            preds = json.load(f)

    ### 4 - Load mapping entity id-to-name/human-readable text ###

    # load the map from file
    with open(entity2text_file, "r") as f:
        entity2text = json.load(f)

    ### 5 - Process predictions from given test run of a model ###

    ### 5.1 - deep dive on incorrect predictions ###

    incorrect_pred_data = pd.DataFrame(
        columns=[
            "id",
            "q_text",
            "q_ent_id",
            "a_id",
            "a_text",
            "pred_id",
            "pred_text",
            "0",
            "1",
            "precision",
            "recall",
            "f1",
            "hit",
            "em",
            "shortest_path_min",
            "shortest_path_mean",
            "shortest_path_max",
            "subgraph_size_nodes",
            "subgraph_size_edges",
            "q_textbased_success",
        ],
    ).set_index("id", inplace=True)

    for q_id, sample in preds.items():

        q_textbased_success = 0

        ## what is the min/max/average hop length from query entities to true answer entities?
        dataset_sample = dataset[q_id]

        # build the subgraph
        subgraph_eid = build_graph_eid(
            dataset_sample["subgraph"]["tuples"],
            idx2ent=idx2ent,
            ent2text=ent2text,
        )

        # build the subgraph
        subgraph_text = build_graph_text(
            dataset_sample["subgraph"]["tuples"],
            idx2ent=idx2ent,
            ent2text=ent2text,
        )

        q_to_true_a_hops = []
        true_a_ent_ids = [ans["kb_id"] for ans in dataset_sample["answers"]]
        true_a_ent_text = [ent2text.get(eid, eid) for eid in true_a_ent_ids]

        # find shortest path from each question entity to each true answer entity
        # we'll compute stats on the list of shortest paths that we find
        for q_ent_idx in dataset_sample["entities"]:

            q_ent_id = idx2ent[q_ent_idx]
            q_ent_text = ent2text.get(q_ent_id, q_ent_id)

            for a_ent_id in true_a_ent_ids:

                a_ent_text = ent2text.get(a_ent_id, a_ent_id)
                try:
                    shortest_paths = [
                        p
                        for p in nx.all_shortest_paths(subgraph_eid, q_ent_id, a_ent_id)
                    ]
                    shortest_path = len(shortest_paths[0]) - 1
                    q_to_true_a_hops.append(shortest_path)

                except:
                    try:
                        shortest_paths = [
                            p
                            for p in nx.all_shortest_paths(
                                subgraph_text, q_ent_text, a_ent_text
                            )
                        ]
                        shortest_path = len(shortest_paths[0]) - 1
                        q_to_true_a_hops.append(shortest_path)
                        q_textbased_success = 1
                    except:
                        continue

        if len(q_to_true_a_hops) == 0:
            q_to_true_a_hops = [99]

        # collect all sample-level data into a df we'll print to csv
        sample_data = {
            "q_text": str(sample["question"]),
            "q_ent_id": str(
                [idx2ent[q_ent_idx] for q_ent_idx in dataset_sample["entities"]]
            ),
            "a_id": str(sample["answers"]),
            "a_text": str([answer_id2text(q_id, dataset)]),
            "pred_id": str([cand[0] for cand in sample["cand"]]),
            "pred_text": str(
                [id2text(cand[0], entity2text) for cand in sample["cand"]]
            ),
            "precision": str(sample["precison"]),
            "recall": str(sample["recall"]),
            "f1": str(sample["f1"]),
            "hit": str(sample["hit"]),
            "em": str(sample["em"]),
            "shortest_path_min": min(q_to_true_a_hops),
            "shortest_path_mean": sum(q_to_true_a_hops) / len(q_to_true_a_hops),
            "shortest_path_max": max(q_to_true_a_hops),
            "subgraph_size_nodes": subgraph_eid.number_of_nodes(),
            "subgraph_size_edges": subgraph_eid.number_of_edges(),
        }

        incorrect_pred_data = pd.concat(
            [
                incorrect_pred_data,
                pd.DataFrame(
                    sample_data, columns=sample_data.keys(), index=[str(q_id)]
                ),
            ]
        )

    incorrect_pred_data.to_csv(incorrect_pred_outputcsvfile)
    print(f"data written to {incorrect_pred_outputcsvfile}")
