import json
import os
from typing import Dict, List
import pandas as pd
import networkx as nx

mode = "gnn"
dataset = "cwq"

F_PREFIX = "/Users/tbassman/Desktop/GitHub/External/"

# folder_name = "pretrain_num_ins_4"
# folder_name = "pretrain_num_ins_2"
folder_name = "checkpoint_num_iter2_num_ins3_num_gnn3"

directory = "/Users/tbassman/Desktop/GitHub/External/GNN-RAG/llm/cs224w"

idx2entity_file = os.path.join(directory, "entities.txt")
entity2text_file = os.path.join(directory, "CWQ_entity2text_COMBINED.json")
idx2relation_file = os.path.join(directory, "relations.txt")
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

    ### 1 - Inputs ###

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

    entity2text_file = os.path.join(
        F_PREFIX, "GNN-RAG/fb_processing/CWQ_entity2text.json"
    )

    ################################

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

        # if sample["em"] < 1:

        q_textbased_success = 0

        ### 5.1.a - what was the correct answer? what were the incorrect ones?

        ## NOTE new calculation -- what is the min/max/average hop length from query entities to true answer entities?
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

        # print(f"This sample has {len(dataset_sample['entities'])} q entities.")

        # find shortest path from each question entity to each true answer entity
        # we'll compute stats on the list of shortest paths that we find
        for q_ent_idx in dataset_sample["entities"]:

            q_ent_id = idx2ent[q_ent_idx]
            q_ent_text = ent2text.get(q_ent_id, q_ent_id)

            # print(
            #     f"Question present in subgraph? {subgraph_eid.has_node(q_ent_id)}, {subgraph_text.has_node(q_ent_text)}"
            # )

            # print(f"This sample has {len(true_a_ent_ids)} true answer entities.")
            for a_ent_id in true_a_ent_ids:

                a_ent_text = ent2text.get(a_ent_id, a_ent_id)
                # print(
                # f"    Answer present in subgraph? {subgraph_eid.has_node(a_ent_id)}, {subgraph_text.has_node(a_ent_text)}"
                # )
                # if a_ent_id[0] == ":":
                #     print(
                #         f"{q_id}, {a_ent_text}, {ent2text.get(a_ent_id[1:],a_ent_id[1:])}"
                #     )
                try:
                    shortest_paths = [
                        p
                        for p in nx.all_shortest_paths(subgraph_eid, q_ent_id, a_ent_id)
                    ]
                    shortest_path = len(shortest_paths[0]) - 1
                    # print(f"        {shortest_path}")
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
                        # print(f"        {shortest_path}")
                        # print(
                        #     f"{q_id}: success through text-based graph for answer entity {a_ent_id}, {a_ent_text}"
                        # )
                        q_to_true_a_hops.append(shortest_path)
                        q_textbased_success = 1
                    except:
                        continue

        if len(q_to_true_a_hops) == 0:
            q_to_true_a_hops = [99]

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
