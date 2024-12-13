from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import List


def find_ent_name(ent_id: str):
    max_num_fb_file = 194
    folder_fb = "./"

    found_name = 0

    for i in range(max_num_fb_file):
        f_fb = open(os.path.join(folder_fb, f"freebase-rdf-2015-08-09-00-0{i}.txt"))
        for line in f_fb:
            if ent_id + ">" in line:
                fb_line_name = line.split("\t")[2].split('"')[1]
                found_name = 1
                break
        f_fb.close()
        if found_name == 1:
            return ent_id + " " + fb_line_name
        if i == max_num_fb_file - 1 and found_name == 0:
            return ent_id + " " + "???"


def map_function():

    fp_entities = "../gnn/data/CWQ/entities2.txt"
    fp_out = "./CWQ_entities_with_names_parallel.txt"

    # collect list of already-processed entities
    already_processed = []
    with open(fp_out, "r") as f:
        for line in f:
            already_processed += [line.split(" ")[0]]

    with open(fp_entities, "r") as f_ent:
        entities_list = f_ent.readlines()
    entities_list = [x.strip("\n") for x in entities_list]
    final_entities_list = []
    for entity in entities_list:
        if entity not in already_processed:
            final_entities_list += [entity]
    print(len(entities_list), len(final_entities_list))

    with ProcessPoolExecutor(max_workers=14) as exe:
        futures = [exe.submit(find_ent_name, ent_id) for ent_id in final_entities_list]

        for future in as_completed(futures):
            ent_id_plus_name = future.result()
            with open(fp_out, "a") as f_out:
                f_out.write(ent_id_plus_name + "\n")


if __name__ == "__main__":
    map_function()
