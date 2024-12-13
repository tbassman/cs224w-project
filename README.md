# cs224w-project
Final course project for Stanford CS224W Fall 2024. "Enhancements to graph neural retrieval for knowledge graph reasoning" by Mu-sheng Lin, Tamika Bassman, and Pravin Ravishanker.

# About

This code base accompanies the final project report published via [Medium](ADD LINK HERE).

For clarity, this code base is separated into three parts, one for each of the three parts of the report. This README documents useful information for navigating and using the code in each of the three parts. 

# Part I

# Part II

## Setup

1. Download the `data.zip` file from this Google Drive [link](https://drive.google.com/drive/folders/1ifgVHQDnvFEunP9hmVYT07Y3rvcpIfQp) (from the original GNN-RAG authors). Place the contents of the `data/CWQ/` file within the data.zip in `part_II/gnn/data/CWQ/`.

2. Also download the `pretrained_lms.zip` file from the same Google Drive link as above. Place the contents of `pretrained_lms` within `part_II/gnn/pretrained_lms/`.

## Running the models

To run the code in this folder (for all of the sensitivity studies performed in Part II), we recommend creating a new Conda environment `conda create --name cs224w-project python=3.12`, navigating inside the `part_II/gnn/` folder, then running `pip install -r requirements.txt`.

### `gnn` folder

This folder contains essentially a fork of the GNN portion of the [original GNN-RAG repository](https://github.com/cmavro/GNN-RAG). We have made minor modifications to the code in this folder (e.g., to the `requirements.txt`, to `load_ckpt` in `train_model.py`, etc.) to streamline running of the code. 

### `gnn_modification` folder

This folder contains all versions of scripts modified by us in order to run experiments 1A, 1B, 2A, and 2B. Please follow the guidance below to run the experiments.

### Experiment 1S (baseline)

#### Training

Navigate to `part_II/gnn/`. Then run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name cwq --experiment_name part_ii_run_1s --data_folder data/CWQ/ --warmup_epoch 80
```

#### Inference

From the Google Drive link [here](https://drive.google.com/drive/folders/1ADnEac18M-AdXFZ8btx1h76bT0FWL3ow?usp=sharing), navigate to the `experiment_1S` folder, download the contents (in particular, the checkpoint file `prn_cwq-rearev-sbert-final.ckpt`), and place it in the local directory in the following location: `part_II/gnn/checkpoint/pretrain/`. Then navigate to `part_II/gnn/` and run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name cwq --load_experiment prn_cwq-rearev-sbert-final.ckpt --data_folder data/CWQ/ --warmup_epoch 80 --is_eval
```

### Experiment 2S 

#### Training

Navigate to `part_II/gnn/`. Then run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3 --num_gnn 5  --name cwq --experiment_name part_ii_run_2s --data_folder data/CWQ/ --warmup_epoch 80
```

#### Inference

From the Google Drive link [here](https://drive.google.com/drive/folders/1ADnEac18M-AdXFZ8btx1h76bT0FWL3ow?usp=sharing), navigate to the `experiment_2S` folder, download the contents (in particular, the checkpoint file `prn_cwq-rearev-sbert-final.ckpt`), and place it in the local directory in the following location: `part_II/gnn/checkpoint/pretrain/`. Then navigate to `part_II/gnn/` and run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3 --num_gnn 5  --name cwq --load_experiment prn_cwq-rearev-sbert-final.ckpt --data_folder data/CWQ/ --warmup_epoch 80 --is_eval
```

### Experiment 3S 

#### Training

Navigate to `part_II/gnn/`. Then run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3 --num_gnn 7  --name cwq --experiment_name part_ii_run_3s --data_folder data/CWQ/ --warmup_epoch 80
```

#### Inference

From the Google Drive link [here](https://drive.google.com/drive/folders/1ADnEac18M-AdXFZ8btx1h76bT0FWL3ow?usp=sharing), navigate to the `experiment_3S` folder, download the contents (in particular, the checkpoint file `prn_cwq-rearev-sbert-final.ckpt`), and place it in the local directory in the following location: `part_II/gnn/checkpoint/pretrain/`. Then navigate to `part_II/gnn/` and run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3 --num_gnn 7  --name cwq --load_experiment prn_cwq-rearev-sbert-final.ckpt --data_folder data/CWQ/ --warmup_epoch 80 --is_eval
```

### Experiment 4S 

#### Training

Navigate to `part_II/gnn/`. Then run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 1 --num_ins 3 --num_gnn 3  --name cwq --experiment_name part_ii_run_4s --data_folder data/CWQ/ --warmup_epoch 80
```

#### Inference

From the Google Drive link [here](https://drive.google.com/drive/folders/1ADnEac18M-AdXFZ8btx1h76bT0FWL3ow?usp=sharing), navigate to the `experiment_4S` folder, download the contents (in particular, the checkpoint file `prn_cwq-rearev-sbert-final.ckpt`), and place it in the local directory in the following location: `part_II/gnn/checkpoint/pretrain/`. Then navigate to `part_II/gnn/` and run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 1 --num_ins 3 --num_gnn 3  --name cwq --load_experiment prn_cwq-rearev-sbert-final.ckpt --data_folder data/CWQ/ --warmup_epoch 80 --is_eval
```

### Experiment 5S 

#### Training

Navigate to `part_II/gnn/`. Then run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 2 --num_ins 2 --num_gnn 3  --name cwq --experiment_name part_ii_run_5s --data_folder data/CWQ/ --warmup_epoch 80
```

#### Inference

From the Google Drive link [here](https://drive.google.com/drive/folders/1ADnEac18M-AdXFZ8btx1h76bT0FWL3ow?usp=sharing), navigate to the `experiment_5S` folder, download the contents (in particular, the checkpoint file `prn_cwq-rearev-sbert-final.ckpt`), and place it in the local directory in the following location: `part_II/gnn/checkpoint/pretrain/`. Then navigate to `part_II/gnn/` and run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 2 --num_ins 2 --num_gnn 3  --name cwq --load_experiment prn_cwq-rearev-sbert-final.ckpt --data_folder data/CWQ/ --warmup_epoch 80 --is_eval
```

### Experiment 6S 

#### Training

Navigate to `part_II/gnn/`. Then run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 2 --num_ins 4 --num_gnn 3  --name cwq --experiment_name part_ii_run_6s --data_folder data/CWQ/ --warmup_epoch 80
```

#### Inference

From the Google Drive link [here](https://drive.google.com/drive/folders/1ADnEac18M-AdXFZ8btx1h76bT0FWL3ow?usp=sharing), navigate to the `experiment_6S` folder, download the contents (in particular, the checkpoint file `prn_cwq-rearev-sbert-final.ckpt`), and place it in the local directory in the following location: `part_II/gnn/checkpoint/pretrain/`. Then navigate to `part_II/gnn/` and run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 2 --num_ins 4 --num_gnn 3  --name cwq --load_experiment prn_cwq-rearev-sbert-final.ckpt --data_folder data/CWQ/ --warmup_epoch 80 --is_eval
```

### Experiment 1A

#### Setup

Navigate to the `part_II/gnn_modification/experiment_1A` folder and copy the script `rearev.py`. In the repo, navigate to `part_II/gnn/models/ReaRev`, and replace the existing version of `rearev.py` with the one you just copied.

#### Training

Navigate to `part_II/gnn/`. Then run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 1 --num_ins 3 --num_gnn 3  --name cwq --experiment_name part_ii_run_1a --data_folder data/CWQ/ --warmup_epoch 80
```

#### Inference

From the Google Drive link [here](https://drive.google.com/drive/folders/1ADnEac18M-AdXFZ8btx1h76bT0FWL3ow?usp=sharing), navigate to the `experiment_1A` folder, download the contents (in particular, the checkpoint file `prn_cwq-rearev-sbert-final.ckpt`), and place it in the local directory in the following location: `part_II/gnn/checkpoint/pretrain/`. Then navigate to `part_II/gnn/` and run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 1 --num_ins 3 --num_gnn 3  --name cwq --load_experiment prn_cwq-rearev-sbert-final.ckpt --data_folder data/CWQ/ --warmup_epoch 80 --is_eval
```

### Experiment 1B

#### Setup

Navigate to the `part_II/gnn_modification/experiment_1B` folder and copy the script `rearev.py`. In the repo, navigate to `part_II/gnn/models/ReaRev`, and replace the existing version of `rearev.py` with the one you just copied.

#### Training

Navigate to `part_II/gnn/`. Then run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name cwq --experiment_name part_ii_run_1b --data_folder data/CWQ/ --warmup_epoch 80
```

#### Inference

From the Google Drive link [here](https://drive.google.com/drive/folders/1ADnEac18M-AdXFZ8btx1h76bT0FWL3ow?usp=sharing), navigate to the `experiment_1B` folder, download the contents (in particular, the checkpoint file `prn_cwq-rearev-sbert-final.ckpt`), and place it in the local directory in the following location: `part_II/gnn/checkpoint/pretrain/`. Then navigate to `part_II/gnn/` and run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name cwq --load_experiment prn_cwq-rearev-sbert-final.ckpt --data_folder data/CWQ/ --warmup_epoch 80 --is_eval
```

### Experiment 2A

#### Setup

Navigate to the `part_II/gnn_modification/experiment_2A` folder and copy the script `reasongnn.py`. In the repo, navigate to `part_II/gnn/modules/kg_reasoning/`, and replace the existing version of `reasongnn.py` with the one you just downloaded from Google Drive.

#### Training

Navigate to `part_II/gnn/`. Then run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name cwq --experiment_name part_ii_run_2a --data_folder data/CWQ/ --warmup_epoch 80
```

#### Inference

From the Google Drive link [here](https://drive.google.com/drive/folders/1ADnEac18M-AdXFZ8btx1h76bT0FWL3ow?usp=sharing), navigate to the `experiment_2A` folder, download the contents (in particular, the checkpoint file `prn_cwq-rearev-sbert-final.ckpt`), and place it in the local directory in the following location: `part_II/gnn/checkpoint/pretrain/`. Then navigate to `part_II/gnn/` and run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name cwq --load_experiment prn_cwq-rearev-sbert-final.ckpt --data_folder data/CWQ/ --warmup_epoch 80 --is_eval
```

### Experiment 2B

#### Setup

FNavigate to the `part_II/gnn_modification/experiment_2B` folder and copy the script `reasongnn.py`. In the repo, navigate to `part_II/gnn/modules/kg_reasoning/`, and replace the existing version of `reasongnn.py` with the one you just downloaded from Google Drive.

#### Training

Navigate to `part_II/gnn/`. Then run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name cwq --experiment_name part_ii_run_2b --data_folder data/CWQ/ --warmup_epoch 80
```

#### Inference

From the Google Drive link [here](https://drive.google.com/drive/folders/1ADnEac18M-AdXFZ8btx1h76bT0FWL3ow?usp=sharing), navigate to the `experiment_2B` folder, download the contents (in particular, the checkpoint file `prn_cwq-rearev-sbert-final.ckpt`), and place it in the local directory in the following location: `part_II/gnn/checkpoint/pretrain/`. Then navigate to `part_II/gnn/` and run the following command line argument:

```
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name cwq --load_experiment prn_cwq-rearev-sbert-final.ckpt --data_folder data/CWQ/ --warmup_epoch 80 --is_eval
```

## Re-running the post-processing analysis

## 
# Part III
