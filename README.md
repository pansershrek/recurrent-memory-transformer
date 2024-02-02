# This repository contains code for reproducing the Recurrent Memory Transformer experiments with the BABILong dataset 


RMT is a memory-augmented segment-level recurrent Transformer. 

![**RMT**](img/RMT_scheme.png?raw=True)

We implement our memory mechanism with no changes to Transformer model by adding special memory tokens to the input sequence. The model is trained to control both memory operations and sequence representations processing.

## Installation
```bash
pip install -e .
```
This command will install `lm_experiments_tools` with only required packages for Trainer and tools.

`lm_experiments_tools` Trainer supports gradient accumulation, logging to tensorboard, saving the best models
based on metrics, custom metrics and data transformations support.

### Install requirements for all experiments
Full requirements for all experiments are specified in requirements.txt. Install requirements after cloning the repo:
```bash
pip install -r requirements.txt
```

### Loading the dataset
Dataset can be found in the `data` directory. To unpack it, run 
```bash
unzip data/tasks_1-20_v1-2.zip
```

Another option is to [download](https://huggingface.co/datasets/facebook/babi_qa) it from Hugging Face.


### Running experiments
For reproducing experiments please use the scripts_exp/babilong directory. Bash scripts contain hyperparameters and commands to run the experiment. 

Example command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NP=4 ./finetune_babilong_qa1_rmt_vary_n_seg.sh
```

Don't forget to pass the path to your dataset folder using the `--babi_path` variable. 