### This repository replicates the experiments of the AAAI-24 paper: Beyond Attention: Breaking the Limits of Transformer Context Length with Recurrent Memory

### Note: The dataset used in this paper has been updated! 
[ [paper](https://arxiv.org/abs/2402.10790) ] [ [code](https://github.com/booydar/recurrent-memory-transformer/tree/babilong-release) ] **BABILong** - a long-context benchmark that supports 20 tasks and various sources of background text. 

[ [paper](https://arxiv.org/abs/2207.06881) ] [ [code](https://github.com/booydar/recurrent-memory-transformer/) ] Implementation and various training examples for **Recurrent Memory Transformer**.


## Installation
```bash
pip install -e .
```
This command will install `lm_experiments_tools` with only required packages for Trainer and tools.

`lm_experiments_tools` Trainer supports gradient accumulation, logging to tensorboard, saving the best models based on metrics, custom metrics and data transformations support. For package details refer to the `README-lm-tools.txt` and [ [package source](https://github.com/yurakuratov/t5-experiments) ].

### Install requirements for all experiments
Full requirements for all experiments are specified in requirements.txt. Install requirements after cloning the repo:
```bash
pip install -r requirements.txt
```

### Running experiments
For reproducing experiments please use the `scripts` direcotry. Bash scripts contain hyperparameters and commands to run the experiments. 

Example command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NP=4 ./finetune_babilong_rmt.sh
```

The ArXiv dataset can be downloaded as the part of the [PILE](https://huggingface.co/datasets/EleutherAI/pile). 

## Citation
If you find our work useful, please consider citing the RMT papers:
```
@misc{bulatov2023scaling,
      title={Scaling Transformer to 1M tokens and beyond with RMT}, 
      author={Aydar Bulatov and Yuri Kuratov and Mikhail S. Burtsev},
      year={2023},
      eprint={2304.11062},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
```
@inproceedings{
        bulatov2022recurrent,
        title={Recurrent Memory Transformer},
        author={Aydar Bulatov and Yuri Kuratov and Mikhail Burtsev},
        booktitle={Advances in Neural Information Processing Systems},
        editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
        year={2022},
        url={https://openreview.net/forum?id=Uynr3iPhksa}
}
```