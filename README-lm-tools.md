This is the readme for lm-experiments-tools.


# Code description:

Training tools, such as Trainer, are in `lm_experiments_tools` package.

## Installation
There are two main parts in the repository:
- `lm_experiments_tools` module
- training scripts (like bert/t5 pretraining) that use `lm_experiments_tools`

### Install only lm_experiments_tools
`lm_experiments_tools` include Trainer with multi-gpu/node with Horovod and APEX torch.cuda.amp FP16 for models
compatible with HF interface. Most of the scripts in the repo use Trainer from `lm_experiments_tools`.

> note: install torch and horovod according to your setup before `lm_experiments_tools` installation.

```bash
pip install -e .
```
This command will install `lm_experiments_tools` with only required packages for Trainer and tools.

`lm_experiments_tools` Trainer supports gradient accumulation, logging to tensorboard, saving the best models
based on metrics, custom metrics and data transformations support.

###  Install Horovod
Depending on your setup just `pip install horovod==0.24.2` might work.

Building Horovod with NCCL for PyTorch:
```bash
HOROVOD_NCCL_HOME=... HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod[pytorch]==0.24.2 --no-binary=horovod
```
check installation with
```bash
horovodrun --check-build
```
For further details check Horovod documentation: https://horovod.readthedocs.io/en/stable/install_include.html

### Install APEX
Install APEX https://github.com/NVIDIA/apex#quick-start
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

apex.amp is moved to torch.cuda.amp https://github.com/NVIDIA/apex/issues/818, but:

speed: `APEX O1` < `torch.cuda.amp` < `APEX O2`

resources (unordered):
 - https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
 - https://pytorch.org/docs/stable/notes/amp_examples.html
 - https://spell.ml/blog/mixed-precision-training-with-pytorch-Xuk7YBEAACAASJam
 - https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
 - https://github.com/horovod/horovod/issues/1089
 - https://github.com/NVIDIA/apex/issues/818

### Install DeepSpeed
DeepSpeed Sparse attention supports only GPUs with compute compatibility >= 7 (V100, T4, A100), CUDA 10.1, 10.2, 11.0, or 11.1 and runs only in FP16 mode (as of DeepSpeed 0.6.0).
```bash
pip install triton==1.0.0
DS_BUILD_SPARSE_ATTN=1 pip install deepspeed==0.6.0 --global-option="build_ext" --global-option="-j8" --no-cache
```
and check installation with
```bash
ds_report
```
#### Triron 1.1.1
Triton 1.1.1 brings x2 speed-up to sparse operations on A100, but DeepSpeed (0.6.5) currently supports only triton 1.0.0.
DeepSpeed fork with triton 1.1.1 support could be used in the cases where such speed-up is needed:
```bash
pip install triton==1.1.1
git clone https://github.com/yurakuratov/DeepSpeed.git
cd DeepSpeed
DS_BUILD_SPARSE_ATTN=1 pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache
```
and run sparse ops tests with
```bash
cd tests/unit
pytest -v test_sparse_attention.py
```