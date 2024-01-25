#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./finetune_babilong_baseline.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.language_modeling_mem_retrieve:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.language_modeling:RecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM
TASK_DATASET=qa1_single-supporting-fact
NOISE_DATASET=pg19
METRIC=exact_match

MODEL_NAME=gpt2  # backbone model

SEGMENT_SIZE=512 # size of one segment in tokens
SAMPLE_SIZE=1024 # length of task sample in tokens
MAX_N_SEGMENTS=4

SAMPLE_SIZES=(512 1024 2048 4096 8192 16384 32768) #(65536 128000) #
MAX_N_SEGMENTSS=(1 2 4 8 16 32 64) #(128 250) #

TBS=1
BS=1

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
SCHEDULER=linear
LR=1e-04
WD=1e-03
K2=-1   # BPTT unroll length

ACCEL_CONFIG=./accelerate.yaml
EXP_PATH=./runs/babilong/qa1_single-supporting-fact/pg19/gpt2/lr5e-05_linear_adamw_wd1e-03_seqlen2048_4x512_mem16_r32_w16_bs64__bptt--1_sp0.5_retr_attention/run_1

for (( i=0; i<${#SAMPLE_SIZES[@]}; i++ ))
do
SAMPLE_SIZE=${SAMPLE_SIZES[i]}
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[i]}

echo RUNNING: TASK_DATASET $TASK_DATASET SEGMENT_SIZE $SEGMENT_SIZE 
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME LR $LR N $N
echo gradient accumulation steps $GRAD_ACC_STEPS

accelerate launch --num_processes $NP --main_process_port 29512 --config_file $ACCEL_CONFIG run_finetuning_babilong_rmt.py \
        --task_dataset $TASK_DATASET \
        --noise_dataset $NOISE_DATASET \
        --babi_path /home/bulatov/datasets/babi/tasks_1-20_v1-2/en-10k \
        --experiment_cfg ${EXP_PATH}/config.json \
        --init_checkpoint ${EXP_PATH}/model_best/pytorch_model.bin \
        --segment_size $SEGMENT_SIZE \
        --sample_size $SAMPLE_SIZE \
        --max_n_segments $MAX_N_SEGMENTS \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --validate_only
        
done
echo "done"