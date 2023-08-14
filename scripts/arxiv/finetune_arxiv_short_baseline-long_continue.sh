#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
BACKBONE_CLS=transformers:AutoModelForCausalLM
TASK_NAME=arxiv

ITERS=525000
TBS=32
BS=8

TGT_LEN=256
INPUT_SEQ_LEN=256
INPUT_SIZE=256

MAX_N_SEGMENTSS=(1)
MEMORY_SIZES=(NA)

for N in 10
do

for MODEL_NAME in gpt2
do

for (( j=0; j<${#MEMORY_SIZES[@]}; j++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[j]}
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]} 

for SEGMENT_ORDERING in regular
do

SCHEDULER=linear

for LR in 1.6692e-05
do

echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N
horovodrun --gloo -np $NP python run_finetuning_arxiv_rmt.py \
        --task_name $TASK_NAME \
        --model_path ../runs/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}-${TGT_LEN}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_iters${ITERS}_${SEGMENT_ORDERING}_continue/run_$N \
        --backbone_cpt /home/jovyan/rmt/runs/arxiv/gpt2/lr5e-05_linear_adamw_wd1e-03_256-256-1x256_memNA_bs32_iters1750000_regular/run_10 \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $BACKBONE_CLS \
        --backbone_cls $BACKBONE_CLS \
        --input_seq_len $INPUT_SEQ_LEN \
        --input_size $INPUT_SIZE \
        --target_seq_len $TGT_LEN \
        --bptt_depth -1 \
        --save_best \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters $ITERS \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 0 \
        --data_n_workers 2 \
        --log_interval $(($ITERS/100)) --valid_interval $(($ITERS/50)) \
        --show_valid_examples 5 \
        --early_stopping_patience 15 \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
        
done
done
done
done
done
echo "done"
