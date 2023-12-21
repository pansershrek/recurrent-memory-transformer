#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.experimental:MemoryCellGenerate
RECURRENT_WRAPPER=modeling_rmt.experimental:RecurrentWrapperGenerate
BACKBONE_CLS=base_models.modeling_gpt_neox:GPTNeoXForCausalLM
TASK_NAME=associative_retrieval_v2
METRIC=exact_match

ITERS=100000
TBS=512

INPUT_SIZE=2048


for MEMORY_SIZE in 2 4 8
do 
MAX_N_SEGMENTS=2
BS=512

for N in 1
do

MODEL_CFG=/home/jovyan/rmt/wip/base_models/gptconfigs/neox_tiny.json

KEY_SIZE=4
NUM_PAIRS=1

for VALUE_SIZE in 16 32 64 128
do

BLOCK_SIZE=$((KEY_SIZE + VALUE_SIZE + 2))

for LR in 1e-03
do

K2=${MAX_N_SEGMENTS}

for SEGMENT_ORDERING in regular
do

for SCHEDULER in linear
do


echo RUNNING: TASK_NAME MEMORY_SIZE KEY_SIZE VALUE_SIZE N_SEG  MODEL_NAME MODEL_CLS LR N
echo RUNNING: $TASK_NAME $MEMORY_SIZE $KEY_SIZE $VALUE_SIZE $MAX_N_SEGMENTS $MODEL_NAME $MODEL_CLS  $LR $N
accelerate launch --num_processes $NP --config_file ./accel_configs/accelerate.yaml --main_process_port 29501 run_finetuning_associative_retrieval.py \
        --task_name $TASK_NAME \
        --model_path ../runs/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_k${KEY_SIZE}-v${VALUE_SIZE}-p${NUM_PAIRS}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${K2}/run_$N \
        --model_cfg $MODEL_CFG \
        --model_cls $BACKBONE_CLS \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --segment_size $BLOCK_SIZE \
        --key_size $KEY_SIZE \
        --value_size $VALUE_SIZE \
        --num_pairs $NUM_PAIRS \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --use_generate_on_valid \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters $ITERS \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS/100)) --valid_interval $(($ITERS/20)) \
        --optimize_metric $METRIC --optimize_mode max \
        --show_valid_examples 5 \
        --early_stopping_patience 15 \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
        
done
done
done
done
done
done
done
echo "done"
