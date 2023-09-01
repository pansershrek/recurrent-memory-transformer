#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.language_modeling:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.language_modeling:RecurrentWrapper
BACKBONE_CLS=base_models.modeling_gpt_neox:GPTNeoXForCausalLM
TASK_NAME=pile

ITERS=70000
TBS=256

INPUT_SIZE=2048

MAX_N_SEGMENTSS=(1)
BSS=(32)

for MEMORY_SIZE in 0
do 

for N in 1
do

for MODEL_NAME in EleutherAI/pythia-70m-deduped
do

for (( j=0; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]} 
BLOCK_SIZE=$((INPUT_SIZE-2*MEMORY_SIZE))
HISTORY_SIZE=$(((MAX_N_SEGMENTS - 1) * BLOCK_SIZE))
BS=${BSS[j]}
K2=${MAX_N_SEGMENTS}

LR=1e-03

for SEGMENT_ORDERING in regular
do

for SCHEDULER in constant_with_warmup
do


echo RUNNING: TASK_NAME MEMORY_SIZE INPUT_SIZE BLOCK_SIZE HISTORY_SIZE N_SEG  MODEL_NAME MODEL_CLS LR N
echo RUNNING: $TASK_NAME $MEMORY_SIZE $INPUT_SIZE $BLOCK_SIZE $HISTORY_SIZE $MAX_N_SEGMENTS $MODEL_NAME $MODEL_CLS  $LR $N
accelerate launch --config_file ./accel_configs/fp16_o1_np2.yaml run_finetuning_pile_rmt.py \
        --task_name $TASK_NAME \
        --model_path ../runs/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}-${TGT_LEN}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${K2}_continue/run_$N \
        --model_cpt ../runs/pile/EleutherAI/pythia-70m-deduped/lr1e-03_linear_adamw_wd1e-03_--1x2048_mem0_bs256_regular_bptt-1/run_1 \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --block_size $BLOCK_SIZE \
        --history_size $HISTORY_SIZE \
        --input_size $INPUT_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --save_best \
        --iters $ITERS \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.01 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS/100)) --valid_interval $(($ITERS/20)) \
        --show_valid_examples 5 \
        --early_stopping_patience 15 \
        --seed $(($N+42)) \
        --clip_grad_value 1.0
        
        --fp16 \
        
done
done
done
done
done
done
done
echo "done"
