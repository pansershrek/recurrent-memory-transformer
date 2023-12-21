#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.experimental:MemoryCellGenerate
RECURRENT_WRAPPER=modeling_rmt.language_modeling:RecurrentWrapper
BACKBONE_CLS=base_models.modeling_gpt_neox:GPTNeoXForCausalLM
TASK_NAME=qasper
METRIC=f1

ITERS=6000 

TBS=256
INPUT_SIZES=(512 1024 2048)
BSS=(128 64 32 16)

for N in 1 2
do

for (( j=0; j<${#INPUT_SIZES[@]}; j++ ))
do
INPUT_SIZE=${INPUT_SIZES[j]}
INPUT_SEQ_LEN=$INPUT_SIZE
BS=${BSS[j]}

MAX_N_SEGMENTSS=(1)
MEMORY_SIZES=(0)



for MODEL_NAME in EleutherAI/pythia-70m-deduped
do

for (( k=0; k<${#MEMORY_SIZES[@]}; k++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[k]}
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[k]}

for SEGMENT_ORDERING in regular
do

SCHEDULER=linear

for LR in 5e-05 1e-04 3e-04 5e-06 1e-06
do

NP=2
GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
ACCEL_CONFIG=./accel_configs/exp/deepspeed_fp16_o2_np${NP}-${GRAD_ACC_STEPS}.yaml

echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N
echo gradient accumulation steps $GRAD_ACC_STEPS
accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29562 run_finetuning_scrolls_rmt_decoder.py \
        --task_name $TASK_NAME \
        --model_path ../runs/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}-${TGT_LEN}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_iters${ITERS}_${SEGMENT_ORDERING}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --input_seq_len $INPUT_SEQ_LEN \
        --input_size $INPUT_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters $ITERS \
        --use_generate_on_valid \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS/100)) --valid_interval $(($ITERS/10)) \
        --optimize_metric $METRIC --optimize_mode max \
        --show_valid_examples 5 \
        --early_stopping_patience 15 \
        --seed $(($N+42)) \
        --clip_grad_norm 2.0
        
done
done
done
done
done
done
done
echo "done"
