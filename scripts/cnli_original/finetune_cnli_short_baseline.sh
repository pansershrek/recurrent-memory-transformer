#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
BACKBONE_CLS=transformers:AutoModelForCausalLM
TASK_NAME=cnli_original
METRIC=exact_match

ITERS=6000
TBS=32

MAX_N_SEGMENTS=1
INPUT_SIZES=(128 256 512)
BSS=(8 4 2)
for N in 1
do

for MODEL_NAME in gpt2
do

for (( j=0; j<${#INPUT_SIZES[@]}; j++ ))
do
BS=${BSS[j]}

INPUT_SIZE=${INPUT_SIZES[j]}
INPUT_SEQ_LEN=$INPUT_SIZE
TGT_LEN=$INPUT_SIZE

for SEGMENT_ORDERING in regular
do

SCHEDULER=linear

for LR in 5e-05
do

echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N
horovodrun --gloo -np $NP python run_finetuning_cnli_rmt_refactor.py \
        --task_name $TASK_NAME \
        --model_path ../runs/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}-${TGT_LEN}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_iters${ITERS}_${SEGMENT_ORDERING}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $BACKBONE_CLS \
        --backbone_cls $BACKBONE_CLS \
        --input_seq_len $INPUT_SEQ_LEN \
        --input_size $INPUT_SIZE \
        --target_seq_len $TGT_LEN \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --use_generate_on_valid \
        --iters $ITERS \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --save_best \
        --optimize_metric $METRIC --optimize_mode max \
        --log_interval $(($ITERS/100)) --valid_interval $(($ITERS/10)) \
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