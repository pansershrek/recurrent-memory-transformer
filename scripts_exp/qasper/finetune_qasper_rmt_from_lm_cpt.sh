#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
# set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.experimental:MemoryCellGenerate
RECURRENT_WRAPPER=modeling_rmt.language_modeling:RecurrentWrapper
BACKBONE_CLS=base_models.modeling_gpt_neox:GPTNeoXForCausalLM
TASK_NAME=qasper
METRIC=f1

ITERS=5000 

TBS=32
INPUT_SIZES=(1024)
BSS=(32 16 )

for N in 1
do

for (( j=0; j<${#INPUT_SIZES[@]}; j++ ))
do
INPUT_SIZE=${INPUT_SIZES[j]}
INPUT_SEQ_LEN=$INPUT_SIZE
BS=${BSS[j]}

MAX_N_SEGMENTSS=(2 3)
MEMORY_SIZES=(2 2)


for MODEL_NAME in EleutherAI/pythia-70m-deduped
do

for (( k=0; k<${#MEMORY_SIZES[@]}; k++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[k]}
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[k]}

for SEGMENT_ORDERING in regular
do

SCHEDULER=linear

for LR in 1e-04 5e-05 1e-05 5e-06
do

NP=1
GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
ACCEL_CONFIG=./accel_configs/exp/accelerate/deepspeed_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
cd accel_configs/
python create_config.py \
        --train_batch_size $TBS\
        --train_micro_batch_size_per_gpu $BS\
        --gradient_accumulation_steps $GRAD_ACC_STEPS\
        --np $NP\
        --gradient_clipping 1.0
cd ..

echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N
echo gradient accumulation steps $GRAD_ACC_STEPS
accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29566 run_finetuning_scrolls_rmt_decoder_v3.py \
        --task_name $TASK_NAME \
        --model_path ../runs/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_iters${ITERS}_${SEGMENT_ORDERING}_v3_from_lm_cpt_5-${MAX_N_SEGMENTS}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --model_cpt ../runs/pile/EleutherAI/pythia-70m-deduped/lr1e-05_linear_adamw_wd1e-03_2044-8176-5x2048_mem2_bs256_regular_bptt-5_from_cpt_cv4_sp0.2_4-5_continue/run_1 \
        --input_seq_len $INPUT_SEQ_LEN \
        --input_size $INPUT_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters $ITERS \
        --use_generate_on_valid \
        --optimizer AdamW  --weight_decay 0.01 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/5)) \
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
