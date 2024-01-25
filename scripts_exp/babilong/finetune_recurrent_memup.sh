#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./finetune_babilong_baseline.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.rec_memup_classification:MemUPModule #modeling_rmt.language_modeling:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.rec_memup_classification:RecurrentMemUP
BACKBONE_CLS=transformers:AutoModelForCausalLM
TASK_DATASET=qa1_single-supporting-fact
NOISE_DATASET=pg19
METRIC=exact_match

MODEL_NAME=gpt2  # backbone model
SEGMENT_SIZE=128 # size of one segment in tokens
SAMPLE_SIZE=512 # length of task sample in tokens
MEMORY_SIZE=16
MAX_N_SEGMENTS=4

ITERS=50
TBS=16
BS=4

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
SCHEDULER=linear
LR=1e-04

for PRED_FREQ in 1 2 4
do

K2=-1   # BPTT unroll length

ACCEL_CONFIG=./accel_configs/accelerate.yaml

echo RUNNING: TASK_DATASET $TASK_DATASET MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE 
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME LR $LR  PRED_FREQ $PRED_FREQ
echo gradient accumulation steps $GRAD_ACC_STEPS

accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29003 run_babilong_memup.py \
      --task_dataset $TASK_DATASET \
      --noise_dataset $NOISE_DATASET \
      --babi_path data/tasks_1-20_v1-2/en-10k \
      --model_path ../runs/debug/${TASK_DATASET}/$MODEL_NAME/${MAX_N_SEGMENTS}_${SEGMENT_SIZE}_align_right_k2_${K2}_pf_${PRED_FREQ}/run_$N \
      --from_pretrained $MODEL_NAME \
      --model_type $MODEL_TYPE \
      --memory_cell_cls $MEMORY_CELL \
      --recurrent_wrapper_cls $RECURRENT_WRAPPER \
      --model_cls $BACKBONE_CLS \
      --segment_size $SEGMENT_SIZE \
      --sample_size $SAMPLE_SIZE \
      --num_mem_tokens $MEMORY_SIZE \
      --max_n_segments $MAX_N_SEGMENTS\
      --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
      --num_training_steps $((ITERS*2)) \
      --iters $ITERS \
      --save_best \
      --k2 $K2 \
      --optimizer AdamW  --weight_decay 0.01 \
      --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
      --data_n_workers 2 \
      --log_interval $(($ITERS/100)) --valid_interval $(($ITERS/20)) \
      --optimize_metric $METRIC --optimize_mode max \
      --show_valid_examples 5 \
      --early_stopping_patience 15 \
      --seed $(($N+42)) \
      --clip_grad_norm 1.0 \
      --segment_alignment right \
      --prediction_frequency $PRED_FREQ
      #--use_generate_on_valid \
done
done
done
done
done
done
done
done
echo "done"
