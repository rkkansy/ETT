#!/bin/bash

DATA_DIR="/home/robert/Documents/ETT/data/GLUE"
MODEL_DIR="/home/robert/Documents/ETT/models/BERT-6L-768H-122k/checkpoints/checkpoint-00007680"
OUTPUT_DIR="/home/robert/Documents/ETT/models/eval/BERT-6L-122k-test_no_dp_fp_test"
MODEL_TYPE="bert"

tasks=("MRPC" "CoLA" "RTE" "QQP" "QNLI" "SST-2" "STS-B" "WNLI" "MNLI")

for task in "${tasks[@]}"; do
    echo "Running GLUE task: $task"
    python run_glue.py \
        --data_dir "$DATA_DIR/$task" \
        --model_type "$MODEL_TYPE" \
        --model_name_or_path "$MODEL_DIR" \
        --task_name "$task" \
        --output_dir "$OUTPUT_DIR/$task" \
        --save_steps 0\
        --num_train_epochs 5\
        --warmup_steps 0.04\
        --do_train \
        --do_eval \
        --fp16
        
done