#!/bin/bash

DATA_DIR="/home/robert/Documents/ETT/data/GLUE"
MODEL_DIR="/home/robert/Documents/ETT/models/BERT-6L-768-122k/checkpoints/checkpoint-00006656"
OUTPUT_DIR="/home/robert/Documents/ETT/models/eval/BERT-6L-122k"
MODEL_TYPE="bert"

tasks=("MRPC" "SST-2" "STS-B" "QQP" "QNLI" "RTE" "WNLI" "MNLI" "CoLA")

for task in "${tasks[@]}"; do
    echo "Running GLUE task: $task"
    python run_glue.py \
        --data_dir "$DATA_DIR/$task" \
        --model_type "$MODEL_TYPE" \
        --model_name_or_path "$MODEL_DIR" \
        --task_name "$task" \
        --output_dir "$OUTPUT_DIR/$task" \
        --save_steps 500 \
        --do_train \
        --do_eval
done