# Activate conda env
# ...

logger = log
# Specify the learning rates, batch sizes, and task names you want to use
learning_rates=("4e-5")
batch_sizes=(16)
task_names=("mrpc" "cola" "rte" "stsb")
long_task_names=("qqp" "qnli" "mnli" "sst2")
# Path to the Python script
python_script=run_glue_n.py
model_path=""
model_name=""
# Common parameters
common_params="--model_name_or_path ${model_path} --logging_steps 100 --max_seq_length 128 --overwrite_output_dir --do_train --do_eval --evaluation_strategy steps --gradient_accumulation_steps 1 --weight_decay 0.01 --max_grad_norm 1.0 --num_train_epochs 5 --lr_scheduler_type cosine --warmup_ratio 0.05"

# Iterate over learning rates, batch sizes, and task names
for lr in "${learning_rates[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for task_name in "${task_names[@]}"; do
            # Create output directory based on task_name, learning_rate, and batch_size
            output_dir="eval/${model_name}${task_name}_lr${lr}_batch${batch_size}"

            # Execute the Python script with the current parameters
            python $python_script $common_params --eval_steps 50 --task_name $task_name --learning_rate $lr --per_device_train_batch_size $batch_size --per_device_eval_batch_size $batch_size --output_dir $output_dir 
        done
        for task_name in "${long_task_names[@]}"; do
            # Create output directory based on task_name, learning_rate, and batch_size
            output_dir="eval/${model_name}${task_name}_lr${lr}_batch${batch_size}"

            # Execute the Python script with the current parameters
            python $python_script $common_params --eval_steps 1000 --task_name $task_name --learning_rate $lr --per_device_train_batch_size $batch_size --per_device_eval_batch_size $batch_size --output_dir $output_dir 
        done
    done
done

end=$(date +"%T")
echo "Completed: $end"