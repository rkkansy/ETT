logger = log
python_script=run_lm_distributed.py
partitions=("easy" "hard" "ambiguous" "low_entropy" "high_entropy")
model_path="/project/data/models/BERT-small-10k"

for partition in "${partitions[@]}"; do
    model_name="${model_path}-${partition}"
    echo $model_name
    common_params="--config configs/bert_wiki.txt --instance_data_path ${model_path} --data_partition ${partition} --mask_set 2 --partition_frac 1.0 --ckpt_steps 100 --logging_steps 50 --do_train --do_eval"

    python3 $python_script $common_params --output_dir ${model_name} --config_name configs/bert-6L-512H.json --max_steps 10240 --gradient_accumulation_steps 16 --train_batch_size 96 --eval_batch_size 96
done

end=$(date +"%T")
echo "Completed: $end"