#!/bin/bash

#SBATCH -J uni
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_outputs/%j.out

#SBATCH -p PH100q
#SBATCH -w node06
#SBATCH -n 1

module load cuda12.2/blas/12.2.2 || true
module load cuda12.2/fft/12.2.2 || true
module load cuda12.2/nsight/12.2.2 || true
module load cuda12.2/profiler/12.2.2 || true
module load cuda12.2/toolkit/12.2.2 || true

export CUDA_VISIBLE_DEVICES=6
export dataset_name="mtc_e2e"
export proj_name="$dataset_name"
export output_dir="output/$dataset_name"
export WANDB_MODE="disabled"

seed=42
fold=0

output_dir_seed="$output_dir/seed_$seed"
input_dir="input/mtc/seed_${seed}-fold_${fold}-mv-e2e"
output_dir_seed_fold="$output_dir_seed/fold_$fold"
mkdir -p $output_dir_seed_fold

# ----------------------------------------
# Run the code
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export HF_EVALUATE_OFFLINE=1
export WANDB_PROJECT=$proj_name
export TOKENIZERS_PARALLELISM=true

python train.py \
    --model_name_or_path t5-base \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file $input_dir/train.json \
    --validation_file $input_dir/validation.json \
    --test_file $input_dir/test.json \
    --task_type AM \
    --view_type mv \
    --voting_threshold 2 \
    --source_seq_column "source" \
    --target_seq_column "target" \
    --source_prefix "" \
    --output_dir $output_dir_seed_fold \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --overwrite_output_dir \
    --predict_with_generate \
    --num_train_epochs 50 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --logging_strategy "epoch" \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --metric_for_best_model "eval_overall_micro_f1" \
    --greater_is_better True \
    --learning_rate "1e-4" \
    --extra_tokens '[" [SEP]", "[AC]", "[/AC]", "[NEW_LINE]", "[ASA]", "[ACI]", "[ARI]"]' \
    --seed $seed \
    --eval_delay 41 \
    --warmup_ratio 0.1 \
    --weight_decay 0 \
    2>&1 | tee ${output_dir_seed_fold}/stdout.log
