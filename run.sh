#!/bin/bash
#SBATCH --job-name=train-llava
#SBATCH --output=Results/output.txt
#SBATCH --error=Results/error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=y.jin@ufl.edu
#SBATCH --account=vemuri
#SBATCH --qos=vemuri
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=96:00:00

FILE="Results/output.txt"
echo "Date      = $(date)" > $FILE
echo "host      = $(hostname -s)" >> $FILE
echo "Directory = $(pwd)" >> $FILE
echo >> $FILE

T1=$(date +%s)

ml conda
conda activate /blue/amolstad/y.jin/anaconda3/envs/MyNlp

srun deepspeed llava_run.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path my-model/model-01 \
    --train_type use_lora \
    --data_path data \
    --bf16 true \
    --fp16 false \
    --output_dir Results \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 4e-4 \
    --logging_steps 10 \
    --model_max_length 2048 >> $FILE

echo >> $FILE
T2=$(date +%s)
ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED" >> $FILE