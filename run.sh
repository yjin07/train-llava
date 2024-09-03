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

nvidia-smi >> $FILE
free -h >> $FILE
echo >> $FILE

T1=$(date +%s)

ml conda
conda activate /blue/amolstad/y.jin/anaconda3/envs/MyNlp

srun --mpi=pmi2 --exclusive deepspeed llava_run.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path /blue/amolstad/y.jin/train-llava/my-model/model-01 \
    --train_type use_lora \
    --data_path /blue/amolstad/y.jin/train-llava/data \
    --resume_from_checkpoint /blue/amolstad/y.jin/train-llava/Results/checkpoint-1000 \
    --bf16 true \
    --fp16 false \
    --output_dir /blue/amolstad/y.jin/train-llava/Results \
    --num_train_epochs 5 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --report_to "tensorboard" \
    --learning_rate 4e-4 \
    --logging_steps 50 >> $FILE

echo >> $FILE
T2=$(date +%s)
ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED" >> $FILE
