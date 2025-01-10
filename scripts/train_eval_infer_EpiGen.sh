#!/bin/bash

### --------------------------------------------------------------
### Train GPT-2-small architecture
### You need a trained tokenizer and config under regaler/EpiGen
### --------------------------------------------------------------
accelerate launch epigen/run_clm_no_trainer.py \
    --model_name_or_path gpt2-small \
    --train_file data/train.csv \
    --validation_file data/val.csv \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 512 \
    --num_train_epochs 100 \
    --tokenizer_name regaler/EpiGen \
    --checkpointing_steps epoch \
    --with_tracking \
    --report_to wandb \
    --gradient_accumulation_steps 1 \
    --gpt2_small \
    --output_dir 241205_example_run
    # --resume_from_checkpoint 241205_example_run/epoch_99 \
    # --learning_rate 5e-5
    # --use_fast_tokenizer True


### --------------------------------------------------------------
### Evaluate all saved checkpoints
### After, use utils.draw_learning_curve() function to select the best ckpt
### --------------------------------------------------------------
# for i in $(seq 0 1 100); do
#   CUDA_VISIBLE_DEVICES=0 python epigen/run_clm_predict.py \
#       --model_name_or_path gpt2-small \
#       --train_file data/train.csv \
#       --validation_file data/val.csv \
#       --per_device_train_batch_size 512 \
#       --per_device_eval_batch_size 512 \
#       --num_train_epochs 101 \
#       --tokenizer_name regaler/EpiGen \
#       --checkpointing_steps epoch \
#       --with_tracking \
#       --report_to wandb \
#       --gradient_accumulation_steps 1 \
#       --gpt2_small \
#       --output_dir 241205_example_run \
#       --inf_out_dir 241205_example_run \
#       --resume_from_checkpoint 241205_example_run/epoch_${i} \
#       # --tokenized_dataset 241202_example_run/lm_datasets.pkl
#       # --inference_mode
# done


### --------------------------------------------------------------
### Inference (Prediction based on TCR sequences only)
### Still, the input file should contain cols: text,label where `label` can be dummy placeholders
### This outputs a csv file where additional columns `pred_{i}` were created.
### --------------------------------------------------------------
datasets=("Glanville" "MIRA")
datasets=("donor1")
datasets=("test_seen_both.csv" "test_unseen_both.csv" "test_unseen_pep.csv" "test_unseen_tcr.csv")
datasets=("PIRD" "IEDB" "VDJdb" "McPAS")
# for dataset in "${datasets[@]}"; do
#     CUDA_VISIBLE_DEVICES=0 python epigen/run_clm_predict.py \
#         --model_name_or_path gpt2-small \
#         --train_file data/train.csv \
#         --validation_file data/val.csv \
#         --per_device_train_batch_size 32 \
#         --per_device_eval_batch_size 32 \
#         --num_train_epochs 101 \
#         --tokenizer_name regaler/EpiGen \
#         --checkpointing_steps epoch \
#         --with_tracking \
#         --report_to wandb \
#         --gradient_accumulation_steps 1 \
#         --output_dir 241205_example_run \
#         --inf_out_dir 241205_example_run/${dataset}" \
#         --resume_from_checkpoint 241205_example_run/epoch_28 \
#         --inference_mode \
#         --gpt2_small
# done
