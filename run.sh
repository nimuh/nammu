#!/bin/bash

python train.py \
    --data_path      tattabio/OMG             \
    --model_dir      TEST                     \
    --resume                                  \
    --nlayers        18                       \
    --d_model        1024                     \
    --d_inter        768                      \
    --d_state        8                        \
    --bidirectional_strategy add              \
    --max_seq_len    20000                    \
    --total_steps    1                        \
    --lr             1e-3                     \
    --min_lr         1e-4                     \
    --lr_scheduler   cosine_with_min_lr       \
    --mlm_probability 0.30                    \
    --bs             2                        \
    --grad_accum     64                       \
    --save_every     2500




