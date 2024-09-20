#!/bin/bash

python train_simulated_regression.py --experiment_name="test_run_reg" \
                                     --dataset="simple_sinusoid" \
                                     --work_dir="work_dir" \
                                     --resolution=1 \
                                     --log_weights \
                                     --log_weights_iters=25 \
                                     --batch_size=512 \
                                     --learning_rate=0.0008 \
                                     --num_training_steps=10000 \
                                     --warmup_steps_proportion=0.1 \
                                     --evaluation_steps=100 \
                                     --early_stopping_persistance=10 \
                                     --optimizer="adam" \
                                     --num_workers=8 \
                                     --device="cuda"