#!/bin/bash

python train_leaky_mnist.py --experiment_name="test_run" \
                            --work_dir="work_dir" \
                            --resolution=50 \
                            --log_weights \
                            --log_weights_iters=25 \
                            --batch_size=128 \
                            --learning_rate=0.0008 \
                            --num_training_steps=300 \
                            --warmup_steps_proportion=0.1 \
                            --evaluation_steps=100 \
                            --early_stopping_persistance=10 \
                            --optimizer="adam" \
                            --num_workers=8 \
                            --device="cuda:7"