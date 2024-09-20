#!/bin/bash

python train_mnist.py --experiment_name="test_leaky" \
                      --work_dir="work_dir" \
                      --resolution=30 \
                      --log_weights \
                      --log_weights_iters=25 \
                      --batch_size=256 \
                      --learning_rate=0.0008 \
                      --num_training_steps=10000 \
                      --warmup_steps_proportion=0.1 \
                      --evaluation_steps=100 \
                      --early_stopping_persistance=10 \
                      --optimizer="adam" \
                      --num_workers=8 \
                      --device="cuda"