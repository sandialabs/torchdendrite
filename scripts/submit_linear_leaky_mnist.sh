#!/bin/bash

python train_leaky_mnist.py --experiment_name="mnist_dendfc_leaky_resolution_1" \
                            --work_dir="work_dir" \
                             --resolution=1 \
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

python train_leaky_mnist.py --experiment_name="mnist_dendfc_leaky_resolution_10" \
                            --work_dir="work_dir" \
                             --resolution=10 \
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

python train_leaky_mnist.py --experiment_name="mnist_dendfc_leaky_resolution_20" \
                            --work_dir="work_dir" \
                             --resolution=20 \
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

python train_leaky_mnist.py --experiment_name="mnist_dendfc_leaky_resolution_30" \
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

python train_leaky_mnist.py --experiment_name="mnist_dendfc_leaky_resolution_40" \
                            --work_dir="work_dir" \
                             --resolution=40 \
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

python train_leaky_mnist.py --experiment_name="mnist_dendfc_leaky_resolution_50" \
                            --work_dir="work_dir" \
                             --resolution=50 \
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

python train_leaky_mnist.py --experiment_name="mnist_dendfc_leaky_resolution_60" \
                            --work_dir="work_dir" \
                             --resolution=60 \
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

python train_leaky_mnist.py --experiment_name="mnist_dendfc_leaky_resolution_70" \
                            --work_dir="work_dir" \
                             --resolution=70 \
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

python train_leaky_mnist.py --experiment_name="mnist_dendfc_leaky_resolution_80" \
                            --work_dir="work_dir" \
                             --resolution=80 \
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

python train_leaky_mnist.py --experiment_name="mnist_dendfc_leaky_resolution_90" \
                            --work_dir="work_dir" \
                             --resolution=90 \
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

python train_leaky_mnist.py --experiment_name="mnist_dendfc_leaky_resolution_100" \
                            --work_dir="work_dir" \
                             --resolution=100 \
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