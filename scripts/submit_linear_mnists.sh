#!/bin/bash

python train_mnist.py --experiment_name="mnist_dendfc_resolution_2" \
                      --work_dir="work_dir/mnist_dendfc_resolution_testing" \
                      --resolution=2 \
                      --log_weights \
                      --log_weights_iters=25 \
                      --batch_size=256 \
                      --learning_rate=0.0008 \
                      --num_training_steps=10000 \
                      --warmup_steps_proportion=0.1 \
                      --evaluation_steps=100 \
                      --early_stopping_persistance=10 \
                      --optimizer="adam" \
                      --num_workers=4 \
                      --device="cuda:6"

python train_mnist.py --experiment_name="mnist_dendfc_resolution_3" \
                      --work_dir="work_dir/mnist_dendfc_resolution_testing" \
                      --resolution=3 \
                      --log_weights \
                      --log_weights_iters=25 \
                      --batch_size=256 \
                      --learning_rate=0.0008 \
                      --num_training_steps=10000 \
                      --warmup_steps_proportion=0.1 \
                      --evaluation_steps=100 \
                      --early_stopping_persistance=10 \
                      --optimizer="adam" \
                      --num_workers=4 \
                      --device="cuda:6"

python train_mnist.py --experiment_name="mnist_dendfc_resolution_4" \
                      --work_dir="work_dir/mnist_dendfc_resolution_testing" \
                      --resolution=4 \
                      --log_weights \
                      --log_weights_iters=25 \
                      --batch_size=256 \
                      --learning_rate=0.0008 \
                      --num_training_steps=10000 \
                      --warmup_steps_proportion=0.1 \
                      --evaluation_steps=100 \
                      --early_stopping_persistance=10 \
                      --optimizer="adam" \
                      --num_workers=4 \
                      --device="cuda:6"

python train_mnist.py --experiment_name="mnist_dendfc_resolution_5" \
                      --work_dir="work_dir/mnist_dendfc_resolution_testing" \
                      --resolution=5 \
                      --log_weights \
                      --log_weights_iters=25 \
                      --batch_size=256 \
                      --learning_rate=0.0008 \
                      --num_training_steps=10000 \
                      --warmup_steps_proportion=0.1 \
                      --evaluation_steps=100 \
                      --early_stopping_persistance=10 \
                      --optimizer="adam" \
                      --num_workers=4 \
                      --device="cuda:6"

python train_mnist.py --experiment_name="mnist_dendfc_resolution_6" \
                      --work_dir="work_dir/mnist_dendfc_resolution_testing" \
                      --resolution=6 \
                      --log_weights \
                      --log_weights_iters=25 \
                      --batch_size=256 \
                      --learning_rate=0.0008 \
                      --num_training_steps=10000 \
                      --warmup_steps_proportion=0.1 \
                      --evaluation_steps=100 \
                      --early_stopping_persistance=10 \
                      --optimizer="adam" \
                      --num_workers=4 \
                      --device="cuda:6"

python train_mnist.py --experiment_name="mnist_dendfc_resolution_7" \
                      --work_dir="work_dir/mnist_dendfc_resolution_testing" \
                      --resolution=7 \
                      --log_weights \
                      --log_weights_iters=25 \
                      --batch_size=256 \
                      --learning_rate=0.0008 \
                      --num_training_steps=10000 \
                      --warmup_steps_proportion=0.1 \
                      --evaluation_steps=100 \
                      --early_stopping_persistance=10 \
                      --optimizer="adam" \
                      --num_workers=4 \
                      --device="cuda:6"

python train_mnist.py --experiment_name="mnist_dendfc_resolution_8" \
                      --work_dir="work_dir/mnist_dendfc_resolution_testing" \
                      --resolution=8 \
                      --log_weights \
                      --log_weights_iters=25 \
                      --batch_size=256 \
                      --learning_rate=0.0008 \
                      --num_training_steps=10000 \
                      --warmup_steps_proportion=0.1 \
                      --evaluation_steps=100 \
                      --early_stopping_persistance=10 \
                      --optimizer="adam" \
                      --num_workers=4 \
                      --device="cuda:6"

python train_mnist.py --experiment_name="mnist_dendfc_resolution_9" \
                      --work_dir="work_dir/mnist_dendfc_resolution_testing" \
                      --resolution=9 \
                      --log_weights \
                      --log_weights_iters=25 \
                      --batch_size=256 \
                      --learning_rate=0.0008 \
                      --num_training_steps=10000 \
                      --warmup_steps_proportion=0.1 \
                      --evaluation_steps=100 \
                      --early_stopping_persistance=10 \
                      --optimizer="adam" \
                      --num_workers=4 \
                      --device="cuda:6"


