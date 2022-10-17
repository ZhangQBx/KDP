#!/bin/bash
#SBATCH -A test
#SBATCH -J fc10d4tem10
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -p gpu4,p40
#SBATCH -t 17:00:00
#SBATCH -o fc10d4tem10.out
source ~/.bashrc
conda activate torch10
python train_teacher.py --data_path ./Data/  \
                        --dataset_type cifar10 \
                        --Trained_teacher_model_path ./Model/Trained_teacher_fc10d4tem10.pt \
                        --teacher_model_type resnet50 \
                        --teacher_batch_size 128 \
                        --teacher_num_workers 4 \
                        --teacher_lr 0.1 \
                        --teacher_lr_gamma 0.1 \
                        --teacher_momentum 0.9 \
                        --teacher_lr_step 80 120 \
                        --teacher_train_epochs 160 \
                        --teacher_optim_type sgd \
                        --teacher_signal undis \
                        --teacher_beta 1.0 \
                        --teacher_temperature 10.0 \
                        --dist_num 4 \
                        --proxy_model untrained \
                        --log_path exp_fc10d4tem10

python train_student.py --data_path ./Data/  \
                        --dataset_type cifar10 \
                        --Trained_teacher_model_path ./Model/Trained_teacher_fc10d4tem10.pt \
                        --Best_student_model_path ./Model/Best_student_fc10d4tem10.pt \
                        --student_model_type resnet18 \
                        --student_alpha 0.9 \
                        --student_temperature 4.0 \
                        --student_batch_size 128 \
                        --student_num_workers 4 \
                        --student_lr 0.05 \
                        --student_lr_gamma 0.1 \
                        --student_momentum 0.9 \
                        --student_lr_step 120 150 170 \
                        --student_train_epochs 180 \
                        --student_optim_type sgd \
                        --student_signal skeptical \
                        --log_path exp_fc10d4tem10

