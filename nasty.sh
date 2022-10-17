#!/bin/bash
#SBATCH -A test
#SBATCH -J 09tem8
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -p gpu4,short,short4
#SBATCH -t 17:00:00
#SBATCH -o 09tem8.out
source ~/.bashrc
conda activate torch10
python train_teacher.py --data_path ../Knockoff/Data/  \
                        --dataset_type cifar100 \
                        --Trained_teacher_model_path ./Model/Nasty_teacher.pt \
                        --teacher_model_type resnet \
                        --teacher_batch_size 128 \
                        --teacher_num_workers 4 \
                        --teacher_lr 0.1 \
                        --teacher_lr_gamma 0.2 \
                        --teacher_momentum 0.9 \
                        --teacher_lr_step 60 120 160 \
                        --teacher_train_epochs 200 \
                        --teacher_optim_type sgd \
                        --teacher_signal normal \
                        --teacher_beta 0.9 \
                        --teacher_temperature 9.0 \
                        --dist_num 4 \
                        --proxy_model trained 

python train_student.py --data_path ../Knockoff/Data/  \
                        --dataset_type cifar100 \
                        --Trained_teacher_model_path ./Model/Nasty_teacher8.pt \
                        --Best_student_model_path ./Model/KD_student.pt \
                        --student_model_type mobilenetv2 \
                        --student_alpha 0.9 \
                        --student_temperature 20.0 \
                        --student_batch_size 128 \
                        --student_num_workers 4 \
                        --student_lr 0.05 \
                        --student_lr_gamma 0.1 \
                        --student_momentum 0.9 \
                        --student_lr_step 120 150 170 \
                        --student_train_epochs 180 \
                        --student_optim_type sgd \
                        --student_signal skeptical \
                        --log_path exp_09tem8
