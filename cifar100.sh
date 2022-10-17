#!/bin/bash
#SBATCH -A test
#SBATCH -J resnet50_1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -p gpu4,p40
#SBATCH -t 1-22:00:00
#SBATCH -o resnet50_1.out
source ~/.bashrc
conda activate torch10
python train_teacher.py --data_path ../Data/  \
                        --dataset_type cifar100 \
                        --Trained_teacher_model_path ./Model/Trained_teacher_resnet50_1.pt \
                        --teacher_model_type resnet50 \
                        --teacher_batch_size 128 \
                        --teacher_num_workers 4 \
                        --teacher_lr 0.1 \
                        --teacher_lr_gamma 0.2 \
                        --teacher_momentum 0.9 \
                        --teacher_lr_step 60 120 160 \
                        --teacher_train_epochs 200 \
                        --teacher_optim_type sgd \
                        --teacher_signal undis \
                        --teacher_beta 0.9 \
                        --teacher_temperature 8.0 \
                        --dist_num 4 \
                        --proxy_model untrained \
                        --log_path exp_resnet50_1

python train_student.py --data_path ../Data/  \
                        --dataset_type cifar100 \
                        --Trained_teacher_model_path ./Model/Trained_teacher_resnet50_1.pt \
                        --Best_student_model_path ./Model/Best_student_resnet18_1.pt \
                        --student_model_type resnet18 \
                        --student_alpha 0.9 \
                        --student_temperature 20.0 \
                        --student_batch_size 128 \
                        --student_num_workers 4 \
                        --student_lr 0.1 \
                        --student_lr_gamma 0.2 \
                        --student_momentum 0.9 \
                        --student_lr_step 60 120 160 \
                        --student_train_epochs 200 \
                        --student_optim_type sgd \
                        --student_signal normal \
                        --log_path exp_resnet18_1

python train_student.py --data_path ../Data/  \
                        --dataset_type cifar100 \
                        --Trained_teacher_model_path ./Model/Trained_teacher_resnet50_1.pt \
                        --Best_student_model_path ./Model/Best_student_shuffle_1.pt \
                        --student_model_type shufflenetv2 \
                        --student_alpha 0.9 \
                        --student_temperature 20.0 \
                        --student_batch_size 128 \
                        --student_num_workers 4 \
                        --student_lr 0.1 \
                        --student_lr_gamma 0.2 \
                        --student_momentum 0.9 \
                        --student_lr_step 60 120 160 \
                        --student_train_epochs 200 \
                        --student_optim_type sgd \
                        --student_signal normal \
                        --log_path exp_shuffle_1

python train_student.py --data_path ../Data/  \
                        --dataset_type cifar100 \
                        --Trained_teacher_model_path ./Model/Trained_teacher_resnet50_1.pt \
                        --Best_student_model_path ./Model/Best_student_mobile_1.pt \
                        --student_model_type mobilenetv2 \
                        --student_alpha 0.9 \
                        --student_temperature 20.0 \
                        --student_batch_size 128 \
                        --student_num_workers 4 \
                        --student_lr 0.1 \
                        --student_lr_gamma 0.2 \
                        --student_momentum 0.9 \
                        --student_lr_step 60 120 160 \
                        --student_train_epochs 200 \
                        --student_optim_type sgd \
                        --student_signal normal \
                        --log_path exp_mobile_1