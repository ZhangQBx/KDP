import torch
import Model
import utils
import argparse
import sys
import datetime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./Data/', help='Dataset path')
    parser.add_argument('--dataset_type', type=str, default='cifar10', help='Choose dataset type')
    parser.add_argument('--Trained_teacher_model_path', type=str, help='Teacher model store path')
    parser.add_argument('--teacher_model_type', type=str, default='resnet18', help='teacher model type')
    parser.add_argument('--teacher_batch_size', type=int, default=128, help='teacher training batch size.')
    parser.add_argument('--teacher_num_workers', type=int, default=4, help='teacher training num_workers.')
    parser.add_argument('--teacher_lr', type=float, default=0.1, help='Teacher lr')
    parser.add_argument('--teacher_lr_gamma', type=float, default=0.1, help='teacher training lr_gamma.')
    parser.add_argument('--teacher_momentum', type=float, default=0.5, help='teacher training momentum.')
    parser.add_argument('--teacher_lr_step', type=int, default=[80, 120], nargs='+', help='teacher training lr_step.')
    parser.add_argument('--teacher_train_epochs', type=int, default=100, help='teacher training epoch.')
    parser.add_argument('--teacher_optim_type', type=str, default='sgd', help='teacher optimizer type.')
    parser.add_argument('--teacher_cpu', type=bool, default=False, help='teacher cpu status.')
    parser.add_argument('--teacher_signal', type=str, default='normal', help='Normal or not.')
    parser.add_argument('--teacher_beta', type=float, default=1e-8, help='teacher beta')
    parser.add_argument('--teacher_temperature', type=float, default=4., help='teacher temperature')
    parser.add_argument('--dist_num', type=int, default=16, help='dist calculated per batch')
    parser.add_argument('--proxy_model', type=str, default='trained', help='proxy model')
    parser.add_argument('--log_path', type=str, default='pic', help='.......')


    print('************************')
    print("      PARAMETERS")
    print('************************')
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(time)

    args = parser.parse_args()
    for arg in vars(args):
        print(format(arg, '<30'), format(str(getattr(args, arg)), '<'))
    sys.stdout.flush()

    return args

def main(args):
    if args.dataset_type == 'cifar10':
        trainset, testset = utils.cifar10(args.data_path)
        classes = 10
    elif args.dataset_type == 'cifar100':
        trainset, testset = utils.cifar100(args.data_path)
        classes = 100
    else:
        raise ValueError('Dataset {} dose not supported yet.'.format(args.dataset_type))


    if args.teacher_model_type == 'resnet18':
        Teacher_Model = Model.resnet18(in_dims=3, out_dims=classes)
        print("Untrained teacher model resnet18 prepared...")
    elif args.teacher_model_type == 'resnet50':
        Teacher_Model = Model.resnet50(in_dims=3, out_dims=classes)
        print("Untrained teacher model resnet50 prepared...")
    elif args.teacher_model_type == 'shufflenetv2':
        Teacher_Model = Model.shufflenetv2(ratio=1, class_num=classes)
        print("Untrained teacher model: shufflenetv2 prepared...")
    else:
        raise ValueError('Teacher model {} is not supported yet.'.format(args.teacher_model_type))


    if args.proxy_model == 'trained':
        proxy_model = torch.load('./Model/cifar10_ResNet.pt')
        print('Load trained ResNet proxy model...')
    else:
        proxy_model = Model.resnet18(in_dims=3, out_dims=classes)
        print('Load Untrained ResNet proxy model...')

    utils.train_teacher_model(Teacher_Model, trainset, testset,
                              args.Trained_teacher_model_path, args.log_path, proxy_model,
                              args.teacher_batch_size, args.teacher_num_workers, args.teacher_lr,
                              args.teacher_lr_gamma, args.teacher_momentum, args.teacher_lr_step,
                              args.teacher_train_epochs, args.teacher_optim_type, args.teacher_temperature,
                              args.teacher_beta, args.dist_num, args.teacher_cpu, args.teacher_signal)

    print("--------------------------------------")

    time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(time2)

    print('*************************')
    print("  TEACHER TRAINING DONE ")
    print('*************************')

    return
if __name__ == '__main__':
    args = get_args()
    main(args)