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
    parser.add_argument('--Best_student_model_path', type=str, help='Student model store path.')
    parser.add_argument('--student_model_type', type=str, default='resnet', help='Student model type.')
    parser.add_argument('--student_alpha', type=float, default=1., help='student loss alpha')
    parser.add_argument('--student_temperature', type=float, default=4., help='Student loss temperature.')
    parser.add_argument('--student_batch_size', type=int, default=128, help='student training batch size.')
    parser.add_argument('--student_num_workers', type=int, default=4, help='student training num_workers.')
    parser.add_argument('--student_lr', type=float, default=0.1, help='student training learning rate.')
    parser.add_argument('--student_lr_gamma', type=float, default=0.1, help='student training lr_gamma.')
    parser.add_argument('--student_momentum', type=float, default=0.5, help='student training momentum.')
    parser.add_argument('--student_lr_step', type=int, default=[400], nargs='+', help='student training lr_step.')
    parser.add_argument('--student_train_epochs', type=int, default=100, help='student training epoch.')
    parser.add_argument('--student_optim_type', type=str, default='sgd', help='student optimizer type.')
    parser.add_argument('--student_cpu', type=bool, default=False, help='student cpu status.')
    parser.add_argument('--student_signal', type=str, default='normal', help='Normal or skeptical.')
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


    if args.student_signal == 'normal':
        print('************************')
        print("  NORMAL DISTILLATION")
        print('************************')
        if args.student_model_type == 'resnet':
            Student_Model = Model.resnet18(in_dims=3, out_dims=classes)
            # Student_Model = Model.resnet18_self(num_class=classes)
            print("Untrained student model: resnet18 prepared...")
        elif args.student_model_type == 'cnn':
            Student_Model = Model.CNN(classes=classes, dropout_rate=0.0)
            print("Untrained student model: cnn prepared...")
        elif args.student_model_type == 'shufflenetv2':
            Student_Model = Model.shufflenetv2(ratio=1, class_num=classes)
            print("Untrained student model: shufflenetv2 prepared...")
        elif args.student_model_type == 'mobilenetv2':
            Student_Model = Model.mobilenetv2(class_num=classes)
            print("Untrained student model: mobilenetv2 prepared...")
        else:
            raise ValueError('Student model {} is not supported yet.'.format(args.student_model_type))
    else:
        print('************************')
        print("   SKEPTICAL STUDENT")
        print('************************')
        if args.student_model_type == 'resnet18':
            # Student_Model = Model.resnet18(in_dims=3, out_dims=classes)
            Student_Model = Model.resnet18_self(num_class=classes)
            print("Untrained student model: resnet18_self prepared...")
        elif args.student_model_type == 'resnet50':
            Student_Model = Model.resnet50_self(num_class=classes)
            print("Untrained student model: resnet50_self prepared...")
        elif args.student_model_type == 'mobilenetv2':
            Student_Model = Model.mobileNetV2_self(class_num=classes)
            print("Untrained student model: mobilenetv2_self prepared...")
        else:
            raise ValueError('Student model {} is not supported yet.'.format(args.student_model_type))


    t_model = torch.load(args.Trained_teacher_model_path)
    utils.train_student_model(Student_Model, t_model, trainset, testset, args.Best_student_model_path,
                              args.student_alpha, args.student_temperature, args.log_path, args.student_batch_size,
                              args.student_num_workers, args.student_lr, args.student_lr_gamma, args.student_momentum,
                              args.student_lr_step, args.student_train_epochs,
                              args.student_optim_type, args.student_cpu ,args.student_signal)

    time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(time2)

    print('*************************')
    print("  STUDENT TRAINING DONE ")
    print('*************************')
    return

if __name__ == '__main__':
    args = get_args()
    main(args)