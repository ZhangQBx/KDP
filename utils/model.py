import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils
import Model
import sys


# import numpy as np
# import matplotlib.pyplot as plt

def train_teacher_step(model, proxy_model, train_loader, optimizer, beta, temperature, dist_num, cpu=False,
                       signal='normal'):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    train_loss_batch = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if not cpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        max_norm = -1
        optimizer.zero_grad()
        outputs = model(inputs)
        if signal == 'normal':
            loss = nn.CrossEntropyLoss()(outputs, labels)
            # Only for testing
            # loss = utils.nasty_teacher_loss(proxy_model, inputs, outputs, labels, temperature, cpu)
        else:
            if beta == 0:
                loss = nn.CrossEntropyLoss()(outputs, labels)
            else:
                loss = utils.loss_teacher(proxy_model, inputs, outputs, labels, temperature, beta, dist_num, cpu)

        loss.backward()

        #gradient normalize
        for pp in model.parameters():
            max_norm = max(max_norm, pp.grad.data.abs().max().item())
        for pp in model.parameters():
            pp.grad.data *= max_norm / pp.grad.data.abs().max().item()


        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        label_target = labels

        correct += predicted.eq(label_target).sum().item()
        train_loss_batch = train_loss / total
        if batch_idx % 20 == 0:
            print("---*", batch_idx, "*---")
            sys.stdout.flush()

    acc = 100. * correct / total
    print("Final train accuracy: ", acc)
    print('Final train loss: ', train_loss_batch)
    sys.stdout.flush()

    return train_loss_batch, acc


def train_student_step(model, t_model, train_loader, optimizer, alpha, temperature, cpu=False):
    model.train()
    t_model.eval()
    train_loss = 0
    correct = 0
    total = 0
    train_loss_batch = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if not cpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        with torch.no_grad():
            teacher_outputs = t_model(inputs)
        loss = utils.loss_kd(outputs, labels, teacher_outputs, alpha, temperature)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        label_target = labels

        correct += predicted.eq(label_target).sum().item()
        train_loss_batch = train_loss / total
        if batch_idx % 20 == 0:
            print("---*", batch_idx, "*---")
            sys.stdout.flush()

    acc = 100. * correct / total
    print("Final train accuracy: ", acc)
    print('Final train loss: ', train_loss_batch)
    sys.stdout.flush()

    return train_loss_batch, acc


def train_skeptical_student_step(model, t_model, train_loader, optimizer, alpha, temperature, cpu=False):
    model.train()
    t_model.eval()
    train_loss = 0
    correct = 0
    total = 0
    train_loss_batch = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if not cpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs, _ = model(inputs)
        with torch.no_grad():
            teacher_outputs = t_model(inputs)
        loss = utils.skeptical_student_loss(outputs, labels, teacher_outputs, alpha, temperature)
        #Only for testing...
        #loss = utils.loss_kd_df(outputs, teacher_outputs, temperature)
        #Testing done...
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs[0].max(1)
        total += labels.size(0)
        label_target = labels

        correct += predicted.eq(label_target).sum().item()
        train_loss_batch = train_loss / total
        if batch_idx % 20 == 0:
            print("---*", batch_idx, "*---")
            sys.stdout.flush()

    acc = 100. * correct / total
    print("Final train accuracy: ", acc)
    print('Final train loss: ', train_loss_batch)
    sys.stdout.flush()

    return train_loss_batch, acc


def test_step(model, test_loader, criterion, cpu=False):
    model.eval()
    test_loss = 0.
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            if not cpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if batch_idx % 10 == 0:
                print("---&", batch_idx, "&---")
                sys.stdout.flush()

    acc = 100. * correct / total
    test_loss /= total
    print("Final test accuracy: ", acc)
    print("Final test loss: ", test_loss)
    sys.stdout.flush()
    return test_loss, acc


def test_skeptical_step(model, test_loader, criterion, cpu=False):
    model.eval()
    test_loss = 0.
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            if not cpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs, _ = model(inputs)
            outputs = outputs[0]
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if batch_idx % 10 == 0:
                print("---&", batch_idx, "&---")
                sys.stdout.flush()

    acc = 100. * correct / total
    test_loss /= total
    print("Final test accuracy: ", acc)
    print("Final test loss: ", test_loss)
    sys.stdout.flush()
    return test_loss, acc


def train_teacher_model(model, trainset, testset, output_path, doc, proxy_model, batch_size=128, num_workers=0, lr=0.1,
                        lr_gamma=0.1, momentum=0.5,
                        lr_step=[80, 120], epochs=100, optim_type='sgd', temperature=1, beta=1e-8, dist_num=16,
                        cpu=False, signal='normal'):
    best_train_acc = -1.
    best_test_acc = -1.
    log = dict()
    doc_name = os.path.join(doc, 'Teacher')
    B = 0

    if signal == 'normal':
        print('************************')
        print("    NORMAL TRAINING")
        print('************************')
    else:
        print('************************')
        print("     UNDISTILLABLE")
        print('************************')

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    criterion_test = nn.CrossEntropyLoss()

    if not cpu:
        model.cuda()

    if optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
        print('Optimizer: sgd')
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print('Optimizer: adam')


    test_acc_list = []

    for i in range(epochs):
        print("Training epoch: ", i + 1)

        lr = adjust_learning_rate(i, lr, lr_gamma, lr_step, optimizer)

        if signal != 'normal':
            if i == lr_step[-2]:
                B = beta
                print("Beta change from {} to {}  ".format(0, B))

        train_loss, train_acc = train_teacher_step(model=model, proxy_model=proxy_model, train_loader=train_loader,
                                                   optimizer=optimizer, beta=B, temperature=temperature,
                                                   dist_num=dist_num, cpu=cpu, signal=signal)
        # scheduler.step()
        utils.add_log(log, 'train_loss', train_loss)
        utils.add_log(log, 'train_acc', train_acc)
        # train_loss_list.append(train_loss)
        # train_acc_list.append(train_acc)
        best_train_acc = max(best_train_acc, train_acc)

        test_loss, test_acc = test_step(model=model, test_loader=test_loader,
                                        criterion=criterion_test, cpu=cpu)
        utils.add_log(log, 'test_loss', test_loss)
        utils.add_log(log, 'test_acc', test_acc)
        # test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        best_test_acc = max(best_test_acc, test_acc)
        if signal == 'normal':
            if test_acc >= best_test_acc:
                print("Upgrade model...")
                torch.save(model, output_path)
                print("Saving done...")
            # Only for testing
            # if test_acc >= 81:
            #     break

    utils.save_log(doc_name, log)

    # utils.plot_pic(test_loss_list, epochs, 'Test_loss_trend', 'Epochs', 'Loss', doc_name)
    # utils.plot_pic(test_acc_list, epochs, 'Test_acc_trend', 'Epochs', 'Acc', doc_name)
    # utils.plot_pic(train_loss_list, epochs, 'Train_loss_trend', 'Epochs', 'Loss', doc_name)
    # utils.plot_pic(train_acc_list, epochs, 'Train_acc_trend', 'Epochs', 'Acc', doc_name)

    print("Trend log saved...")

    if signal == 'normal':
        print("Best teacher model saved...")
        print("Best teacher test accuracy: ", best_test_acc)

    else:
        print("Upgrade model...")
        torch.save(model, output_path)
        print("Saving done...")

        print("Teacher model saved...")
        print("Final teacher test accuracy: ", test_acc_list[-1])


def train_student_model(model, t_model, trainset, testset, output_path, alpha, temperature, doc, batch_size=128,
                        num_workers=0, lr=0.1, lr_gamma=0.1,
        momentum=0.9, lr_step=[40], epochs=100, optim_type='sgd', cpu=False, signal='normal'):
    best_train_acc = -1.
    best_test_acc = -1.
    log = dict()
    doc_name = os.path.join(doc, 'Student')

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    criterion_test = nn.CrossEntropyLoss()
    if not cpu:
        model.cuda()

    if optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
        print('Optimizer: sgd')
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print('Optimizer: adam')

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    for i in range(epochs):
        print("Training epoch: ", i + 1)

        lr = adjust_learning_rate(i, lr, lr_gamma, lr_step, optimizer)

        if signal == 'normal':
            train_loss, train_acc = train_student_step(model, t_model, train_loader, optimizer, alpha, temperature,
                                                       cpu=cpu)
            test_loss, test_acc = test_step(model, test_loader, criterion_test, cpu=cpu)

        else:
            train_loss, train_acc = train_skeptical_student_step(model, t_model, train_loader, optimizer, alpha,
                                                                 temperature, cpu=cpu)
            test_loss, test_acc = test_skeptical_step(model, test_loader, criterion_test, cpu=cpu)

        # scheduler.step()
        utils.add_log(log, 'train_loss', train_loss)
        utils.add_log(log, 'train_acc', train_acc)
        # train_loss_list.append(train_loss)
        # train_acc_list.append(train_acc)
        best_train_acc = max(best_train_acc, train_acc)

        # test_loss, test_acc = test_step(model, test_loader, criterion_test, cpu=cpu)
        utils.add_log(log, 'test_loss', test_loss)
        utils.add_log(log, 'test_acc', test_acc)
        # test_loss_list.append(test_loss)
        # test_acc_list.append(test_acc)
        best_test_acc = max(best_test_acc, test_acc)

        if test_acc >= best_test_acc:
            print("Upgrade model...")
            torch.save(model, output_path)
            print("Saving done...")

    utils.save_log(doc_name, log)
    # utils.plot_pic(test_loss_list, epochs, 'Test_loss_trend', 'Epochs', 'Loss', doc_name)
    # utils.plot_pic(test_acc_list, epochs, 'Test_acc_trend', 'Epochs', 'Acc', doc_name)
    # utils.plot_pic(train_loss_list, epochs, 'Train_loss_trend', 'Epochs', 'Loss', doc_name)
    # utils.plot_pic(train_acc_list, epochs, 'Train_acc_trend', 'Epochs', 'Acc', doc_name)

    print("Trend log saved...")
    print("Best student model saved...")
    print("Best student test accuracy: ", best_test_acc)


def adjust_learning_rate(epoch, lr, lr_gamma, lr_step, optimizer):
    if epoch in lr_step:
        lr = lr * lr_gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print("Learning rate decay...")
        print("Learning rate now: ", lr)

    return lr

