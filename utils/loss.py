import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


def loss_kd(outputs, labels, teacher_outputs, alpha, temperature):
    T = temperature
    soft_loss = (T * T * alpha) * nn.KLDivLoss(reduction='batchmean') \
        (F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T,
                                                      dim=1))
    hard_loss = (1. - alpha) * nn.CrossEntropyLoss()(outputs, labels)
    Loss = soft_loss + hard_loss

    return Loss

#Only for testing...
def loss_kd_df(outputs, teacher_outputs, temperature):
    T = temperature
    loss_1 = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[2] / T, dim=1),
                                                 F.softmax(teacher_outputs / T, dim=1))
    loss_2 = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[1] / T, dim=1),
                                                 F.softmax(outputs[0] / T, dim=1))
    df_loss = loss_1 + loss_2
    return df_loss
#Testing done..

#Only for testing
def skeptical_student_loss(outputs, labels, teacher_outputs, alpha, temperature):
    T = temperature
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[2] / T, dim=1),
                                                  F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
              nn.CrossEntropyLoss()(outputs[2], labels) * (1. - alpha) + \
              (0.3 * nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[1] / T, dim=1),
                                                         F.softmax(outputs[0] / T, dim=1)) + 0.7 * nn.CrossEntropyLoss()(outputs[1], labels)) + \
              nn.CrossEntropyLoss()(outputs[0], labels)

    return KD_loss
#Testing done...



def loss_teacher(proxy_model, inputs, outputs, labels, temperature,
                 beta, dist_num, cpu):
    # print(temperature)
    loss_1 = nn.CrossEntropyLoss()(outputs, labels)
    loss_2 = 0
    length = inputs.shape[0]
    if (length // dist_num) == 0:
        return loss_1
    else:
        Compute_idx = random.sample(range(0, length), length // dist_num)

        for i in Compute_idx:
            dist = utils.Compute_Dist(proxy_model, inputs[i],
                                      outputs[i], labels[i], temperature, cpu)
            loss_2 += dist

        loss = loss_1 - beta * (loss_2 / len(Compute_idx))
        return loss

# Only for testing
def nasty_teacher_loss(proxy_model, inputs, outputs, labels, temperature, cpu):
    if not cpu:
        proxy_model.cuda()
    proxy_model.eval()
    T = temperature
    tch_loss = nn.CrossEntropyLoss()(outputs, labels)
    with torch.no_grad():
        output_stu = proxy_model(inputs)  # logit without SoftMax
    output_stu = output_stu.detach()
    adv_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output_stu / T, dim=1),
                                                   F.softmax(outputs / T, dim=1)) * (T * T)  # wish to max this item

    loss = tch_loss - 0.04 * adv_loss + 100.0
    return loss
#Testing done...

