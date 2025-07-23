from __future__ import print_function, division

import sys
import time
import torch
from distiller_zoo import DistillKL
from .util import AverageMeter, accuracy_k



def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    end = time.time()

    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target = data
            index=idx
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            # index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        logit_s = model_s(input)
        feat_s= model_s(input, is_feat=True)
        with torch.no_grad():
            logit_t = model_t(input)
            feat_t = model_t(input, is_feat=True)
            # feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        # loss_div = criterion_div(logit_s, logit_t)
        if opt.distill == 'MSKD':
            loss_div = criterion_div(logit_s, logit_t, current_epoch=epoch)
        else:
            loss_div = criterion_div(logit_s, logit_t)
        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            # f_s = module_list[1](feat_s[opt.hint_layer])
            # f_t = feat_t[opt.hint_layer]
            # loss_kd = criterion_kd(feat_s, feat_t)
            loss_kd = criterion_kd(feat_s[-1], feat_t[-1].detach())
        elif opt.distill == 'MSKD':
            loss_kd = criterion_kd(feat_s[-1], feat_t[-1].detach())
            # temperature_adjustment_factor = 0.1
            # if loss_cls > 3:  # 3.5举例，根据具体情况调整阈值,,5.2
            #     opt.kd_T *= (1 + temperature_adjustment_factor)
            #     print(f"Loss_cls: {loss_cls.item()}, New kd_T: {opt.kd_T}")  # 增大温度参数
            # elif 0.04 < loss_cls < 0.08:
            #     opt.kd_T *= (1 - 0.09)
            #     print(f"Loss_cls: {loss_cls.item()}, New kd_T: {opt.kd_T}")  # 减小温度参数
            # criterion_div = DistillKL(opt.kd_T)
        # elif opt.distill == 'crd':
        #     f_s = feat_s[-1]
        #     f_t = feat_t[-1]
        #     loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        # elif opt.distill == 'attention':
        #     g_s = feat_s[1:-1]
        #     g_t = feat_t[1:-1]
        #     loss_group = criterion_kd(g_s, g_t)
        #     loss_kd = sum(loss_group)
        # elif opt.distill == 'nst':
        #     g_s = feat_s[1:-1]
        #     g_t = feat_t[1:-1]
        #     loss_group = criterion_kd(g_s, g_t)
        #     loss_kd = sum(loss_group)
        # elif opt.distill == 'similarity':
        #     g_s = [feat_s[-2]]
        #     g_t = [feat_t[-2]]
        #     loss_group = criterion_kd(g_s, g_t)
        #     loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'OFD':
            # 假设 model_s 和 model_t 的 forward 支持返回特征列表
            feat_s = model_s(input, is_feat=True)
            feat_t = model_t(input, is_feat=True)

            loss_kd = criterion_kd(feat_s, feat_t)
        # elif opt.distill == 'pkt':
        #     f_s = feat_s[-1]
        #     f_t = feat_t[-1]
        #     loss_kd = criterion_kd(f_s, f_t)
        # elif opt.distill == 'kdsvd':
        #     g_s = feat_s[1:-1]
        #     g_t = feat_t[1:-1]
        #     loss_group = criterion_kd(g_s, g_t)
        #     loss_kd = sum(loss_group)
        # elif opt.distill == 'correlation':
        #     f_s = module_list[1](feat_s[-1])
        #     f_t = module_list[2](feat_t[-1])
        #     loss_kd = criterion_kd(f_s, f_t)
        # elif opt.distill == 'vid':
        #     g_s = feat_s[1:-1]
        #     g_t = feat_t[1:-1]
        #     loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
        #     loss_kd = sum(loss_group)
        # elif opt.distill == 'abound':
        #     # can also add loss to this stage
        #     loss_kd = 0
        # elif opt.distill == 'fsp':
        #     # can also add loss to this stage
        #     loss_kd = 0
        # elif opt.distill == 'factor':
        #     factor_s = module_list[1](feat_s[-2])
        #     factor_t = module_list[2](feat_t[-2], is_factor=True)
        #     loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)
        # pid = PIDController(Kp=0.5, Ki=0.05, Kd=0.1)
        # current_error = loss_div.item()  # 当前蒸馏误差
        # alpha = pid.update(current_error)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy_k(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        _, preds = logit_s.max(1)

        # Calculate true positives for precision and recall
        true_positives = (preds[target == preds]).float().sum()
        precision = true_positives / (preds.float().sum() + 1e-10)  # 防止分母为0
        recall = true_positives / (target.float().sum() + 1e-10)  # 防止分母为0

        precision_meter.update(precision.item(), input.size(0))
        recall_meter.update(recall.item(), input.size(0))
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} '
          'Precision {precision_meter.avg:.3f} Recall {recall_meter.avg:.3f}'.format(
        top1=top1, top5=top5, precision_meter=precision_meter, recall_meter=recall_meter))

    return top1.avg,top5.avg, losses.avg,loss_kd,precision_meter.avg, recall_meter.avg

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
def validate(val_loader, model_s, model_t, criterion, opt, save_heatmap=False):
    """val"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()

    all_teacher_outputs = []
    all_student_outputs = []
    # switch to evaluate mode
    model_s.eval()
    model_t.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output

            student_output = model_s(input)
            teacher_output = model_t(input)
            loss = criterion(student_output, target)

            all_teacher_outputs.append(teacher_output.cpu().numpy())
            all_student_outputs.append(student_output.cpu().numpy())

            # measure accuracy and record loss
            acc1, acc5 = accuracy_k(student_output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # Precision and Recall Calculation
            _, preds = student_output.max(1)
            true_positives = (preds[target == preds]).float().sum()
            precision = true_positives / (preds.float().sum() + 1e-10)  # 防止分母为0
            recall = true_positives / (target.float().sum() + 1e-10)  # 防止分母为0

            precision_meter.update(precision.item(), input.size(0))
            recall_meter.update(recall.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Precision {precision_meter.val:.3f} ({precision_meter.avg:.3f})\t'
                      'Recall {recall_meter.val:.3f} ({recall_meter.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5, precision_meter=precision_meter, recall_meter=recall_meter))
        # 新增：生成热力图
        if save_heatmap and len(all_teacher_outputs) > 0:
            # 合并所有批次的输出
            teacher_outputs = np.vstack(all_teacher_outputs)
            student_outputs = np.vstack(all_student_outputs)

            # 计算相关性矩阵
            corr_matrix = np.zeros((teacher_outputs.shape[1], student_outputs.shape[1]))
            for i in range(teacher_outputs.shape[1]):
                for j in range(student_outputs.shape[1]):
                    corr, _ = pearsonr(teacher_outputs[:, i], student_outputs[:, j])
                    corr_matrix[i, j] = corr

            # 绘制热力图
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                        xticklabels=False, yticklabels=False)
            plt.title('Teacher-Student Output Correlation Matrix')
            plt.xlabel('Student Logits')
            plt.ylabel('Teacher Logits')

            # 保存图像
            heatmap_path = os.path.join(opt.save_folder, 'kd_correlation_heatmap.png')
            plt.savefig(heatmap_path, bbox_inches='tight')
            plt.close()
            print(f'Saved KD correlation heatmap to {heatmap_path}')
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} '
              'Precision {precision_meter.avg:.3f} Recall {recall_meter.avg:.3f}'.format(
               top1=top1, top5=top5, precision_meter=precision_meter, recall_meter=recall_meter))

    return top1.avg, top5.avg, losses.avg, precision_meter.avg, recall_meter.avg
