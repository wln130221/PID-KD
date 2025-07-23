from __future__ import print_function, division

import sys
import time
import torch
from distiller_zoo import DistillKL
from .util import AverageMeter, accuracy_k
from sklearn.metrics import precision_score, recall_score

def train_vanilla(epoch, train_loader, model, criterion, optimizer, args):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    correct = 0.0
    end = time.time()
    for idx, (inputs, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs = inputs.float() #转换为浮点数,方便计算
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        # print("Inputs are on:", inputs.device)
        # print("Labels are on:", labels.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy_k(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))


        _, preds = outputs.max(1)
        correct = preds.eq(labels).sum()
        true_positives = (preds[labels == preds]).float().sum()  # 真实正例
        precision = true_positives / (preds.float().sum() + 1e-10)  # 防止除以0
        recall = true_positives / (labels.float().sum() + 1e-10)  # 防止除以0

        precision_meter.update(precision.item(), inputs.size(0))
        recall_meter.update(recall.item(), inputs.size(0))
        accuracy = (100. * correct.float()) / len(train_loader.dataset)
        # print('accuracy'.format(accuracy))
        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()


        # print info
        if idx % args.print_freq == 0:
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

    return top1.avg,top5.avg, losses.avg,precision_meter.avg, recall_meter.avg


def validate(val_loader, model, criterion, args):
    """Validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc1, acc5 = accuracy_k(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            _, preds = outputs.topk(1, 1, True, True)  # 取得最大预测
            all_preds.extend(preds.cpu().numpy().flatten())  # 展平并存储预测
            all_labels.extend(labels.cpu().numpy().flatten())  # 展平并存储真实标签

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % args.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    # 打印 Precision 和 Recall
    print('Precision: {:.4f}\tRecall: {:.4f}'.format(precision, recall))
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg,precision,recall

