from __future__ import print_function, division

import sys
import time
import torch
from distiller_zoo import DistillKL
from .util import AverageMeter, accuracy_k

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import csv
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json
import numpy as np
def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()


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
        if opt.distill == 'TripleT_KD':
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
        elif opt.distill == 'TripleT_KD':
            loss_kd = criterion_kd(feat_s[-1], feat_t[-1].detach())
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
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


def tsne_visualization(features, labels, method_name, save_folder, n_classes=39):
    """
    生成并保存t-SNE可视化图
    :param features: 特征向量数组
    :param labels: 对应标签数组
    :param method_name: 方法名称（用于标题和文件名）
    :param save_folder: 保存目录
    :param n_classes: 类别数量
    """
    # 随机采样以减少计算量
    sample_size = min(1000, len(features))
    indices = np.random.choice(len(features), sample_size, replace=False)
    sampled_features = features[indices]
    sampled_labels = labels[indices]

    # 动态调整perplexity参数
    perplexity = min(30, max(5, sample_size // 10))  # 确保在合理范围内

    # 执行t-SNE降维
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(sampled_features)

    # 创建可视化
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", n_classes)

    # 为每个类别绘制点（不添加标签）
    for i in range(n_classes):
        idx = sampled_labels == i
        if np.sum(idx) > 0:  # 确保该类别有样本
            plt.scatter(
                tsne_results[idx, 0],
                tsne_results[idx, 1],
                color=palette[i],
                alpha=0.7,
                s=50
            )

    plt.title(f't-SNE Visualization: {method_name}', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(alpha=0.2)

    # 保存图像
    os.makedirs(save_folder, exist_ok=True)
    safe_method_name = method_name.replace(":", "_").replace("/", "_").replace("\\", "_")
    file_path = os.path.join(save_folder, f'tsne_{safe_method_name}.png')
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f'Saved t-SNE visualization to {file_path}')
    return file_path
def validate(val_loader, model_s, model_t, criterion, opt, save_heatmap=False, is_test=False, n_cls=4):
    """val"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()

    # 添加特征收集变量
    all_features = []
    all_labels = []
    all_preds = []
    all_targets = []

    # 获取类别名称
    try:
        class_names = val_loader.dataset.classes
    except AttributeError:
        class_names = [str(i) for i in range(n_cls)]

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

            # 获取学生模型的倒数第二层特征
            if hasattr(model_s, 'get_features'):
                features = model_s.get_features(input)
            else:
                # 尝试标准方式获取特征
                try:
                    _, features = model_s(input, is_feat=True)
                    features = features[-1]  # 取最后一层特征
                except:
                    features = student_output  # 如果无法获取特征，使用输出作为特征

            # 收集特征和标签
            all_features.append(features.cpu().numpy())
            all_labels.append(target.cpu().numpy())

            # measure accuracy and record loss
            acc1, acc5 = accuracy_k(student_output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # Precision and Recall Calculation
            _, preds = student_output.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

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
            teacher_outputs = np.vstack(all_teacher_outputs)
            student_outputs = np.vstack(all_student_outputs)

            corr_matrix = np.zeros((teacher_outputs.shape[1], student_outputs.shape[1]))
            for i in range(teacher_outputs.shape[1]):
                for j in range(student_outputs.shape[1]):
                    corr, _ = pearsonr(teacher_outputs[:, i], student_outputs[:, j])
                    corr_matrix[i, j] = corr

            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                        xticklabels=False, yticklabels=False)
            plt.title('Teacher-Student Output Correlation Matrix')
            plt.xlabel('Student Logits')
            plt.ylabel('Teacher Logits')

            heatmap_path = os.path.join(opt.save_folder, 'kd_correlation_heatmap.png')
            plt.savefig(heatmap_path, bbox_inches='tight')
            plt.close()
            print(f'Saved KD correlation heatmap to {heatmap_path}')

        # 如果是测试集，计算并保存混淆矩阵和详细指标
        results = {}
        if is_test and len(all_preds) > 0:
            # 计算混淆矩阵
            cm = confusion_matrix(all_targets, all_preds)

            # 计算每个类别的指标
            report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)

            # 计算整体指标
            accuracy = top1.avg / 100  # 转换为比例
            precision = precision_meter.avg
            recall = recall_meter.avg

            # 计算混淆矩阵的四个基本量
            TP = np.diag(cm)
            FP = np.sum(cm, axis=0) - TP
            FN = np.sum(cm, axis=1) - TP
            TN = np.sum(cm) - (FP + FN + TP)

            # 创建结果字典
            results = {
                "Model": f"{opt.model_s}_T_{opt.model_t}_{opt.distill}",
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "Confusion Matrix": cm.tolist(),
                "Classification Report": report,
                "TP per class": TP.tolist(),
                "FP per class": FP.tolist(),
                "FN per class": FN.tolist(),
                "TN per class": TN.tolist()
            }

            # 保存混淆矩阵图像
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            confusion_path = os.path.join(opt.save_folder,
                                          f'confusion_matrix_{opt.model_s}_{opt.model_t}_{opt.distill}.png')
            plt.savefig(confusion_path, bbox_inches='tight')
            plt.close()
            print(f'Saved confusion matrix to {confusion_path}')

            # # 保存详细指标到JSON文件
            # metrics_path = os.path.join(opt.save_folder, f'metrics_{opt.model_s}_{opt.model_t}_{opt.distill}.json')
            # with open(metrics_path, 'w') as f:
            #     json.dump(results, f, indent=4)
            # print(f'Saved detailed metrics to {metrics_path}')
            #
            # # 保存简要指标到CSV文件
            # csv_path = os.path.join(opt.save_folder, 'test_results.csv')
            # file_exists = os.path.isfile(csv_path)
            #
            # with open(csv_path, 'a', newline='') as csvfile:
            #     fieldnames = ['Model', 'Accuracy', 'Precision', 'Recall', 'Confusion Matrix Path', 'Metrics Path']
            #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #
            #     if not file_exists:
            #         writer.writeheader()
            #
            #     writer.writerow({
            #         'Model': results["Model"],
            #         'Accuracy': f"{accuracy * 100:.2f}%",
            #         'Precision': f"{precision * 100:.2f}%",
            #         'Recall': f"{recall * 100:.2f}%",
            #         'Confusion Matrix Path': confusion_path,
            #         'Metrics Path': metrics_path
            #     })
            # print(f'Appended test results to {csv_path}')

            # 添加t-SNE可视化
            if len(all_features) > 0:
                all_features = np.vstack(all_features)
                all_labels = np.hstack(all_labels)

                # 生成方法名称
                method_name = f"{opt.model_s}_T_{opt.model_t}_{opt.distill}"

                # 执行t-SNE可视化
                tsne_path = tsne_visualization(
                    all_features,
                    all_labels,
                    method_name,
                    opt.save_folder,
                    n_cls
                )

                # # 将t-SNE路径保存到结果中
                # results['t-SNE Path'] = tsne_path
                # # 更新JSON文件
                # with open(metrics_path, 'w') as f:
                #     json.dump(results, f, indent=4)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} '
          'Precision {precision_meter.avg:.3f} Recall {recall_meter.avg:.3f}'.format(
        top1=top1, top5=top5, precision_meter=precision_meter, recall_meter=recall_meter))

    return top1.avg, top5.avg, losses.avg, precision_meter.avg, recall_meter.avg