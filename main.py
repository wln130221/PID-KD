"""
the general training framework
"""

from __future__ import print_function
import torchvision
import os
import argparse
import socket
import time
import torchvision.transforms as transforms
import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from dataset.data import get_PestDisease_dataloaders
from models import model_dict
import matplotlib.pyplot as plt
from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss,FADLoss,TDA_KLLoss, Attention, RKDLoss,OFD,ReviewKD,CRD,CAT_KD, AttnFD, HDKD, DKD,NormKD
import csv
from helper.distill import train_distill as train
from helper.loops import validate as validate



def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('PyTorch Knowledge Distillation - Student training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=300, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='PlantVillage_split', choices=['PlantVillage_split','Rice Leaf Disease1','soybean','Wheat_split'], help='dataset')
    parser.add_argument('--dataset_path', type=str, default='./dataset/PlantVillage_split', help='dataset_path')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8x4',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'OD_MobileNetV2','ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default='./save/models/ResNet50_PlantVillage_split_lr_0.001_decay_0.0005_trial_0/ResNet50_best.pth', help='teacher model snapshot')
    # distillation
    parser.add_argument('--distill', type=str, default='TripleT_KD', choices=['kd', 'hint', 'TripleT_KD','OFD', 'ReviewKD',
                                                                      'CAT-KD', 'AttnFD', 'HDKD', 'DKD',
                                                                      'NormKD', 'RKD', 'CRD', 'factor', 'nst'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0.5, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.5, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=7, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t_s', default=0.04, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_t_t', default=0.10, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])

# hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment

    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.trial)
    opt.model_name = opt.model_name.replace(':', '_').replace('\\', '_').replace('/', '_')
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    print(segments)
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]



def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model

def main():
    import torch
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_acc = 0

    opt = parse_option()
    # 添加CSV文件保存目录
    csv_file_path = os.path.join(opt.save_folder, 'train_log.csv')
    os.makedirs(opt.save_folder, exist_ok=True)
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train_Acc', 'Top5_Train_Acc', 'Train_Loss', 'Test_Acc', 'Top5_Test_Acc', 'Test_Loss'])
    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'PlantVillage_split':
        train_loader, val_loader,test_loader = get_PestDisease_dataloaders(data_dir=opt.dataset_path, batch_size=opt.batch_size,
                                                               num_workers=opt.num_workers)
        n_cls = 39
    elif opt.dataset == 'Wheat_split':
        train_loader, val_loader,test_loader = get_PestDisease_dataloaders(data_dir=opt.dataset_path, batch_size=opt.batch_size,
                                                               num_workers=opt.num_workers)
        n_cls = 4
    elif opt.dataset == 'soybean':
        train_loader, val_loader,test_loader = get_PestDisease_dataloaders(data_dir=opt.dataset_path, batch_size=opt.batch_size,
                                                               num_workers=opt.num_workers)
        n_cls = 3
    elif opt.dataset == 'Rice Leaf Disease':
        train_loader, val_loader,test_loader = get_PestDisease_dataloaders(data_dir=opt.dataset_path, batch_size=opt.batch_size,
                                                               num_workers=opt.num_workers)
        n_cls = 4
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t,n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)
    total_params = sum(p.numel() for p in model_s.parameters())
    print(f"Total parameters: {total_params}")

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    if opt.distill == 'TripleT_KD':
        criterion_div = TDA_KLLoss(opt.kd_T)
    else:
        criterion_div = DistillKL(opt.kd_T)

    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
    elif opt.distill == 'TripleT_KD':
        criterion_kd = FADLoss()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'CRD':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        n_data = len(train_loader.dataset)
        opt.n_data = n_data
        criterion_kd = CRD(opt.s_dim, opt.t_dim, n_data)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'OFD':
        criterion_kd = OFD()
    elif opt.distill == 'ReviewKD':
        criterion_kd = ReviewKD()
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'ReviewKD':
        criterion_kd = ReviewKD()
    elif opt.distill == 'CAT-KD':
        criterion_kd = CAT_KD()
    elif opt.distill == 'AttnFD':
        criterion_kd = AttnFD()
    elif opt.distill == 'HDKD':
        criterion_kd = HDKD()
    elif opt.distill == 'DKD':
        criterion_kd = DKD()
    elif opt.distill == 'NormKD':
        criterion_kd = NormKD()
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # 分类损失，学生预测和真实值
    criterion_list.append(criterion_div)    # 输出层kd损失
    criterion_list.append(criterion_kd)     # 中间层

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    # optimizer = optim.Adam(trainable_list.parameters(),
    #                        lr=opt.learning_rate)
    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    module_list = module_list.to(device)
    criterion_list = criterion_list.to(device)
    if torch.cuda.is_available():
        cudnn.benchmark = True

    # validate teacher accuracy
    # teacher_acc, _, _ ,_,_= validate(val_loader, model_t, criterion_cls, opt)
    # print('teacher accuracy: ', teacher_acc)

    # routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")
        time1 = time.time()
        train_acc, train_acctop5,train_loss,loss_kd,precision,recall = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        test_acc, tect_acc_top5, test_loss ,precision,recall= validate(val_loader, model_s, model_t, criterion_cls, opt)
        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, train_acc, train_acctop5, train_loss, test_acc, tect_acc_top5, test_loss])
        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
        if opt.dataset == 'PlantVillage_split':
            n_cls = 39
        elif opt.dataset == 'Wheat_split':
            n_cls = 4
        elif opt.dataset == 'soybean':
            n_cls = 3
        elif opt.dataset == 'Rice Leaf Disease':
            n_cls = 4
        else:
            raise NotImplementedError(opt.dataset)
    print('best accuracy:', best_acc)
    test_acc, test_acc_top5, test_loss, precision, recall = validate(
        test_loader, model_s, model_t, criterion_cls, opt,
        save_heatmap=True, is_test=True,n_cls=n_cls
    )
    print(f'Test Acc@1: {test_acc:.3f}, Test Acc@5: {test_acc_top5:.3f}, Test Loss: {test_loss:.3f}')
    print(f'Test Precision: {precision:.3f}, Test Recall: {recall:.3f}')
    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)

if __name__ == '__main__':
    main()