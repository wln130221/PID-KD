from __future__ import print_function
import os
import argparse
import socket
import time
import csv
import tensorboard_logger as tb_logger
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
import torch.nn as nn
from models import model_dict
from helper.util import adjust_learning_rate, accuracy_k, AverageMeter
from helper.loops import train_vanilla as train, validate
import torch
from dataset.data import get_PestDisease_dataloaders
import torchvision.transforms as transforms
def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('PyTorch Knowledge Distillation - Teacher training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=300, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='',
                        choices=['resnet8',  'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', ])
    parser.add_argument('--dataset', type=str, default='Wheat_split', choices=['PlantVillage_split','Rice Leaf Disease1','soybean','Wheat_split'], help='dataset')
    parser.add_argument('--dataset_path', type=str, default='./dataset/Wheat_split', help='dataset_path')
    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/models'
        opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def main():
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    torch.cuda.empty_cache()
    best_acc = 0

    opt = parse_option()

    # dataloader
    if opt.dataset == 'PlantVillage_split':
        train_loader, val_loader, test_loader = get_PestDisease_dataloaders(data_dir=opt.dataset_path,
                                                                            batch_size=opt.batch_size,
                                                                            num_workers=opt.num_workers)
        n_cls = 39
    elif opt.dataset == 'Wheat_split':
        train_loader, val_loader, test_loader = get_PestDisease_dataloaders(data_dir=opt.dataset_path,
                                                                            batch_size=opt.batch_size,
                                                                            num_workers=opt.num_workers)
        n_cls = 4
    elif opt.dataset == 'soybean':
        train_loader, val_loader, test_loader = get_PestDisease_dataloaders(data_dir=opt.dataset_path,
                                                                            batch_size=opt.batch_size,
                                                                            num_workers=opt.num_workers)
        n_cls = 3
    elif opt.dataset == 'Rice Leaf Disease':
        train_loader, val_loader, test_loader = get_PestDisease_dataloaders(data_dir=opt.dataset_path,
                                                                            batch_size=opt.batch_size,
                                                                            num_workers=opt.num_workers)
        n_cls = 4
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = model_dict[opt.model](num_classes=n_cls)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    model = model.to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")


    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss().to(device)


    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")

    csv_file_path = os.path.join(opt.save_folder, 'train_log.csv')
    os.makedirs(opt.save_folder, exist_ok=True)
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train_Acc', 'Top5_Train_Acc', 'Train_Loss', 'Test_Acc', 'Top5_Test_Acc', 'Test_Loss'])
    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc,top_5, train_loss,precision,recall = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss,precision,recall = validate(val_loader, model, criterion, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, train_acc, top_5, train_loss, test_acc, test_acc_top5, test_loss])
        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)
    test_acc, test_acc_top5, test_loss, precision, recall = validate(val_loader, model, criterion, opt)
    print(f'Test Acc@1: {test_acc:.3f}, Test Acc@5: {test_acc_top5:.3f}, Test Loss: {test_loss:.3f}')
    print(f'Test Precision: {precision:.3f}, Test Recall: {recall:.3f}')

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)

    # save_conv_weights(model, os.path.join(opt.save_folder, f'{opt.model}_conv_weights.pth'))
    # save_fc_weights(model, os.path.join(opt.save_folder, f'{opt.model}_fc_weights.pth'))




if __name__ == '__main__':
    main()
