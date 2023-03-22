import argparse
import pandas as pd
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.optim as optim
import torch
import torch.nn as nn
from model.circle_detector import Net
from data_handling.noisy_circle_dataset import NoisyImages
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch Model Training')

parser.add_argument('-n', '--name',default='v8', type=str,
                    help='Name of the experiment.')
parser.add_argument('-o', '--out_file', default='new_out.txt',
                    help='path to output features file')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--resume',
                    default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-d','--data', default='train_set.csv', metavar='DIR',
                    help='path to imagelist file')
parser.add_argument('--print_freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number')
parser.add_argument('--save_freq', default=5, type=int,
                    help='Number of epochs to save after')
parser.add_argument('-e','--envhome', default='', type=str,
                    help='Home directory')

def main():
    args = parser.parse_args()
    print(args)

    print("=> creating model")
    model = Net()
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())

    print("Number of params is ",pytorch_total_params)

    if args.resume:
        print("=> loading checkpoint: " + args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)
        args.start_epoch = int(args.resume.split('/')[1].split('_')[0])
        print("=> checkpoint loaded. epoch : " + str(args.start_epoch))
    else:
        print("=> Start from the scratch ")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), 0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 10], gamma=0.5, last_epoch=-1)

    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    trainset = NoisyImages(
        args.envhome+args.data,
        transforms.Compose([
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    output = open(args.envhome+args.out_file, "w")
    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, criterion1, criterion2, optimizer, epoch, args, device, len(trainset), output)
        scheduler.step()


def train(train_loader, model, criterion1, criterion2, optimizer, epoch, args, device, len, file):

    # switch to train mode
    model.train()
    running_loss = 0.0

    for i, (images, target) in tqdm(enumerate(train_loader)):

        images = images.to(device)
        target = target.to(device)

        output = model(images)

        loss = criterion1(output, target/200) + 0.1 * criterion2(output, target/200)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % args.print_freq == args.print_freq - 1 or i == int(len/args.batch_size):    # print every 50 mini-batches
            new_line = 'Epoch: [%d][%d/%d] loss: %f' % \
                       (epoch + 1, i + 1, int(len/args.batch_size) + 1, running_loss / args.print_freq)
            file.write(new_line + '\n')
            print(new_line)
            running_loss = 0.0

        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), args.envhome + 'saved_models/' + str(epoch) + '_epoch_' + args.name + '_checkpoint.pth.tar')

if __name__=='__main__':
    main()

# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate"""
#     lr = 0.001
#     if 20 < epoch <= 30:
#         lr = 0.0001
#     elif 30 < epoch :
#         lr = 0.00001
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     print("learning rate -> {}\n".format(lr))
