"""
Train a neural network on a training dataset. Based on the PyTorch ImageNet example.
"""
import argparse
import os
import random
import shutil
import time
import warnings
import glob

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import webdataset as wds
import pandas as pd

import networks

model_names = sorted(
    name
    for name in networks.__dict__
    if name.islower() and not name.startswith("__")
    and callable(networks.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset to train on')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg11stochastic',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: vgg11stochastic)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=-1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--save-every-epoch', action='store_true',
                    help='whether to save all epochs')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop-lr', '--drop-learning-rate', default=10, type=int,
                    metavar='N', help='drop learning rate after this many epochs (default: 10)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--dropout', default=0.5, type=float,
                    metavar='N', help='set the dropout rate (default: 0.5)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=str,
                    help='GPU id to use.')
parser.add_argument('-c', '--classifier-size', default=4096, type=int,
                    help='Size of the fully connected layers. Defaults to 4096.')
parser.add_argument('--num', default=None, type=int,
                    help='Iteration of the model. Set this when training multiple version of the same model.')

best_prec1 = 0
device = torch.device('cpu')
target_vectors = None

args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')

if args.gpu is not None:
    warnings.warn('You have chosen a specific GPU. This will completely '
                  'disable data parallelism.')

# Data loading code
transform=transforms.Compose([
    #transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


print(f'{args.data}/train/*.tar')
train_dataset = (
    wds.WebDataset(glob.glob(f'{args.data}/train/*.tar'), shardshuffle=True)
    .shuffle(1000)
    .decode('pil')
    .to_tuple('png;jpg;jpeg cls')
    .map_tuple(transform, None)
    .batched(args.batch_size)
)
test_dataset = (
    wds.WebDataset(glob.glob(f'{args.data}/test/*.tar'), shardshuffle=False)
    .decode('pil')
    .to_tuple('png;jpg;jpeg cls')
    .map_tuple(transform, None)
    .batched(args.batch_size)
)
train_metadata = pd.read_csv(f'{args.data}/train.csv', index_col=0)
test_metadata = pd.read_csv(f'{args.data}/test.csv', index_col=0)
train_len = len(train_metadata) // args.batch_size
test_len = len(test_metadata) // args.batch_size
target_num_classes = len(train_metadata['label'].unique())

print("=> creating model '{}'".format(args.arch))
if args.resume:
    if not os.path.isfile(args.resume):
        raise ValueError("No checkpoint found at '{}'".format(args.resume))

    print(f"=> loading checkpoint '{args.resume}'")
    checkpoint = torch.load(args.resume, map_location='cpu')
    model = networks.__dict__[args.arch].from_checkpoint(checkpoint, num_classes=target_num_classes, classifier_size=args.classifier_size)
else:
    model = networks.__dict__[args.arch](num_classes=target_num_classes, classifier_size=args.classifier_size)

if args.gpu is not None:
    device = torch.device(args.gpu)
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(args.workers)

if hasattr(model, 'set_dropout'):
    model.set_dropout(args.dropout)

print("=> using device", device)
model = model.to(device)
model.share_memory()

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.get_sgd_params(args),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

if args.resume:
    if(args.start_epoch) == -1:
        args.start_epoch = checkpoint['epoch']
    #best_prec1 = checkpoint['best_prec1']
    #optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.resume, checkpoint['epoch']))

print(model)
model_name = args.arch
if args.classifier_size != 4096:
    model_name += f'-cs{args.classifier_size}'
model_name += '_'
if args.resume and ('imagenet' in args.resume):
    model_name += 'first_imagenet_then_'
model_name += f'_{os.path.basename(args.data)}'
if args.num is not None:
    model_name += f'_iter-{args.num}'
print(f'Model name: {model_name}')

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=None,
    num_workers=args.workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=None,
    num_workers=args.workers, pin_memory=False)

def train(train_loader, model, criterion, optimizer, epoch):
    print("=> training on device", device)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        torch.cuda.empty_cache()

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device, non_blocking=True)
        if type(target) == list:
            target = [t.to(device, non_blocking=True) for t in target]
        else:
            target = target.to(device, non_blocking=True)

        # compute output
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            #print('Density:', output[0].shape[1] - torch.sum(output[0] == 0, dim=1).float().mean())
            print(f'Epoch: [{epoch}][{i:04d}/{train_len:04d}] '
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                  f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})', flush=True)


def validate(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(test_loader):
            input = input.to(device, non_blocking=True)
            if type(target) == list:
                target = [t.to(device, non_blocking=True) for t in target]
            else:
                target = target.to(device, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(f'Test: [{i:04d}/{test_len:04d}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})', flush=True)

        print(f' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}', flush=True)

    return top1.avg


def save_checkpoint(state, is_best, epoch=None):
    if epoch is not None:
        filename = f'{model_name}_epoch-{epoch:02d}.pth.tar'
    else:
        filename = f'{model_name}_checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'{model_name}_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every drop_lr epochs"""
    #lr = args.lr * (0.1 ** (epoch // args.drop_lr))
    for param_group in optimizer.param_groups:
        if epoch % args.drop_lr == 0:
            #print(param_group['lr'], '-->', lr)
            #param_group['lr'] = lr
            param_group['lr'] *= 0.1


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        if type(output) == tuple:
            assert type(target) == list
            output = output[-1]
            target = target[-1]

        batch_size = target.size(0)

        if len(target.size()) == 1:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
        else:
            target_indices = torch.cdist(target, target_vectors).argsort()[:, 0]
            pred = torch.cdist(output, target_vectors).argsort()[:, :maxk].t()
            correct = pred.eq(target_indices.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if args.evaluate:
    validate(test_loader, model, criterion)
else:
    for epoch in range(0 if args.start_epoch == -1 else args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(test_loader, model, criterion).cpu()

        print('|| 1 ||', flush=True)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            'args' : args,
        }, is_best, epoch=epoch+1 if args.save_every_epoch else None)

        print('|| 2 ||', flush=True)
