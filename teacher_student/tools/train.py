# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import logging
import timeit
import torch.nn.functional as F

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import unet
import datasets
from configs import config
from configs import update_config
from utils.criterion import OhemCrossEntropy, SILogLoss, CESLoss
from utils.function import train, validate
from utils.utils import create_logger
from models.unet_adaptive_bins import UnetAdaptiveBins
import models_seg
from utils.utils import AverageMeter
import time
from infer import Inference_depth, Inference_seg

def load_checkpoint(fpath, model, optimizer=None):
    ckpt = torch.load(fpath, map_location='cpu')
    if optimizer is None:
        optimizer = ckpt.get('optimizer', None)
    else:
        optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt['epoch']

    if 'model' in ckpt:
        ckpt = ckpt['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    modified = {}  # backward compatibility to older naming of architecture blocks
    for k, v in load_dict.items():
        if k.startswith('adaptive_bins_layer.embedding_conv.'):
            k_ = k.replace('adaptive_bins_layer.embedding_conv.',
                           'adaptive_bins_layer.conv3x3.')
            modified[k_] = v
            # del load_dict[k]

        elif k.startswith('adaptive_bins_layer.patch_transformer.embedding_encoder'):

            k_ = k.replace('adaptive_bins_layer.patch_transformer.embedding_encoder',
                           'adaptive_bins_layer.patch_transformer.embedding_convPxP')
            modified[k_] = v
            # del load_dict[k]
        else:
            modified[k] = v  # else keep the original

    model.load_state_dict(modified)
    return model, optimizer, epoch


def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    return model
    
def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr



def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def pixel_acc(pred, label):
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

def main():
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)        

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    if torch.cuda.device_count() != len(gpus):
        print("The gpu numbers do not match!")
        return 0
 
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    # prepare data
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        mode = "train")
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True)

    sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                            thres=config.LOSS.OHEMTHRES,
                            min_kept=config.LOSS.OHEMKEEP,
                            weight=train_dataset.class_weights)
    
    depth_criterion = SILogLoss()
    soft_criterion = CESLoss()
    
    
    
    
    student_model = unet.UNet(3,5).cuda()
    
    depth_model = UnetAdaptiveBins.build(n_bins=256, min_val=1e-3, max_val=1)
    pretrained_path = "/mnt/RG/AdaBins/pretrained/AdaBins_nyu.pt"
    depth_model, _, _ = load_checkpoint(pretrained_path, depth_model)
    depth_teacher = Inference_depth(depth_model)
    
    seg_model = models_seg.pidnet.get_pred_model('pidnet-s',5)
    seg_model = load_pretrained(seg_model, "/mnt/RG/PIDNet/output/auto/pidnet_small_auto/checkpoint.pth.tar")
    seg_teacher = Inference_seg(seg_model)
    
    

    # optimizer
    params_dict = dict(student_model.named_parameters())
    params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

    optimizer = torch.optim.SGD(params,
                            lr=config.TRAIN.LR,
                            momentum=config.TRAIN.MOMENTUM,
                            weight_decay=config.TRAIN.WD,
                            nesterov=config.TRAIN.NESTEROV,
                            )

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

    last_epoch = 0

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    real_end = 120+1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch
    
    for epoch in range(last_epoch, real_end):

        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)



        batch_time = AverageMeter()
        ave_loss = AverageMeter()
        ave_acc  = AverageMeter()
        avg_sem_loss = AverageMeter()
        avg_depth_loss = AverageMeter()
        avg_cse_loss = AverageMeter()

        tic = time.time()
        cur_iters = epoch*epoch_iters
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        
        for i_iter, batch in enumerate(trainloader, 0):
            images, labels = batch
            images = images.float().cuda()
            labels = labels.long().cuda()
            
            seg, depth = student_model(images)
            
            _, teacher_depth_pred = depth_teacher.predict(images)
            teacher_depth_pred = torch.from_numpy(teacher_depth_pred).cuda()
            
            teacher_seg_pred = seg_teacher.predict(images)
            teacher_seg_pred = teacher_seg_pred.cuda()
            
            loss_s = sem_criterion(seg, labels) 
            loss_st = soft_criterion(seg, teacher_seg_pred)
            
            mask = depth > 1e-3
            loss_d = depth_criterion(depth, teacher_depth_pred, mask=mask.to(torch.bool), interpolate=True)
            
            loss = loss_s + loss_st + 0.5*loss_d
            losses = [loss_s, loss_st, 0.5*loss_d]
            
            acc  = pixel_acc(seg, labels)
            loss = loss.mean()
            acc  = acc.mean()
            
            student_model.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            # measure elapsed time
            batch_time.update(time.time() - tic)
            tic = time.time()

            # update average loss
            ave_loss.update(loss.item())
            ave_acc.update(acc.item())
            avg_sem_loss.update(losses[0].mean().item())
            avg_depth_loss.update(losses[2].mean().item())
            avg_cse_loss.update(losses[1].mean().item())
            

            lr = adjust_learning_rate(optimizer,
                                    config.TRAIN.LR,
                                    num_iters,
                                    i_iter+cur_iters)
            if i_iter % config.PRINT_FREQ == 0:
                lr = [x['lr'] for x in optimizer.param_groups]
                msg = f'Epoch: [{epoch}/{config.TRAIN.END_EPOCH}] Iter:[{i_iter}/{epoch_iters}], Time: {batch_time.average():.2f},  \
                        lr: {lr}, Loss: {ave_loss.average():.6f}, Acc:{ave_acc.average():.6f}, Semantic loss: {avg_sem_loss.average():.6f}, CSE loss: {avg_cse_loss.average():.6f}, Depth loss: {avg_depth_loss.average():.6f}' 
                logging.info(msg)
            
            writer.add_scalar('train_loss', ave_loss.average(), global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
                
        logger.info('=> saving checkpoint to {}'.format(
            final_output_dir + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch+1,
            'state_dict': student_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir,'checkpoint.pth.tar'))


    torch.save(student_model.state_dict(),
            os.path.join(final_output_dir, 'final_state.pt'))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % np.int64((end-start)/3600))
    logger.info('Done')

if __name__ == '__main__':
    main()
