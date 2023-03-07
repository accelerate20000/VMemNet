

import random
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from datasets import *
from utils import *
import numpy as np
from PIL import Image
import os
import math
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

batch_size = 4
frame_num = 7
k_fold = 8
term = 'short'
split_seg = list(range(0, 8001, 8000 // k_fold))  # len=2000
fine_tune_encoder = False

random_list = './list/train_random_list_0.bin'
data_root_dir = '/media/Datasets/VideoMem/'

def choose_fold(which_fold):
    global epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder 
    if which_fold==0:
        checkpoint = './checkpoint_fold0_rc_0.5237312056529898.pth.tar'

    if which_fold==1:
        checkpoint = './checkpoint_fold1_rc_0.5175212760594858.pth.tar'

    if which_fold==2:
        checkpoint = './checkpoint_fold2_rc_0.5148327401644722.pth.tar'

    if which_fold==3:
        checkpoint = './checkpoint_fold3_rc_0.5206630396687968.pth.tar'

    if which_fold==4:
        checkpoint = './checkpoint_fold4_rc_0.5172841501215397.pth.tar'

    if which_fold==5:
        checkpoint = './checkpoint_fold5_rc_0.5404364794276509.pth.tar'

    if which_fold==6:
        checkpoint = './checkpoint_fold6_rc_0.5351826770676688.pth.tar'

    if which_fold==7:
        checkpoint = './checkpoint_fold7_rc_0.5610508470133193.pth.tar'


    epochs_since_improvement = 0
    best_val = 0.

    if random_list is None:
        raise NameError
    else:
        temp_list = np.fromfile(random_list, dtype=int)
        all_video = temp_list.tolist()

    checkpoint = torch.load(checkpoint, map_location="cuda:0")
    start_epoch = checkpoint['epoch'] + 1
    epochs = checkpoint['epoch'] + 2
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    best_val = checkpoint['rc_level']
    encoder = checkpoint['encoder']
    semantic_att = checkpoint['semantic_att']
    spatial_att = checkpoint['spatial_att']
    fusion = checkpoint['fusion']
    encoder_optimizer = checkpoint['encoder_optimizer']
    semantic_att_optimizer = checkpoint['semantic_att_optimizer']
    spatial_att_optimizer = checkpoint['spatial_att_optimizer']
    fusion_optimizer = checkpoint['fusion_optimizer']
    if fine_tune_encoder is True and encoder_optimizer is None:
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    semantic_att = semantic_att.to(device)
    encoder = encoder.to(device)
    spatial_att = spatial_att.to(device)
    fusion = fusion.to(device)
    criterion = nn.MSELoss().to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    split_method = all_video[split_seg[which_fold]:
                             split_seg[which_fold + 1]]
    split_method.sort()
    frame_seg = 42 // frame_num 

    val_loader = torch.utils.data.DataLoader(
        VMDataset(data_root_dir, frame_seg, split_method, 'VAL', term,
                  transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    for epoch in range(start_epoch, epochs):      
        rc, gt_dic, pred_dic = validate(val_loader=val_loader,
                             encoder=encoder,
                             semantic_att=semantic_att,
                             spatial_att=spatial_att,
                             fusion=fusion,
                             criterion=criterion)
        save_predic(which_fold, gt_dic, pred_dic, rc)


def validate(val_loader, encoder, semantic_att, spatial_att, fusion, criterion):
    semantic_att.eval()
    fusion.eval()
    encoder.eval()
    spatial_att.eval()

    pred_scores = []
    GT_scores = []
    ann_num = []
    pred_dict = {}
    gt_dict = {}

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()
    remain_time = TimeShow()

    with torch.no_grad():
        for step, (frames, feature_vector, score, ann, video_name) in enumerate(val_loader):
            # Move to GPU, if available
            frames = frames.to(device)
            feature_vector = feature_vector.to(device)
            score = score.to(device)

            encoded_frames_1024,encoded_frames = encoder(frames)  
            spatial_att_out, beta_out = spatial_att(encoded_frames_1024)
            fusion_out = fusion(encoded_frames_1024, feature_vector)  

            preds_mix= semantic_att(fusion_out).squeeze(1)
            spatial_att_out=spatial_att_out.squeeze(1)
            pred_late_fusion = preds_mix*(0.6) + spatial_att_out*(0.4)
           
            for i in range(pred_late_fusion.size(0)):
                pred_scores.append(pred_late_fusion[i].cpu().numpy().item())
                GT_scores.append(score[i].item())
                ann_num.append(int(ann[i].item()))
                pred_dict[video_name[i]] = pred_late_fusion[i].cpu().numpy().item()
                gt_dict[video_name[i]] = score[i].item()

            # Calculate loss
            loss = criterion(pred_late_fusion, score)

            # Keep track of metrics
            losses.update(loss.item())
            batch_time.update(time.time() - start)
            start = time.time()
            remain_time.update(batch_time.avg * (len(val_loader) - step))
            print_freq = 1
            if step % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Loss {loss.val:.5f} (avg:{loss.avg:.5f})\n'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Remain Time {remain_time.hour:d}h {remain_time.min:d}m {remain_time.sec:d}s\n'
                      .format(step, len(val_loader), loss=losses,
                              batch_time=batch_time, remain_time=remain_time))

        spearman_score = spearman_correlation(pred_scores, GT_scores)
        print(
            '\n * LOSS - {loss.avg:.3f}, Spearman-Correlation - {spearman_score}\n'.format(
                loss=losses,
                spearman_score=spearman_score))
        return spearman_score, gt_dict, pred_dict

if __name__ == '__main__': 
    for i in range(0, 8):
        print('Starting fold-{0}\n\n'.format(i))
        choose_fold(i)
        print('Ending fold-{0}\n\n'.format(i))
