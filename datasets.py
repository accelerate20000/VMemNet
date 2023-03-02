import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import csv
import cv2


class VMDataset(Dataset):

    def __init__(self, data_root_dir, frame_seg, split_method, split, term, transform=None):

        self.split = split
        assert self.split in {'TRAIN', 'VAL'}
        self.term = term
        assert self.term in {'short'}

        self.transform = transform

        self.dir = data_root_dir

        if split == 'TRAIN':
            split_method = list(set(list(range(1, 8001))).difference(set(split_method)))
        self.split_method = split_method
        self.len = len(split_method)
        self.frame_seg = frame_seg

    def __getitem__(self, i):

        with open(self.dir + 'ground-truth_dev-set.csv') as f:
            f_csv = csv.reader(f)
            for num, row in enumerate(f_csv):
                if num == self.split_method[i]:
                    video_name, _ = row[0].split('.webm')
                    frames = np.zeros((42 // self.frame_seg, 360, 640, 3), dtype=np.uint8)
                    for j in range(0, 168, 4*self.frame_seg):#[0,24,48,72,96,120,144]
                        im = Image.open(self.dir + 'frames/' + video_name + '/{}.jpg'.format(j))
                        if im.size != (640, 360):
                            im = transforms.RandomCrop(size=(720, 1280), pad_if_needed=True)(im)#依据给定的size随机裁剪
                            im = np.array(im)
                            im = cv2.resize(im,None,fx=0.5,fy=0.5)
                        frames[j // (4*self.frame_seg), :] = np.array(im)

                    feature_vector = np.fromfile(self.dir + 'dynamic-vectors/' + video_name, dtype=np.float16)
                    short_score = float(row[1])
                    short_ann = int(row[2])
                    long_score = self.short_score = float(row[3])
                    long_ann = int(row[4])
                elif num > self.split_method[i]:
                    break

        frames = frames.transpose(0, 3, 1, 2)

        frames = torch.FloatTensor(frames / 255.)


        if self.transform is not None:
            for temp in range(frames.size(0)):
                frames[temp, :] = self.transform(frames[temp, :])

        feature_vector = torch.Tensor(feature_vector)

        if self.term is 'short':
            short_score = torch.Tensor([short_score])
            short_ann = torch.Tensor([short_ann])
            return frames, feature_vector, short_score, short_ann, video_name
        else:
            print("term error")

    def __len__(self):
        return self.len

