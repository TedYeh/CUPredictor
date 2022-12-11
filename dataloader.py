from random import shuffle
import torch
import csv, os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Dataset, SequentialSampler
from sklearn.model_selection import train_test_split
from torchvision.io import read_image
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import math

class imgDataset(Dataset):
    def __init__(self, path, mode='train'):
        self.path = path  
        self.mode = mode
        self.transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        } 
        self.trans = self.transform[mode]
        self.data = self.get_data()

    def get_data(self):
        data = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                file_name_list = line.split(' ')
                if not self.mode in file_name_list:continue
                label, h = 0 if file_name_list[2]=="big" else 1, int(file_name_list[3])
                data.append([os.path.join('images', file_name_list[0], file_name_list[2], file_name_list[1]), label, h])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):   
        img_path, label, h = self.data[idx]
        inp_img = Image.open(img_path).convert("RGB")
        image_tensor = self.trans(inp_img)
        return image_tensor, label, torch.tensor(h, dtype=torch.float)

if __name__ == "__main__":
    train_dataset = imgDataset('labels.txt', mode='train')
    test_dataset = imgDataset('labels.txt', mode='val')
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print(len(train_dataset), len(test_dataset))
    print(next(iter(train_dataloader)))