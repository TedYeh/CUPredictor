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
from transformers import AutoImageProcessor

class imgDataset(Dataset):
    def __init__(self, path, mode='train', use_processor=True):
        self.path = path  
        self.mode = mode
        self.use_processor = use_processor
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
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

    def convert_body_to_int(self, pos, file_name_list):
        body_str = file_name_list[1].split('-')[pos]
        if not body_str: body_str = '62'
        body = int(body_str[1:3]) if not body_str.isdigit() else int(body_str)
        body = 100+body if body <= 25 else body
        return body

    def get_data(self):
        data = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                file_name_list = line.split(' ')
                if not self.mode in file_name_list:continue
                label, h = 0 if file_name_list[2]=="big" else 1, float(file_name_list[3])
                b = self.convert_body_to_int(0, file_name_list)
                w = self.convert_body_to_int(1, file_name_list)
                hh = self.convert_body_to_int(2, file_name_list)
                data.append([os.path.join('images', file_name_list[0], file_name_list[2], file_name_list[1]), label, h, b, w, hh])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):   
        img_path, label, h, b, w, hh = self.data[idx]
        inp_img = Image.open(img_path).convert("RGB")
        if not self.use_processor: image_tensor = self.trans(inp_img)
        else:image_tensor = self.image_processor(images=inp_img, return_tensors="pt")
        return image_tensor, label, torch.tensor(h, dtype=torch.float), torch.tensor(b, dtype=torch.float), torch.tensor(w, dtype=torch.float), torch.tensor(hh, dtype=torch.float)

if __name__ == "__main__":
    train_dataset = imgDataset('labels.txt', mode='train')
    test_dataset = imgDataset('labels.txt', mode='val')
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print(len(train_dataset), len(test_dataset))
    print(next(iter(train_dataloader)))