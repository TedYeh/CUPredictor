from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from dataloader import imgDataset
import time
import os
import copy

PATH = './images/'

class CUPredictor(nn.Module):
    def __init__(self, num_class=2):
        super(CUPredictor, self).__init__()
        self.base = torchvision.models.resnet50(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad = False

        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Linear(num_ftrs, num_ftrs//2)         
        self.classifier = nn.Linear(num_ftrs//2, num_class)
        self.height_regressor = nn.Linear(num_ftrs//2, 1)
        self.relu = nn.ReLU()

    def forward(self, input_img):
        output = self.base(input_img)
        predict_cls = self.classifier(output)
        predict_height = self.relu(self.height_regressor(output))
        return predict_cls, predict_height


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated 
    plt.savefig(f'images/preds/prediction.png')   

def train_model(model, device, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_ce_loss = 0.0
            running_rmse_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, heights in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                heights = heights.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_c, outputs_h = model(inputs)
                    _, preds = torch.max(outputs_c, 1)
                    ce_loss = ce(outputs_c, labels)
                    rmse_loss = torch.sqrt(mse(outputs_h, heights.unsqueeze(-1)))
                    loss = ce_loss + rmse_loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                # statistics
                running_ce_loss += ce_loss.item() * inputs.size(0)
                running_rmse_loss += rmse_loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)            

            epoch_ce_loss = running_ce_loss / dataset_sizes[phase]
            epoch_mse_loss = running_rmse_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} CE_Loss: {epoch_ce_loss:.4f} MSE_Loss: {epoch_mse_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, device, dataloaders, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'pred: {class_names[preds[j]]}|tar: {class_names[labels[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)      

def evaluation(model, epoch, device, dataloaders):
    model.load_state_dict(torch.load(f'models/model_{epoch}.pt'))
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            print(preds)

def inference(img_path, classes = ['big', 'small'], epoch = 25):
    model = model = CUPredictor()
    model.load_state_dict(torch.load(f'models/model_{epoch}.pt'))
    model.eval()

    device = torch.device("cpu")
    trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    inp_img = Image.open(img_path).convert("RGB")

    image_tensor = trans(inp_img)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        inputs = image_tensor.to(device)
        outputs_c, outputs_h = model(inputs)
        _, preds = torch.max(outputs_c, 1)
        idx = preds.numpy()[0]
    return outputs_c, classes[idx], f"{outputs_h.numpy()[0][0]:.2f}"

def main(epoch = 15, mode = 'val'):
    cudnn.benchmark = True
    plt.ion()   # interactive mode
    model = CUPredictor()
    train_dataset = imgDataset('labels.txt', mode='train')
    test_dataset = imgDataset('labels.txt', mode='val')
    dataloaders = {
                    "train": DataLoader(train_dataset, batch_size=32, shuffle=True),
                    "val": DataLoader(test_dataset, batch_size=32, shuffle=True)
    }
    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(test_dataset)
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")   
    model = model.to(device)
    model_conv = train_model(model, device, dataloaders, dataset_sizes, num_epochs=epoch)
    torch.save(model_conv.state_dict(), f'models/model_{epoch}.pt')

def divide_class_dir(path):
    file_list = os.listdir(path)
    for img_name in file_list:
        dest_path = os.path.join(path, img_name.split('-')[3])
        if not os.path.exists(dest_path):
            os.mkdir(dest_path)  # 建立資料夾
        os.replace(os.path.join(path, img_name), os.path.join(dest_path, img_name))

def get_label(types):
    with open('labels.txt', 'w', encoding='utf-8') as f:
        for f_type in types:
            for img_type in CLASS:
                path = os.path.join('images', f_type, img_type)
                file_list = os.listdir(path)
                for file_name in file_list:
                    file_name_list = file_name.split('-')
                    f.write(" ".join([f_type, file_name, img_type, file_name_list[4].split('_')[0], '\n']))

if __name__ == "__main__":
    
    CLASS = ['big', 'small']
    mode = 'train'
    get_label(['train', 'val'])
    main(35, mode = mode)
    
    outputs, preds, heights = inference('images/test/lin.png', CLASS, epoch=35)
    print(outputs, preds, heights)
    #print(CUPredictor())
    #divide_class_dir('./images/train')
    #divide_class_dir('./images/val')
    ''''''