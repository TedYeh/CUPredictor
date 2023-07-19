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
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoImageProcessor, ResNetModel

PATH = './images/'

class CUPredictor_v2(nn.Module):
    def __init__(self, num_class=2):
        super(CUPredictor_v2, self).__init__()
        self.base = ResNetModel.from_pretrained("microsoft/resnet-50")
        num_ftrs = 2048
        #self.base.fc = nn.Linear(num_ftrs, num_ftrs//2) 
        self.classifier = nn.Linear(num_ftrs, num_class)
        self.height_regressor = nn.Linear(num_ftrs, 1)
        self.relu = nn.ReLU()

    def forward(self, input_img):
        output = self.base(input_img['pixel_values'].squeeze(1)).pooler_output.squeeze() 
        predict_cls = self.classifier(output)
        predict_height = self.relu(self.height_regressor(output))
        return predict_cls, predict_height

class CUPredictor(nn.Module):
    def __init__(self, num_class=2):
        super(CUPredictor, self).__init__()
        self.base = torchvision.models.resnet50(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad = False

        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Sequential(
          nn.Linear(num_ftrs, num_ftrs//4),
          nn.ReLU(),
          nn.Linear(num_ftrs//4, num_ftrs//8),
          nn.ReLU()
        ) 
        self.classifier = nn.Linear(num_ftrs//8, num_class)
        self.regressor_h = nn.Linear(num_ftrs//8, 1)
        self.regressor_b = nn.Linear(num_ftrs//8, 1)
        self.relu = nn.ReLU()

    def forward(self, input_img):
        output = self.base(input_img)    
        predict_cls = self.classifier(output)
        predict_height = self.relu(self.regressor_h(output))
        predict_bust = self.relu(self.regressor_b(output))
        return predict_cls, predict_height, predict_bust


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
    optimizer = optim.AdamW(model.parameters(), lr=0.0008)
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
            for inputs, labels, heights, bust in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                heights = heights.to(device)
                bust = bust.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_c, outputs_h, outputs_b = model(inputs)
                    _, preds = torch.max(outputs_c, 1)
                    ce_loss = ce(outputs_c, labels)
                    rmse_loss_h = torch.sqrt(mse(outputs_h, heights.unsqueeze(-1)))
                    rmse_loss_b = torch.sqrt(mse(outputs_b, bust.unsqueeze(-1)))
                    rmse_loss = rmse_loss_h + rmse_loss_b
                    loss = ce_loss + (rmse_loss)*2

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
            epoch_rmse_loss = running_rmse_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} CE_Loss: {epoch_ce_loss:.4f} RMSE_Loss: {epoch_rmse_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        #if epoch %2 == 0 and phase == 'val':print(outputs_c, outputs_h)
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

def inference(img_path, classes = ['big', 'small'], epoch = 6):
    device = torch.device("cpu")

    model = model = CUPredictor()
    model.load_state_dict(torch.load(f'models/model_{epoch}.pt'))
    # load image-to-text model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()

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
        outputs_c, outputs_h, outputs_b = model(inputs)
        _, preds = torch.max(outputs_c, 1)
        idx = preds.numpy()[0]

        # unconditional image captioning
        inputs = processor(inp_img, return_tensors="pt")
        out = model_blip.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)
    return outputs_c, classes[idx], f"{outputs_h.numpy()[0][0]:.2f}", f"{outputs_b.numpy()[0][0]:.2f}", description

def main(epoch = 15, mode = 'val'):
    cudnn.benchmark = True
    plt.ion()   # interactive mode
    model = CUPredictor()
    train_dataset = imgDataset('labels.txt', mode='train', use_processor=False)
    test_dataset = imgDataset('labels.txt', mode='val', use_processor=False)
    dataloaders = {
                    "train": DataLoader(train_dataset, batch_size=64, shuffle=True),
                    "val": DataLoader(test_dataset, batch_size=64, shuffle=False)
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
    epoch = 10
    main(epoch, mode = mode)
    
    outputs, preds, heights, bust, description = inference('images/test/lin.png', CLASS, epoch=epoch)
    print(outputs, preds, heights, bust)
    #print(CUPredictor())
    #divide_class_dir('./images/train')
    #divide_class_dir('./images/val')
    ''''''