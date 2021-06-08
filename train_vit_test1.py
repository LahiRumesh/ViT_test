import torch
from vit_pytorch import ViT
import cv2
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms   
from torch.optim.lr_scheduler import StepLR 
from linformer import Linformer   
import tqdm
from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 64
epochs = 50
lr = 4e-5
gamma = 0.7 

training_data=np.load("generated_data/training_data.npy",allow_pickle=True)
testing_data=np.load("generated_data/testing_data.npy",allow_pickle=True)

class Classification_DATASET(Dataset):
    def __init__(self,data_set,transform=None):
        self.data_set=data_set
        self.transform=transform

    def __len__(self):
        return(len(self.data_set))

    def __getitem__(self,idx):
        data = self.data_set[idx][0]
        pil_image = Image.fromarray(data)

        if self.transform:
            data=self.transform(pil_image)
            return (data,self.data_set[idx][1])
        else:
            return (data,self.data_set[idx][1])

data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

train_data = Classification_DATASET(training_data,transform=data_transform)
test_data = Classification_DATASET(testing_data,transform=data_transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


model = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)
            val_output = model(data)
            val_loss = criterion(val_output, label)
            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)
    print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")