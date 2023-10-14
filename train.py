import torch
import os, sys, json
from time import strftime
from torch.utils.data import DataLoader
from german_dataloader import GermanTrafficSignDataset
import argparse
import numpy as np
import torch.nn as nn
from torchvision.models import resnet50

'''
classifier를 학습시키는 코드 
'''

# parser 정의 
parser = argparse.ArgumentParser(description="training network", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch", default=8, type=int, dest="batch size")
parser.add_argument("--epochs", default=100, type=int, dest="train epochs")
args = parser.parse_args()

# hyperparameter 정의
batch_size = 8
end_epochs = 100
lr = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"

# logging directory 설정 
timestamp = strftime("%Y-%m-%d_%H-%M")
directory = "results_{}".format(timestamp)
if not os.path.exists(directory):
    os.makedirs(directory)

# dataset 가져오기 
csv_path = os.path.join(os.getcwd(), "german-traffic-sign\Train.csv")
train_dataset = GermanTrafficSignDataset(csv_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# model 불러오기 
model = resnet50(weights="IMAGENET1K_V2")
model.fc = nn.Linear(in_features=2048, out_features=43, bias=True)
model = model.to(device)

# optimizer, loss function 정의하기 
fn_loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

# model 학습시키기 
for epoch in range(end_epochs+1):
    print("Epoch {}".format(epoch))
    loss_arr = []
    for batch, samples in enumerate(train_loader):
        input_x, target = samples
        input_x = input_x.to(device)
        target = target.to(device)
        output = model(input_x)
        optim.zero_grad()
        loss = fn_loss(output, target)
        loss.backward()
        optim.step()
        loss_arr += [loss.item()]

    print("Epoch: {}, Avg_loss: {}".format(epoch, np.mean(loss_arr))) 
    
    if epoch % 10 == 0:
        torch.save(model.state_dict(), directory + '/classifier_' + str(epoch)+'.pt')

