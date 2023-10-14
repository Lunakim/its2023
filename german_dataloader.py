import torch
import os
import pandas as pd 
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision import transforms
from PIL import Image

class GermanTrafficSignDataset(Dataset):
    def __init__(self, csv_path):
        self.data_df = pd.read_csv(csv_path)
        self.data_path = os.path.join(os.getcwd(), 'german-traffic-sign')
        self.data_paths = self.data_df['Path']
        self.labels = self.data_df['ClassId']
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        # 이미지 로드 
        end_path = self.data_paths[idx].replace('/', '\\')
        path = os.path.join(self.data_path, end_path)
        data = Image.open(path)
        data = self.transform(data) 
        # 라벨 로드
        label = self.labels[idx] # class 0~42
        label = torch.tensor(label)

        return data, label 


if __name__=="__main__":
    csv_path = os.path.join(os.getcwd(), "german-traffic-sign\Train.csv")

    dataset_all = GermanTrafficSignDataset(csv_path)
    data, label = dataset_all.__getitem__(0)




