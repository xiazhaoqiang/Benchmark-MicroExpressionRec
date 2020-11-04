import os
import numpy as np
import torch.utils.data
from PIL import Image

class MEDB_CF():
    """MEDB_CF dataset class deals with the datasets for conventional features"""

    def __init__(self, imgList):
        self.img_path = []
        self.label = []
        self.dbtype = []
        with open(imgList,'r') as f:
            for textline in f:
                texts= textline.strip('\n').split(' ')
                self.img_path.append(texts[0])
                # notice the index: python starts from 1
                self.label.append(int(texts[1]))

    def getitems(self):
        fileN = len(self.img_path)
        for i, file_path in enumerate(self.img_path):
            feat = np.loadtxt("".join(file_path), delimiter=",")
            if i == 0:
                # shape (n_samples, n_features)
                feat_volume = np.zeros((fileN, feat.shape[0]), dtype=feat.dtype)
                feat_volume[i, :] = feat.T
            else:
                feat_volume[i, :] = feat.T
        return {"data": feat_volume, "class_label": self.label}

class MEDB_GDM(torch.utils.data.Dataset):
    """MEDB_DM dataset class deals with the datasets for deep geometric models"""

    def __init__(self, imgList, transform=None):
        self.imgPath = []
        self.label = []
        with open(imgList,'r') as f:
            for textline in f:
                texts = textline.strip('\n').split(' ')
                self.imgPath.append(texts[0])
                # notice the index: pytorch starts from 0
                self.label.append(int(texts[1])-1)
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open("".join(self.imgPath[idx]),'r').convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return {"data": img, "class_label": self.label[idx]}

    def __len__(self):
        return len(self.imgPath)

class MEDB_ADM(torch.utils.data.Dataset):
    """MEDB_DM dataset class deals with the datasets for deep appearance models"""

    def __init__(self, imgList, transform=None):
        self.img_dir = []
        self.label = []
        with open(imgList,'r') as f:
            for textline in f:
                texts = textline.strip('\n').split(' ')
                self.img_dir.append(texts[0])
                # notice the index: pytorch starts from 0
                self.label.append(int(texts[1])-1)
        self.transform = transform

    def __getitem__(self, idx):
        X = []
        for file in os.listdir(self.img_dir[idx]):
            filepath = os.path.join(self.img_dir[idx], file)
            if os.path.splitext(filepath)[1] == '.jpg':
                # img = Image.open(filepath).convert('RGB')
                img = Image.open(filepath).convert('L')
            elif os.path.splitext(filepath)[1] == '.bmp':
                img = Image.open(filepath).convert('L')
            else:
                raise AssertionError('Not supported image format!')
            if self.transform is not None:
                img = self.transform(img)
            X.append(img.squeeze_(0))
        X = torch.stack(X, dim=0)

        return {"data": X, "class_label": self.label[idx]}

    def __len__(self):
        return len(self.imgPath)