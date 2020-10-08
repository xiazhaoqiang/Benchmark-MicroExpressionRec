import numpy as np
import torch.utils.data
from PIL import Image

class MEDB_CF():
    """MEDB_CF dataset class deals with the datasets for conventional methods"""

    def __init__(self, imgList):
        self.img_path = []
        self.label = []
        self.dbtype = []
        with open(imgList,'r') as f:
            for textline in f:
                texts= textline.strip('\n').split(' ')
                self.img_path.append(texts[0])
                # notice the index: python starts from 0
                # self.label.append(int(texts[1])-1)
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

class MEDB_DM(torch.utils.data.Dataset):
    """MEDB_DM dataset class deals with the datasets for deep models"""

    def __init__(self, imgList, transform=None):
        self.imgPath = []
        self.label = []
        self.dbtype = []
        with open(imgList,'r') as f:
            for textline in f:
                texts= textline.strip('\n').split(' ')
                self.imgPath.append(texts[0])
                self.label.append(int(texts[1]))
                self.dbtype.append(int(texts[2]))
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open("".join(self.imgPath[idx]),'r').convert('RGB')
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
        return {"data":img, "class_label":self.label[idx], 'db_label':self.dbtype[idx]}

    def __len__(self):
        return len(self.imgPath)