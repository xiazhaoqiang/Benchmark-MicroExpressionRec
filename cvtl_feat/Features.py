import os, natsort, fnmatch
import numpy as np
import skimage.feature
from skimage import io, color
import matplotlib.pyplot as plt

class LBP_TOP():
    def __init__(self, blocks=8):
        super(LBP_TOP, self).__init__()
        self.P = 8
        self.R = 1.0
        self.type = 'uniform' # default, ror, uniform, nri_uniform, var
        self.bins = range(0,10,1)
        self.block_size = blocks

    def extractfeat_seq(self,volumeT):
        """
            GETFEAT_SEQ extracts feature vectors from a sequence of images or video frames
            volumeT: H*W*T
        """
        h = volumeT.shape[0] #y
        w = volumeT.shape[1] #x
        t = volumeT.shape[2] #t
        bs = int(self.R)
        # unfold the tensor into three big matrices and calculate the lbp maps
        # note the unfold order: 'C' order
        # YOX plane
        bmat = volumeT.transpose((0, 2, 1)) # YTX
        bmat_yx = bmat.reshape((h,w*t))
        lbp_yx = skimage.feature.local_binary_pattern(bmat_yx, self.P, self.R,method=self.type)
        lbp_yx = lbp_yx.reshape((h,t,w))
        lbp_yx = lbp_yx.transpose((0, 2, 1))
        lbp_yox = lbp_yx[0 + bs:h - bs, 0 + bs:w - bs, 0 + bs:t - bs]
        # TOX plane
        bmat = volumeT.transpose((2,0,1)) # TYX
        bmat_tx = bmat.reshape((t, h * w))
        lbp_tx = skimage.feature.local_binary_pattern(bmat_tx, self.P, self.R, method=self.type)
        lbp_tx = lbp_tx.reshape((t, h, w))
        lbp_tx = lbp_tx.transpose((1,2,0))
        lbp_tox = lbp_tx[0 + bs:h - bs, 0 + bs:w - bs, 0 + bs:t - bs]
        # YOT plane
        bmat = volumeT  # YXT
        bmat_yt = bmat.reshape((h, w * t))
        lbp_yt = skimage.feature.local_binary_pattern(bmat_yt, self.P, self.R, method=self.type)
        lbp_yt = lbp_yt.reshape((h, w, t))
        lbp_yot = lbp_yt[0 + bs:h - bs, 0 + bs:w - bs, 0 + bs:t - bs]
        # plt.imshow(lbp_yot[:,:,0])
        # plt.show()
        # statistics
        lbph_yox = np.histogram(lbp_yox,self.bins, density=True)[0]
        lbph_tox = np.histogram(lbp_tox,self.bins, density=True)[0]
        lbph_yot = np.histogram(lbp_yot,self.bins, density=True)[0]
        return np.hstack((lbph_yox,lbph_tox,lbph_yot))

    def extractfeat_dir(self, rootdir):
        """
            GETFEAT_SEQ extracts feature vectors from an assigned directory
        """
        # emotion_files = fnmatch.filter(os.listdir(rootdir),'*.jpg')+fnmatch.filter(os.listdir(rootdir),'*.png') #only filenames
        emotion_files = []
        for file in os.listdir(rootdir):
            filepath = os.path.join(rootdir, file)
            if os.path.splitext(filepath)[1] == '.jpg':
                emotion_files.append(filepath)
            elif os.path.splitext(filepath)[1] == '.bmp':
                emotion_files.append(filepath)
        natsort.natsorted(emotion_files)  # sort the filenames in a natutrual order
        fileN = len(emotion_files)
        # img_volume = np.array([])
        for j, filepath in enumerate(emotion_files):
            # print(filepath)
            img = color.rgb2gray(io.imread(filepath))
            if j == 0:
                img_volume = np.zeros((img.shape[0], img.shape[1], fileN), dtype=img.dtype)
                img_volume[:, :, j] = img
            else:
                img_volume[:, :, j] = img
        # extract features from image volume
        return self.extractfeat_seq(img_volume)