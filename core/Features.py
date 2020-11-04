import os, natsort, fnmatch
import numpy as np
import skimage.feature
from skimage import io, color
import cv2
import matplotlib.pyplot as plt

class PreProcess():
    def __init__(self, img_size=(224,224), frm_size=10):
        super(PreProcess, self).__init__()
        self.img_size = img_size
        self.frm_size = frm_size

    def normalize_temporal(self, volumeT):
        """
            NORMALIZE_TEMPORAL temporally normalizes the gray images
            volumeT: H*W*T
        """
        h = volumeT.shape[0] #y, rows
        w = volumeT.shape[1] #x, cols
        t = volumeT.shape[2] #t, temps
        return volumeT

    def process_dir(self, rootdir):
        emotion_files = []
        for file in os.listdir(rootdir):
            filepath = os.path.join(rootdir, file)
            if os.path.splitext(filepath)[1] == '.jpg':
                emotion_files.append(filepath)
            elif os.path.splitext(filepath)[1] == '.bmp':
                emotion_files.append(filepath)
            else:
                raise AssertionError('Not supported image format!')
        emotion_files = natsort.natsorted(emotion_files)  # sort the file names in a natural order
        fileN = len(emotion_files)
        for j, filepath in enumerate(emotion_files):
            # print(filepath)
            # img = color.rgb2gray(io.imread(filepath))
            img = io.imread(filepath, as_gray=True)
            if j == 0:
                img_volume = np.zeros((img.shape[0], img.shape[1], fileN), dtype=img.dtype)
                img_volume[:, :, j] = img
            else:
                img_volume[:, :, j] = img
        return self.normalize_temporal(img_volume)


class LBP_TOP():
    def __init__(self, P=8, R=1.0, type='uniform', blocks=(4,4,2)):
        super(LBP_TOP, self).__init__()
        self.P = P
        self.R = R
        self.type = type # default, ror, uniform, nri_uniform, var
        self.blocks = blocks

    def extractfeat_seq(self,volumeT):
        """
            EXTRACTFEAT_SEQ extracts feature vectors from a sequence of images or video frames
            volumeT: H*W*T
        """
        h = volumeT.shape[0] #y, rows
        w = volumeT.shape[1] #x, cols
        t = volumeT.shape[2] #t, temps
        bs = int(self.R) # boundary size
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
        # codes for debugging
        # plt.imshow(lbp_yot[:,:,0])
        # plt.show()
        # statistics
        if self.type == 'uniform':
            bins = range(0, 60, 1)
            num_bin = 59
        else:
            bins = range(0,256,1)
            num_bin = 255
        # transverse each cuboid
        h = lbp_yot.shape[0]
        w = lbp_yot.shape[1]
        t = lbp_yot.shape[2]
        rows_b = self.blocks[0]
        cols_b = self.blocks[1]
        tems_b = self.blocks[2]
        blocks_h = np.floor(h / self.blocks[0]).astype(int)
        blocks_w = np.floor(w / self.blocks[1]).astype(int)
        blocks_t = np.floor(t / self.blocks[2]).astype(int)
        num_blocks = rows_b*cols_b*tems_b # block number in one sequence
        lbph_yox = np.zeros(num_blocks * num_bin)
        lbph_tox = np.zeros(num_blocks * num_bin)
        lbph_yot = np.zeros(num_blocks * num_bin)
        num_elemnts = blocks_h*blocks_w*blocks_t # element number in one block
        for i in range(0,rows_b):
            for j in range(0,cols_b):
                for k in range(0,tems_b):
                    lbph_yox[(i*cols_b*tems_b+j*tems_b+k)*num_bin:(i*cols_b*tems_b+j*tems_b+k+1)*num_bin] = \
                        np.histogram(lbp_yox[i*blocks_h:(i+1)*blocks_h, j*blocks_w:(j+1)*blocks_w,k*blocks_t:(k+1)*blocks_t].reshape((num_elemnts, 1)), bins, density=True)[0]
                    lbph_tox[(i*cols_b*tems_b+j*tems_b+k)*num_bin:(i*cols_b*tems_b+j*tems_b+k+1)*num_bin] = \
                        np.histogram(lbp_tox[i*blocks_h:(i+1)*blocks_h, j*blocks_w:(j+1)*blocks_w,k*blocks_t:(k+1)*blocks_t].reshape((num_elemnts, 1)), bins, density=True)[0]
                    lbph_yot[(i*cols_b*tems_b+j*tems_b+k)*num_bin:(i*cols_b*tems_b+j*tems_b+k+1)*num_bin] = \
                        np.histogram(lbp_yot[i*blocks_h:(i+1)*blocks_h, j*blocks_w:(j+1)*blocks_w,k*blocks_t:(k+1)*blocks_t].reshape((num_elemnts, 1)), bins, density=True)[0]
        return np.hstack((lbph_yox,lbph_tox,lbph_yot))
        # counting with no blocks
        # lbph_yox = np.histogram(lbp_yox, bins, density=True)[0]
        # lbph_tox = np.histogram(lbp_tox, bins, density=True)[0]
        # lbph_yot = np.histogram(lbp_yot, bins, density=True)[0]
        # return np.hstack((lbph_yox,lbph_tox,lbph_yot))

    def extractfeat_dir(self, rootdir):
        """
            EXTRACTFEAT_DIR extracts feature vectors from an assigned directory
        """
        # emotion_files = fnmatch.filter(os.listdir(rootdir),'*.jpg')+fnmatch.filter(os.listdir(rootdir),'*.png') #only filenames
        emotion_files = []
        for file in os.listdir(rootdir):
            filepath = os.path.join(rootdir, file)
            if os.path.splitext(filepath)[1] == '.jpg':
                emotion_files.append(filepath)
            elif os.path.splitext(filepath)[1] == '.bmp':
                emotion_files.append(filepath)
            else:
                raise AssertionError('Not supported image format!')
        emotion_files = natsort.natsorted(emotion_files)  # sort the filenames in a natutrual order
        fileN = len(emotion_files)
        # img_volume = np.array([])
        for j, filepath in enumerate(emotion_files):
            # print(filepath)
            # img = color.rgb2gray(io.imread(filepath))
            img = io.imread(filepath, as_gray=True)
            if j == 0:
                img_volume = np.zeros((img.shape[0], img.shape[1], fileN), dtype=img.dtype)
                img_volume[:, :, j] = img
            else:
                img_volume[:, :, j] = img
        # extract features from image volume
        return self.extractfeat_seq(img_volume)


class BiWOOF():
    def __init__(self, blocks=(5, 5), bins=8, img_size=(100,100)):
        super(BiWOOF, self).__init__()
        self.bins = bins
        self.blocks = blocks
        self.img_size = img_size
        self.flow_extractor = cv2.createOptFlow_DualTVL1()

    def extractflow(self, img_f, img_s):
        """
            EXTRACTFLOW extracts flow maps from a sequence of images or video frames by onset and apex frames
        """
        img_rows = self.img_size[0]
        img_cols = self.img_size[1]
        flowmap = self.flow_extractor.calc(img_f,img_s,None)
        uGx = cv2.Sobel(flowmap[:,:,0], -1, 1, 0, ksize=3) # x-axis
        uGy = cv2.Sobel(flowmap[:,:,0], -1, 0, 1, ksize=3) # y-axis
        vGx = cv2.Sobel(flowmap[:,:,1], -1, 1, 0, ksize=3) # x-axis
        vGy = cv2.Sobel(flowmap[:,:,1], -1, 0, 1, ksize=3) # y-axis
        flow_strain = np.sqrt(np.power(uGx,2)+np.power(vGy,2)+0.5*np.power(uGy+vGx,2))
        flow_strain = np.reshape(flow_strain,(flow_strain.shape[0],flow_strain.shape[1],1))
        flowmap_e = np.concatenate((flowmap,flow_strain),axis=2)
        flowmap_e = cv2.resize(flowmap_e, (img_rows, img_cols))
        min_value = np.min(flowmap_e, axis=(0,1))
        min_value = np.tile(min_value,(flowmap_e.shape[0],flowmap_e.shape[1],1))
        max_value = np.max(flowmap_e, axis=(0, 1))
        max_value = np.tile(max_value, (flowmap_e.shape[0], flowmap_e.shape[1], 1))
        flowmap_n = np.divide(flowmap_e - min_value, np.maximum(max_value - min_value, 1e-8))
        return flowmap_e, flowmap_n

    def extractfeat(self,flowmap):
        """
            EXTRACTFEAT extracts feature vectors from a single flow map
            flowmap: H*W*3
        """
        # calculating the characteristics of flow maps
        rows = flowmap.shape[0] #y
        cols = flowmap.shape[1] #x
        # channels = flowmap.shape[2] #c
        rows_b = self.blocks[0] # the number of blocks in rows
        cols_b = self.blocks[1] # the number of blocks in cols
        bs_r = np.floor(rows/rows_b).astype(int)
        bs_c = np.floor(cols/cols_b).astype(int)
        bins = np.arange(-np.pi,np.pi+0.01,(np.pi*2)/self.bins, dtype=flowmap.dtype)
        hist = np.zeros((rows_b*cols_b*self.bins,1),dtype=flowmap.dtype)
        # transform the UV to MO space
        mag = np.sqrt(np.power(flowmap[:,:,0],2) + np.power(flowmap[:,:,1],2))
        ort = np.arctan2(flowmap[:,:,1],flowmap[:,:,0]) # the output of arctan2 is [-pi, pi]
        stn = flowmap[:,:,2]
        for i in range(0,rows_b):
            for j in range(0,cols_b):
                mag_v = mag[i * bs_r:(i + 1) * bs_r, j * bs_c:(j + 1) * bs_c].reshape((bs_r * bs_c, 1))
                ort_v = ort[i * bs_r:(i + 1) * bs_r, j * bs_c:(j + 1) * bs_c].reshape((bs_r * bs_c, 1))
                stn_v = stn[i * bs_r:(i + 1) * bs_r, j * bs_c:(j + 1) * bs_c].reshape((bs_r * bs_c, 1))
                hist_b = np.histogram(ort_v,bins,weights=mag_v,density=True)[0] # output 1D array: (d), rather than (d,1)
                hist[(i*cols_b+j)*self.bins:(i*cols_b+j+1)*self.bins] = np.sum(stn_v)*np.reshape(hist_b,(hist_b.shape[0],1))
        return hist

    def extractfeat_dir(self, rootdir, onset, apex, offset):
        """
            EXTRACTFEAT_DIR extracts feature vectors from an assigned directory
        """
        emotion_files = []
        for file in os.listdir(rootdir):
            filepath = os.path.join(rootdir, file)
            if os.path.splitext(filepath)[1] == '.jpg':
                emotion_files.append(filepath)
            elif os.path.splitext(filepath)[1] == '.bmp':
                emotion_files.append(filepath)
            else:
                raise AssertionError('Not supported image format!')
        emotion_files = natsort.natsorted(emotion_files)  # sort the file names in a natural order
        # Note: file index starting from 1 in the original labels
        #       image was input as float64 formatting, needs to be transformed
        #       color.rgb2gray is not used as the behavior of rgb2gray will change in scikit-image 0.19
        img_f = io.imread(emotion_files[0], as_gray = True)
        img_f = img_f.astype(np.float32)
        img_s = io.imread(emotion_files[apex-onset], as_gray = True)
        img_s = img_s.astype(np.float32)
        # extract flow maps from images
        flowmap, flowmap_n = self.extractflow(img_f, img_s)
        # extract features from flow maps
        hist = self.extractfeat(flowmap)
        return hist, flowmap_n*255