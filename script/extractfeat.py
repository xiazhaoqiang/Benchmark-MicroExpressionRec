import os, sys, argparse, time
import numpy as np
from skimage import io

# the core classes are in a different directory while sharing the same parent directory
sys.path.append('..')
from core import Features

# public variales and strings
cvtl_method_list = ['lbptop', 'biwoof']
dbtype = ['smic', 'casme2', 'samm']
dbmeta_fn = ['smic-3classes', 'casme2-5classes', 'samm-5classes']

# arguments for CMD
def arg_process():
    """Define several parameters for the main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='lbptop', help='the feature method')
    args = parser.parse_args()
    return args

def main():
    # define some directories
    imgdb_dir = os.path.join('..', 'dataset', 'benchmark_db')
    # build the folder for features
    args = arg_process()
    if args.method == 'lbptop':
        featdb_dir = os.path.join('..', 'dataset', 'lbptop_db')
        feat_extractor = Features.LBP_TOP(P=8, R=1.0, type='uniform', blocks=(4, 4, 2))
        if not os.path.exists(featdb_dir):
            os.makedirs(featdb_dir)
    elif args.method == 'biwoof':
        featdb_dir = os.path.join('..', 'dataset', 'biwoof_db')
        flow_dir = os.path.join('..', 'dataset', 'flow_db')
        feat_extractor = Features.BiWOOF(blocks=(6, 6), img_size=(60, 60))
        if not os.path.exists(featdb_dir):
            os.makedirs(featdb_dir)
        if not os.path.exists(flow_dir):
            os.makedirs(flow_dir)
    else:
        print('No method has been found!\n')
    # dealing with each dataset
    time_s = time.time()
    for i, dbname in enumerate(dbtype):
        print('The {}-th database'.format(i))
        # read the meta file
        metafile_path = os.path.join(imgdb_dir, '{}.csv'.format(dbmeta_fn[i]))
        meta_dict = {'subject':[],'filename':[],'onset':[],'apex':[],'offset':[],'emotion':[]}
        with open(metafile_path,'r') as f:
            for textline in f:
                texts = textline.strip('\n').split(',')
                meta_dict['subject'].append(texts[0])
                meta_dict['filename'].append(texts[1])
                meta_dict['onset'].append(int(texts[2]))
                meta_dict['apex'].append(int(texts[3]))
                meta_dict['offset'].append(int(texts[4]))
                meta_dict['emotion'].append(int(texts[5]))
        # saving folder
        subject_svdir = os.path.join(featdb_dir, dbname)
        if not os.path.exists(subject_svdir):
            os.makedirs(subject_svdir)
        if args.method == 'biwoof':
            flow_svdir = os.path.join(flow_dir, dbname)
            if not os.path.exists(flow_svdir):
                os.makedirs(flow_svdir)
        # dealing with each emotion folder
        for j,_ in enumerate(meta_dict['subject']):
            emotion_folder = os.path.join(imgdb_dir,dbname,meta_dict['subject'][j],meta_dict['filename'][j])
            if args.method == 'lbptop':
                feat_vec = feat_extractor.extractfeat_dir(emotion_folder)
            elif args.method == 'biwoof':
                feat_vec, flow_map = feat_extractor.extractfeat_dir(emotion_folder, meta_dict['onset'][j],
                                                          meta_dict['apex'][j], meta_dict['offset'][j])
                path_sv = os.path.join(flow_svdir,
                                       '{}_{}.png'.format(meta_dict['subject'][j], meta_dict['filename'][j]))
                io.imsave(path_sv,flow_map.astype(np.uint8))
            path_sv = os.path.join(subject_svdir,'{}_{}.csv'.format(meta_dict['subject'][j],meta_dict['filename'][j]))
            np.savetxt(path_sv, feat_vec, delimiter=",", fmt="%f")
            print('\tThe {}-th folder'.format(j))

    time_e = time.time()
    hours, rem = divmod(time_e - time_s, 3600)
    miniutes, seconds = divmod(rem, 60)
    print('\nElapsed time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(miniutes), seconds))

if __name__ == '__main__':
    main()