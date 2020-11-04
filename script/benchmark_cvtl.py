import os, sys, argparse, datetime, random, time
import numpy as np
import torch, sklearn.svm

# the core classes are in a different directory while sharing the same parent directory
sys.path.append('..')
from core import Datasets, Metrics

# public variales and strings
cvtl_method_list = ['lbptop','biwoof']
dbtype = ['smic', 'casme2', 'samm']
dbmeta_fn = ['smic-3classes', 'casme2-5classes', 'samm-5classes']

# arguments for CMD
def arg_process():
    """Define several parameters for the main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='casme2', help='the name of dataset')
    parser.add_argument('--gpuid', default='cuda:0', help='the gpu index for training')
    parser.add_argument('--dataversion', type=int, default=1, help='the version of input data')
    parser.add_argument('--feature', default='lbptop', help='the feature method') # lbptop, biwoof
    parser.add_argument('--classifier', default='svm', help='the classification method')
    parser.add_argument('--C', type=float, default=1.0, help='the classification method')
    args = parser.parse_args()
    return args

def main():
    """
     Goal: process images by file lists, evaluating the datasize with different model size
     Version: 1.1
     """
    now = datetime.datetime.now()
    random.seed(1)
    # setup the hyper-parameters
    args = arg_process()
    runFileName = sys.argv[0].split('.')[0]
    device = 'cpu'
    verFolder = 'v_{}'.format(args.dataversion)
    if args.dataset == 'smic':
        classes = 3
    else:
        classes = 5
    logPath = os.path.join('..','result', 'log_{}_v{}'.format(args.dataset, args.dataversion) + '.txt')

    # obtian the subject information in LOSO
    file_path = os.path.join('..','dataset', verFolder, args.dataset, 'subName.txt')
    subjects = []
    with open(file_path, 'r') as f:
        for textline in f:
            texts = textline.strip('\n')
            subjects.append(texts)
    # predicted and label vectors
    preds_db = {}
    preds_db['all'] = torch.tensor([])
    labels_db = {}
    labels_db['all'] = torch.tensor([])
    # open the log file and begin to record
    log_f = open(logPath, 'a')
    log_f.write('{}\n'.format(now))
    log_f.write('-' * 80 + '\n')
    log_f.write('-' * 80 + '\n')
    log_f.write('Results:\n')
    time_s = time.time()
    for subject in subjects:
        print('Subject Name: {}'.format(subject))
        print('---------------------------')
        # setup a dataloader for training
        img_dir = os.path.join('..', 'dataset', verFolder, args.dataset, '{}_train.txt'.format(subject))
        image_db_train = Datasets.MEDB_CF(imgList=img_dir)
        # Initialize the model
        print('\tCreating convolutional model....')
        if args.classifier == 'svm':
            model_ft = sklearn.svm.SVC(args.C, kernel='linear', tol=0.00001) # rbf,linear
        # Train and evaluate
        fea_db = image_db_train.getitems()
        model_ft.fit(fea_db['data'], np.array(fea_db['class_label']))
        # Test model
        img_dir = os.path.join('..', 'dataset', verFolder, args.dataset, '{}_test.txt'.format(subject))
        image_db_test = Datasets.MEDB_CF(imgList=img_dir)
        fea_db = image_db_test.getitems()
        preds = model_ft.predict(fea_db['data'])
        print(preds)
        # transform the data formatting from numpy to list
        # ten calculate the evaluation metrics
        preds = torch.from_numpy(preds).float()
        y = torch.tensor(fea_db['class_label']).float()
        acc = torch.sum(preds == y).double() / len(preds)
        # saving the subject results
        preds_db['all'] = torch.cat((preds_db['all'], preds), 0)
        labels_db['all'] = torch.cat((labels_db['all'], y), 0)
        # output the results to console
        print('\tSubject {} has the accuracy:{:.4f}\n'.format(subject, acc))
        print('---------------------------\n')
        log_f.write('\tSubject {} has the accuracy:{:.4f}\n'.format(subject, acc))
    # save and print the results
    time_e = time.time()
    hours, rem = divmod(time_e - time_s, 3600)
    miniutes, seconds = divmod(rem, 60)
    # evaluate all data
    eval_acc = Metrics.accuracy()
    eval_f1 = Metrics.f1score()
    acc_w, acc_uw = eval_acc.eval(preds_db['all'], labels_db['all'])
    f1_w, f1_uw = eval_f1.eval(preds_db['all'], labels_db['all'])
    print('\nThe dataset has the ACC and F1:{:.4f} and {:.4f}'.format(acc_w, f1_w))
    print('\nThe dataset has the UAR and UF1:{:.4f} and {:.4f}'.format(acc_uw, f1_uw))
    log_f.write('\nOverall:\n\tthe ACC and F1 of all data are {:.4f} and {:.4f}\n'.format(acc_w, f1_w))
    log_f.write('\tthe UAR and UF1 of all data are {:.4f} and {:.4f}\n'.format(acc_uw, f1_uw))
    # writing parameters into log file
    print('\tNetname:{}, Dataversion:{}.'.format(args.classifier, args.dataversion))
    print('\tElapsed time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(miniutes), seconds))
    log_f.write(
        '\nOverall:\n\tthe weighted and unweighted accuracy of all data are {:.4f} and {:.4f}\n'.format(acc_w, acc_uw))
    log_f.write(
        '\nSetting:\tNetname:{}, Dataversion:{}.\n'.format(args.classifier, args.dataversion))
    log_f.write('-' * 80 + '\n')
    log_f.write('-' * 80 + '\n')
    log_f.write('\n')
    log_f.close()

if __name__ == '__main__':
    main()