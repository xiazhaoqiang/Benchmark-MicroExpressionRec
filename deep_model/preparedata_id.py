import os, random
import numpy as np
from shutil import copyfile

dbtype = ['smic', 'casme2', 'samm']
# dbmeta_fn = ['smic-indv', 'casme2-crs2db', 'samm-crs2db']
dbmeta_fn = ['smic-3classes', 'casme2-5classes', 'samm-5classes']

def main():
    version = 1 # 0, 1, 2, 4
    verFolder = 'v_{}'.format(version)
    alphas = range(0,1)
    dataDir = os.path.join('../dataset', verFolder)
    if not os.path.exists(dataDir):
        os.makedirs(dataDir)

    for i, dbname in enumerate(dbtype):
        subjectDir = os.path.join(dataDir, dbname)
        if not os.path.exists(subjectDir):
            os.makedirs(subjectDir)
        flowDir = os.path.join('../dataset', 'flow_db', dbname)
        filePath = os.path.join('../dataset', 'benchmark_db', '{}.csv'.format(dbmeta_fn[i]))
        meta_dict = {'subject':[],'filename':[],'emotion':[]}
        with open(filePath,'r') as f:
            for textline in f:
                texts = textline.strip('\n').split(',')
                meta_dict['subject'].append(texts[0])
                meta_dict['filename'].append(texts[1])
                meta_dict['emotion'].append(int(texts[5]))
        subjects = list(set(meta_dict['subject']))
        subjects.sort()
        sampleNum = len(meta_dict['filename'])
        subFilePath = os.path.join(subjectDir, 'subName.txt')
        sub_f = open(subFilePath, 'w')
        for subject in subjects:
            # open the training/val/test list file
            filePath = os.path.join(subjectDir, '{}_train.txt'.format(subject))
            train_f = open(filePath,'w')
            filePath = os.path.join(subjectDir, '{}_test.txt'.format(subject))
            test_f = open(filePath,'w')
            sub_f.write('{}\n'.format(subject))
            # traverse each item
            for alpha in alphas:
                fileDir = os.path.join(subjectDir, 'flow_alpha{}'.format(alpha))
                if not os.path.exists(fileDir):
                    os.makedirs(fileDir)
                for j in range(0,sampleNum):
                    fileName = '{}_{}.png'.format(meta_dict['subject'][j], meta_dict['filename'][j])
                    filePath = os.path.join(fileDir, fileName)
                    filePath_src = os.path.join(flowDir, fileName)
                    if meta_dict['subject'][j] == subject:
                        test_f.write('{} {}\n'.format(filePath,meta_dict['emotion'][j]))
                        copyfile(filePath_src,filePath)
                    else:
                        train_f.write('{} {}\n'.format(filePath,meta_dict['emotion'][j]))
                        copyfile(filePath_src, filePath)
            print('The subject: {}.'.format(subject))
            train_f.close()
            test_f.close()
        sub_f.close()

if __name__ == '__main__':
    main()