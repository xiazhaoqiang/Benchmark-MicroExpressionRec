import os, sys, datetime, time, random, argparse

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import transforms

from deep_model.Datasets import MEGC2019 as MEGC2019
# from Datasets import MEGC2019_SI as MEGC2019
from deep_model import LossFunctions, MERNets, CNNs


def arg_process():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='samm', help='the name of dataset')
    parser.add_argument('--dataversion', type=int, default=1, help='the version of input data')
    parser.add_argument('--epochs', type=int, default=1, help='the number of training epochs')
    parser.add_argument('--gpuid', default='cuda:0', help='the gpu index for training')
    parser.add_argument('--learningrate', type=float, default=0.0005, help='the learning rate for training')
    parser.add_argument('--modelname', default='menet_a', help='the model architecture')
    parser.add_argument('--modelversion', type=int, default=3, help='the version of created model')
    parser.add_argument('--batchsize', type=int, default=64, help='the batch size for training')
    parser.add_argument('--featuremap', type=int, default=16, help='the feature map size')
    parser.add_argument('--poolsize', type=int, default=5, help='the average pooling size')
    parser.add_argument('--lossfunction', default='crossentropy', help='the loss functions')
    args = parser.parse_args()
    return args

def train_model(model, dataloaders, criterion, optimizer, device='cpu', num_epochs=25):
    since = time.time()
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    for epoch in range(num_epochs):
        print('\tEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('\t'+'-' * 10)
        # Each epoch has a training
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        # Iterate over data
        for j, samples in enumerate(dataloaders):
            inputs, class_labels = samples["data"], samples["class_label"]
            inputs = torch.FloatTensor(inputs).to(device)
            class_labels = class_labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward to get model outputs and calculate loss
            output_class = model(inputs)
            loss = criterion(output_class, class_labels)
            # backward
            loss.backward()
            optimizer.step()
            # statistics
            _, predicted = torch.max(output_class.data,1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted == class_labels)

        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)
        print('\t{} Loss: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('\tTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model

def test_model(model, dataloaders, device):

    model.eval()
    num_samples = len(dataloaders.dataset)
    predVec = torch.zeros(num_samples)
    labelVec = torch.zeros(num_samples)
    start_idx = 0
    end_idx = 0
    for j, samples in enumerate(dataloaders):
        inputs, class_labels = samples['data'], samples['class_label']
        inputs = torch.FloatTensor(inputs).to(device)
        # update the index of ending point
        end_idx = start_idx + inputs.shape[0]
        output_class = model(inputs)
        _, predicted = torch.max(output_class.data, 1)
        predVec[start_idx:end_idx] = predicted
        labelVec[start_idx:end_idx] = class_labels
        # update the starting point
        start_idx += inputs.shape[0]
    return predVec, labelVec

def main():
    """
    Goal: process images by file lists, evaluating the datasize with different model size
    Version: 5.0
    """
    print('PyTorch Version: ', torch.__version__)
    print('Torchvision Version: ', torchvision.__version__)
    now = datetime.datetime.now()
    random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = arg_process()
    runFileName = sys.argv[0].split('.')[0]
    # need to modify according to the enviroment
    db_name = args.dataset
    version = args.dataversion
    gpuid = args.gpuid
    model_name = args.modelname
    num_epochs = args.epochs
    lr = args.learningrate
    batch_size = args.batchsize
    model_version = args.modelversion
    feature_map = args.featuremap
    loss_function = args.lossfunction
    pool_size = args.poolsize
    if db_name == 'smic':
        classes = 3
    else:
        classes = 5

    # logPath = os.path.join('result', model_name+'_log.txt')
    logPath = os.path.join('../result', 'log_{}_v{}'.format(db_name, args.dataversion) + '.txt')
    # logPath = os.path.join('result', runFileName+'_log_'+'v{}'.format(version)+'.txt')
    # resultPath = os.path.join('result', 'result_'+'v{}'.format(version)+'.pt')
    data_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
    # move to GPU
    print(gpuid)
    device = torch.device(gpuid if torch.cuda.is_available() else 'cpu')
    # obtian the subject information in LOSO
    verFolder = 'v_{}'.format(version)
    filePath = os.path.join('../dataset', verFolder, db_name, 'subName.txt')
    subjects = []
    with open(filePath, 'r') as f:
        for textline in f:
            texts = textline.strip('\n')
            subjects.append(texts)
    # predicted and label vectors
    preds_db = {}
    preds_db['all'] = torch.tensor([])
    labels_db = {}
    labels_db['all'] = torch.tensor([])
    # open the log file and begin to record
    log_f = open(logPath,'a')
    log_f.write('{}\n'.format(now))
    log_f.write('-' * 80 + '\n')
    log_f.write('-' * 80 + '\n')
    log_f.write('Results:\n')
    time_s = time.time()
    for subject in subjects:
        print('Subject Name: {}'.format(subject))
        print('---------------------------')
        # random.seed(1)
        # setup a dataloader for training
        imgDir = os.path.join('../dataset', verFolder, db_name, '{}_train.txt'.format(subject))
        image_db_train = MEGC2019(imgList=imgDir,transform=data_transforms)
        dataloader_train = torch.utils.data.DataLoader(image_db_train, batch_size=batch_size, shuffle=True, num_workers=1)
        # Initialize the model
        print('\tCreating deep model....')
        if model_name == 'menet_a':
            model_ft = MERNets.MeNet_A(num_input=3, featuremaps=feature_map, num_classes=classes, num_layers=1, pool_size=pool_size, model_version=model_version)
        elif model_name == 'menet_d':
            model_ft = MERNets.MeNet_D(num_input=3, featuremaps=feature_map, num_classes=classes, num_layers=1, pool_size=pool_size)
        elif model_name == 'menet_w':
            model_ft = MERNets.MeNet_W(num_input=3, featuremaps=feature_map, num_classes=classes, num_layers=1, pool_size=pool_size)
        elif model_name == 'menet_h':
            model_ft = MERNets.MeNet_H(num_input=3, featuremaps=feature_map, num_classes=classes, num_layers=1, pool_size=pool_size)
        elif model_name == 'menet_c':
            model_ft = MERNets.MeNet_C(num_input=3, featuremaps=feature_map, num_classes=classes, num_layers=1, pool_size=pool_size)
        elif model_name == 'menet_e':
            model_ft = MERNets.MeNet_E(num_input=3, featuremaps=feature_map, num_classes=classes, num_layers=1, pool_size=pool_size)
        elif model_name == 'menet_r':
            model_ft = MERNets.MeNet_R(num_input=3, featuremaps=feature_map, num_classes=classes, num_layers=1, pool_size=pool_size)
        elif model_name == 'shallownet':
            model_ft = CNNs.ShallowNet()
        elif model_name == 'dualinception':
            model_ft = CNNs.DualInceptionNet(num_input=3, num_classes=classes)
        params_to_update = model_ft.parameters()
        optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=0.9)
        # optimizer_ft = optim.Adam(params_to_update, lr=lr)
        if loss_function == 'crossentropy':
            criterion = nn.CrossEntropyLoss()
        elif loss_function == 'focal':
            criterion = LossFunctions.FocalLoss(class_num=classes, device=device)
        elif loss_function == 'balanced':
            criterion = LossFunctions.BalancedLoss(class_num=classes, device=device)
        elif loss_function == 'cosine':
            criterion = LossFunctions.CosineLoss()
        model_ft = model_ft.to(device) # from cpu to gpu
        # Train and evaluate
        model_ft = train_model(model_ft, dataloader_train, criterion, optimizer_ft, device, num_epochs=num_epochs)
        # torch.save(model_ft, os.path.join('data', 'model_s{}.pth').format(subject))

        # Test model
        imgDir = os.path.join('../dataset', verFolder, db_name, '{}_test.txt'.format(subject))
        image_db_test = MEGC2019(imgList=imgDir,transform=data_transforms)
        dataloaders_test = torch.utils.data.DataLoader(image_db_test, batch_size=batch_size, shuffle=False,
                                                       num_workers=1)

        preds, labels = test_model(model_ft, dataloaders_test, device)
        acc = torch.sum(preds == labels).double()/len(preds)
        print('\tSubject {} has the accuracy:{:.4f}\n'.format(subject,acc))
        print('---------------------------\n')
        log_f.write('\tSubject {} has the accuracy:{:.4f}\n'.format(subject,acc))

        # saving the subject results
        preds_db['all'] = torch.cat((preds_db['all'], preds), 0)
        labels_db['all'] = torch.cat((labels_db['all'], labels), 0)
    time_e = time.time()
    hours, rem = divmod(time_e-time_s,3600)
    miniutes, seconds = divmod(rem,60)
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
    print('\tNetname:{}, Dataversion:{}\n\tLearning rate:{}, Epochs:{}, Batchsize:{}.'.format(model_name,version,lr,num_epochs,batch_size))
    print('\tElapsed time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(miniutes),seconds))
    log_f.write('\nOverall:\n\tthe weighted and unweighted accuracy of all data are {:.4f} and {:.4f}\n'.format(acc_w,acc_uw))
    log_f.write('\nSetting:\tNetname:{}, Dataversion:{}\n\tLearning rate:{}, Epochs:{}, Batchsize:{}.\n'.format(model_name,
                                                                                                        version,
                                                                                                        lr,
                                                                                                        num_epochs,
                                                                                                        batch_size))
    # # save subject's results
    # torch.save({
    #     'predicts':preds_db,
    #     'labels':labels_db,
    #     'weight_acc':acc_w,
    #     'unweight_acc':acc_uw
    # },resultPath)
    log_f.write('-' * 80 + '\n')
    log_f.write('-' * 80 + '\n')
    log_f.write('\n')
    log_f.close()

if __name__ == '__main__':
    main()