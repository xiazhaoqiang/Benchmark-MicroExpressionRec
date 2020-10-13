import argparse

# pre-define model lists in benchmark
cvtl_method_list = ['biwoof', 'lbptop']
deep_method_list = ['strcn']

def arg_process():
    """Define several parameters for the main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', default='cuda:0', help='the gpu index for training')
    parser.add_argument('--dataversion', type=int, default=33, help='the version of input data')
    parser.add_argument('--modelname', default='strcn', help='the model architecture')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs')
    parser.add_argument('--learningrate', type=float, default=0.0005, help='the learning rate for training')
    parser.add_argument('--batchsize', type=int, default=64, help='the batch size for training')
    parser.add_argument('--featuremap', type=int, default=16, help='the feature map size')
    parser.add_argument('--poolsize', type=int, default=5, help='the average pooling size')
    parser.add_argument('--lossfunction', default='crossentropy', help='the loss functions')
    args = parser.parse_args()
    return args

def main():
    """
    main: the main entrance of a benchmark evaluation for micro-expression recognition
    Version: 1.0
    Date: 2020.3.27
    """
    args = arg_process()
    if args.modelname in cvtl_method_list:
        print('The conventional method is beginning to evaluate...')
    elif args.modelname in deep_method_list:
        print('The deep method is beginning to evaluate...')
    else:
        print('None model can be found in the repository!')

if __name__ == '__main__':
    main()