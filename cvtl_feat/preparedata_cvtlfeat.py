import os, argparse

# public variales and strings
cvtl_method_list = ['biwoof', 'lbptop']

# arguments for CMD
def arg_process():
    """Define several parameters for the main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', default='cuda:0', help='the gpu index for training')
    parser.add_argument('--dataversion', type=int, default=33, help='the version of input data')
    parser.add_argument('--method', default='lbp_top', help='the feature method')
    args = parser.parse_args()
    return args

def main():
    # define some directories
    imgdb_dir = os.path.join('..', 'dataset', 'benchmark_db')
    # build the folder for features
    args = arg_process()
    if args.method == 'lbptop':
        featdb_dir = os.path.join('..','dataset', 'lbptop_db')
        if not os.path.exists(featdb_dir):
            os.makedirs(featdb_dir)
    elif args.method == 'biwoof':
        featdb_dir = os.path.join('..', 'dataset', 'biwoof_db')
        if not os.path.exists(featdb_dir):
            os.makedirs(featdb_dir)

if __name__ == '__main__':
    main()