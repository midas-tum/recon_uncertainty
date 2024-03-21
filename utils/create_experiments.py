import numpy as np
import argparse
import shutil

def main(args):

    # check if experiment name is already existing in yaml file
    import copy
    if args.seed_start is not None:
        init_seed = int(args.seed_start)
    else:
        init_seed = 42
    if args.nensembles is not None:
        nensembles = int(args.nensembles)
    else:
        nensembles = 20
    if args.yaml_file is not None:
        yaml_file = args.yaml_file
    else:
        yaml_file = 'config/unet.yml'
    if args.exp_name is not None:
        exp_name = args.exp_name
    else:
        exp_name = 'unet_real2ch_unrolled_sure_e0_acc4'

    print_init = exp_name + ':\n\
        <<: *unet_real2ch_sure\n\
        seed: ' + str(init_seed) + '\n\
        accelerations:\n\
            - 4\n\
        center_fractions:\n\
            - 0.08'

    print_init = print_init.split('\n')

    for idx in range(nensembles):
        print_str = copy.deepcopy(print_init)
        print_str[0] = print_str[0].replace('e0', f'e{idx}')
        print_str[2] = print_str[2].replace('seed: ' + str(init_seed), f'seed: {init_seed+idx}')
        print('\n'.join(print_str))
        print('')

    print_init = 'python3 experiments/train_ensemble.py --config ' + yaml_file + ' --experiment ' + exp_name
    for idx in range(nensembles):
        print_str = copy.deepcopy(print_init)
        print_str = print_str.replace('e0', f'e{idx}')
        print(print_str)
        print('')


if __name__ == '__main__':
    # python3 create_experiments.py --
    parser = argparse.ArgumentParser(description='Create experiment seeds in YAML and SLURM file.')
    parser.add_argument('yaml_file', help='YAML file to modify, e.g. config/cnn.yml.')
    parser.add_argument('exp_name', help='Experiment name with 0 (in e0) being substituted with seed number, e.g. cnn_real2ch_unrolled_e0_acc4')
    parser.add_argument('seed_start', help='Starting seed index')
    parser.add_argument('nensembles', help='Amount of parallel ensembles')

    args = parser.parse()
    main(args)