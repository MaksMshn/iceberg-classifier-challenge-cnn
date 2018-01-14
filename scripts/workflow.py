import datetime as dt
import json, time
import numpy as np
import random
from random import choice
import glob
import sys, os
import argparse

from utils import create_dataset
from cnn_train import train, evaluate
import models


def parase_arguments(argv):
    parser = argparse.ArgumentParser(description="""ML workflow controller
            """)
    parser.add_argument(
        '-c',
        '--config',
        metavar='config.json',
        help="""Workflow configuration file. Must be in json format. If none,
        will use a default configuration.""")
    parser.add_argument(
        '-i',
        '--iters',
        help="""Generate i random configurations using get_randomish function
        and train them. Ignores options such as load or config.""")
    parser.add_argument(
        '--evaluate',
        help="""Do not train, just do a single evaluation run.""",
        action='store_true')
    parser.add_argument(
        '-l',
        '--load',
        help="""Load specified model instead of using built ins.""")
    parser.add_argument(
        '-n', '--name', help="""Set model name.""")
    # some useful model configurations
    parser.add_argument('--activation', help="""Activatio name.""")
    parser.add_argument('--epochs', help="""Number of epochs.""", type=int)
    parser.add_argument('--model_fn', help="""Specify type of model.""")
    parser.add_argument('--meta', help="""Use metadata.""", action='store_true')
    parser.add_argument(
        '--pseudo', help="""Use pseudo labeling.""", action='store_true')
    return parser.parse_args(argv)


####  Hyperparameters ####


def gen_randomish_config3(name=None):
    name = name or 'deep_randomish_config'
    if random.random() < -0.25:  #never
        configs = glob.glob('../config/config*json')
        config = choice(configs)
        with open(config) as f:
            config = json.load(f)
        config['pseudo_train'] = True
        prev_tmp = config['tmp']
        prev_name = config['name']
        prev_weights = "../weights/weights_{}_{}.hdf5".format(
            prev_name, prev_tmp)
        if os.path.isfile(prev_weights):
            config['model_fn'] = prev_weights
            return config
        else:
            return None
    else:
        shift_ranges = 0.4 + random.uniform(-1, 1) * .2
        channels = choice([2, 3,3])
        if channels == 2:
            preproc_strat = 'band2'
        else:
            preproc_strat = 'band3'
        config = {
            'name':
                name,
            # training
            'lr':
                np.random.lognormal(sigma=0.5) * 1.0e-5,
            'decay':
                np.random.lognormal() * 3.4670665307866379e-06,
            'relu_type':
                choice(['selu']),
            'epochs':
                choice([200, 300, 500]),
            'full_cycls_per_epoch':
                8,
            'batch_size':
                choice([64]),
            'lr_patience':
                choice([50, 100]),
            'stop_patience':
                choice([25, 50, 100, 300]),
            # data augmentation
            'hflip_prob':
                .1 + random.uniform(-1, 1) * .28,
            'vflip_prob':
                .5 + random.uniform(-1, 1) * .28,
            'rot90_prob':
                0.73007147223015134 + random.uniform(-1, 1) * .14,
            'rot_prob':
                0.39286442681668626 + random.uniform(-1, 1) * .28,
            'rotate_rg':
                30.352924041777015 + random.uniform(-1, 1) * 14,
            'shift_prob':
                0.45233683486776546 + random.uniform(-1, 1) * .14,
            'shift_width_rg':
                shift_ranges,
            'shift_height_rg':
                shift_ranges,
            'zoom_prob':
                0.43867510928667224 + random.uniform(-1, 1) * .14,
            'zoom_rg': (1 - random.random() / 2, 1 + random.random()),
            'noise_prob':
                0.1338943735248288 + random.uniform(-1, 1) * .1,
            'noise_rg':
                np.random.lognormal(sigma=.2) * 0.019821861298954285,
            # model
            'use_meta':
                False,
            'model_fn':
                'model2_meta',
            # preprocessing
            'preproc_strat':
                preproc_strat,
            'channels':
                channels,
            'inc_angle_fill':
                40,
            'band3_op':
                choice([
                    'lambda x1, x2: x1-x2', 'lambda x1, x2: x1+x2',
                    'lambda x1, x2: x1*x2'
                ]),
            'soft_targets':
                False,
            'soft_val':
                0.99,  # only if soft_targets = True, must be 0.5 < x <= 1.0
            # pseudo train
            'pseudo_train':
                False,
            # deep model config
            'depth':
                choice(list(range(6)))
        }
        if 'meta' in config['model_fn']:
            config['use_meta'] = True
        return config


def gen_randomish_config2(name=None):
    name = name or 'optimized_m1_randomish_config'
    if random.random() < -0.25: #never
        configs = glob.glob('../config/config*json')
        config = choice(configs)
        with open(config) as f:
            config = json.load(f)
        config['pseudo_train'] = True
        prev_tmp = config['tmp']
        prev_name = config['name']
        prev_weights = "../weights/weights_{}_{}.hdf5".format(
            prev_name, prev_tmp)
        if os.path.isfile(prev_weights):
            config['model_fn'] = prev_weights
            return config
        else:
            return None
    else:
        shift_ranges = 0.3 + random.uniform(-1, 1) * .2
        channels = choice([2, 2, 3])
        if channels == 2:
            preproc_strat = 'band2'
        else:
            preproc_strat = 'band3'
        config = {
            'name':
                name,
            # training
            'lr':
                np.random.lognormal(sigma=0.5) * 0.00027790185561569629,
            'decay':
                np.random.lognormal() * 3.4670665307866379e-06,
            'relu_type':
                choice(['selu', 'elu']),
            'epochs':
                choice([30, 50, 75, 100]),
            'full_cycls_per_epoch':
                8,
            'batch_size':
                choice([32, 64]),
            'lr_patience':
                choice([50, 100]),
            'stop_patience':
                choice([15, 20, 25]),
            # data augmentation
            'hflip_prob':
                .1 + random.uniform(-1, 1) * .28,
            'vflip_prob':
                .5 + random.uniform(-1, 1) * .28,
            'rot90_prob':
                0.73007147223015134 + random.uniform(-1, 1) * .14,
            'rot_prob':
                0.39286442681668626 + random.uniform(-1, 1) * .28,
            'rotate_rg':
                15.352924041777015 + random.uniform(-1, 1) * 14,
            'shift_prob':
                0.045233683486776546 + random.uniform(-1, 1) * .14,
            'shift_width_rg':
                shift_ranges,
            'shift_height_rg':
                shift_ranges,
            'zoom_prob':
                0.23867510928667224 + random.uniform(-1, 1) * .14,
            'zoom_rg': (1 - random.random() / 2, 1 + random.random()),
            'noise_prob':
                0.6338943735248288 + random.uniform(-1, 1) * .16,
            'noise_rg':
                np.random.lognormal(sigma=.2) * 0.019821861298954285,
            # model
            'use_meta':
                False,
            'model_fn':
                choice([
                    'model1_deeper_meta',
                    'model1_deeper',
                ]),
            # preprocessing
            'preproc_strat':
                preproc_strat,
            'channels':
                channels,
            'inc_angle_fill':
                40,
            'band3_op':
                choice([
                    'lambda x1, x2: x1-x2', 'lambda x1, x2: x1+x2',
                    'lambda x1, x2: x1*x2'
                ]),
            'soft_targets':
                False,
            'soft_val':
                0.99,  # only if soft_targets = True, must be 0.5 < x <= 1.0
            # pseudo train
            'pseudo_train':
                False,
            # deep model config
            'depth':
                choice(list(range(8))),
            'alpha_drop':
                choice([True, False])
        }
        if 'meta' in config['model_fn']:
            config['use_meta'] = True
        return config


def gen_randomish_config(name=None):
    name = name or 'continue training prev model'
    if random.random() < 1.4:  # always
        configs = glob.glob('../config/config*json')
        config = choice(configs)
        with open(config) as f:
            config = json.load(f)
        config['pseudo_train'] = True
        prev_tmp = config['tmp']
        prev_name = config['name']
        prev_weights = "../weights/weights_{}_{}.hdf5".format(
            prev_name, prev_tmp)
        if os.path.isfile(prev_weights):
            config['model_fn'] = prev_weights
            return config
        else:
            return None
    else:
        shift_ranges = random.uniform(-1,1)*.2 + .3
        channels = choice([2, 3, 3])
        if channels == 2:
            preproc_strat = 'band2'
        else:
            preproc_strat = 'band3'
        config = {
            'name':
                name,
            # training
            'lr':
                np.random.lognormal(sigma=0.5) * 1e-4,
            'decay':
                np.random.lognormal() * 1e-6,
            'relu_type':
                choice(['selu', 'relu', 'elu', 'elu']),
            'epochs':
                choice([30, 40, 50]),
            'full_cycls_per_epoch':
                8,
            'batch_size':
                choice([16, 32, 64]),
            'lr_patience':
                choice([5, 10]),
            'stop_patience':
                choice([5, 7, 10]),
            # data augmentation
            'hflip_prob':
                random.random(),
            'vflip_prob':
                random.random(),
            'rot90_prob':
                random.random() / 2 + 0.5,
            'rot_prob':
                random.random() / 2,
            'rotate_rg':
                random.random() * 45,
            'shift_prob':
                random.random() / 2,
            'shift_width_rg':
                shift_ranges,
            'shift_height_rg':
                shift_ranges,
            'zoom_prob':
                random.random() / 2,
            'zoom_rg': (1 - random.random(), 1 + random.random()),
            'noise_prob':
                random.random() / 2,
            'noise_rg':
                np.random.lognormal(sigma=.5) * .02,
            # model
            'use_meta':
                False,
            'model_fn':
                choice([
                    'model0', 'model0', 'model1_meta', 'model1_deeper',
                    'model2_meta'
                ]),
            # preprocessing
            'preproc_strat':
                preproc_strat,
            'channels':
                channels,
            'inc_angle_fill':
                40,
            'band3_op':
                choice([
                    'lambda x1, x2: x1-x2', 'lambda x1, x2: x1+x2',
                    'lambda x1, x2: x1*x2'
                ]),
            'soft_targets':
                False,
            'soft_val':
                0.99,  # only if soft_targets = True, must be 0.5 < x <= 1.0
            # pseudo train
            'pseudo_train':
                False
        }
        if 'meta' in config['model_fn']:
            config['use_meta'] = True
        return config


def get_default_config(cor_zoom=True):
    if cor_zoom:
        zoom_rg = (.6, 1.4)
    else:
        zoom_rg = (.1, .1)
    return {
        'name': 'default',
        'output_name': 'default_out.json',
        # training
        'lr': 8e-5,
        'decay': 1e-6,
        'relu_type': 'elu',
        'epochs': 250,
        'full_cycls_per_epoch': 8,
        'batch_size': 32,
        'lr_patience': 10,
        'stop_patience': 60,
        # data augmentation
        'hflip_prob': .5,
        'vflip_prob': .5,
        'rot90_prob': .999,
        'rot_prob': .2,
        'rotate_rg': 10,
        'shift_prob': .1,
        'shift_width_rg': .1,
        'shift_height_rg': .1,
        'zoom_prob': .2,
        'zoom_rg': zoom_rg,
        'noise_prob': .3,
        'noise_rg': .02,
        # model
        'use_meta': False,
        'model_fn': 'model0',
        # preprocessing
        'preproc_strat': 'band3',
        'inc_angle_fill': 40,
        'band3_op': 'lambda x1, x2: (x1+x2)/2',
        'soft_targets': False,
        'soft_val': 0.99,  # only if soft_targets = True, must be 0.5 < x <= 1.0
        # pseudo
        'pseudo_train': False,
        'pseudo_retrain_freq': 8,
        'pseudo_type': 'soft',
        'pseudo_prop': .4,
        'pseudo_clip_val': .99
    }


def runtime(start):
    end_time = time.time()
    sec_elapsed = end_time - start
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def single_run(config, training=True):
    start = time.time()

    if config['model_fn'].endswith('hdf5'):
        # using preset config
        model = models.use_saved_model(config['model_fn'], **config)
    else:
        model_fn = getattr(models, config['model_fn'])
        model = model_fn(**config)
    model.summary()
    print('Model compiled after {}'.format(runtime(start)))

    tmp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_%f")
    config['tmp'] = tmp
    config_name = '../config/config_{}_{}.json'.format(tmp, config['name'])
    output_name = '../output/out_{}_{}.json'.format(tmp, config['name'])
    config['model_w_name'] = "../weights/weights_{}_{}.hdf5".format(
        config['name'], tmp)
    config['output_name'] = output_name

    print('Saving configuration file to: {}'.format(config_name), flush=True)
    print('and output file to: {}'.format(output_name), flush=True)
    with open(config_name, 'w') as f:
        json.dump(config, f, indent=4)

    print('Using following configuration:')
    print(json.dumps(config, indent=2))

    if training:
        labels, data, meta = create_dataset('train.json', True, **config)
        if config.get('pseudo_train', False):
            idxs, test, test_meta = create_dataset('test.json', False, **config)
            dataset = ((labels, data, meta), (idxs, test, test_meta))
        else:
            dataset = (labels, data, meta)
        print('Data loaded after {}'.format(runtime(start)))
        model = train(dataset, model, **config)
        print('Model trained after {}'.format(runtime(start)))

    idxs, test, test_meta = create_dataset('test.json', False, **config)
    dataset = (idxs, test, test_meta)
    evaluate(model, dataset, **config)
    print('Scriped successfully completed after {}'.format(runtime(start)))


def run_iterations(iters, name=None):
    """ Run iters iterations using random config func."""
    for i in range(int(iters)):
        print('Starting iteration {}/{}\n\n'.format(i, iters))
        #conf = gen_randomish_config(name)
        conf = gen_randomish_config2(name)
        if conf:
            single_run(conf)
        else:
            continue
        print('config iteration completed!!!')
        print('#' * 40)
        print('\n\n\n\n\n\n\n\n', flush=True)


if __name__ == '__main__':
    args = parase_arguments(sys.argv[1:])
    # random iters are quite different from normal model behaviour
    if args.iters:
        run_iterations(args.iters)
        sys.exit(0)
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        print('Using loaded config from: {}'.format(args.config))
    else:
        config = get_default_config()
    if args.load:
        config['model_fn'] = args.load
    elif args.model_fn:
        config['model_fn'] = args.model_fn
    if args.name:
        config['name'] = args.name
    if args.activation:
        config['relu_type'] = args.activation
    if args.epochs:
        config['epochs'] = args.epochs
    if args.pseudo:
        config['pseudo_train'] = args.pseudo
    if args.meta:
        config['use_meta'] = args.meta
    if args.evaluate:
        single_run(config, training=False)
    else:
        single_run(config)
