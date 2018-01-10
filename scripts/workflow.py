import datetime as dt
import json, time
import numpy as np
import random
from random import choice

from utils import create_dataset
from cnn_train import train, evaluate
import models
import sys



####  Hyperparameters ####

def gen_randomish_config():
    shift_ranges = random.random()*.4
    channels = choice([2,3,3])
    if channels==2:
        preproc_strat = 'band2'
    else:
        preproc_strat = 'band3'
    
    return {
        'name': 'randomish_config',
        # training
        'lr': np.random.lognormal(sigma=0.5)*8e-5,
        'decay': np.random.lognormal()*1e-6,
        'relu_type': choice(['selu','relu','elu','elu','elu','elu']),
        'epochs': choice([150, 200, 225, 250, 275, 300, 500]),
        'full_cycls_per_epoch': 8,
        'batch_size': choice([16,32,64]),
        'lr_patience': choice([10, 15, 25, 50, 100]),
        'stop_patience': choice([15, 25, 50, 60, 100]),
        # data augmentation
        'hflip_prob': random.random(),
        'vflip_prob': random.random(),
        'rot90_prob': random.random()/2 + 0.5,
        'rot_prob': random.random()/2,
        'rotate_rg': random.random()*45,
        'shift_prob': random.random()/4,
        'shift_width_rg': shift_ranges,
        'shift_height_rg': shift_ranges,
        'zoom_prob': random.random()/4,
        'zoom_rg': (1-random.random()/4, 1+random.random()/4),
        'noise_prob': random.random()*2/3,
        'noise_rg': np.random.lognormal(sigma=.5)*.02,
        # model
        'use_meta': False,
        'model_fn': choice(['model1_wider','model0','model0','model0',
                            'model0','model0']),
        # preprocessing
        'preproc_strat': preproc_strat,
        'channels': channels,
        'inc_angle_fill': -1,
        'band3_op': choice(['lambda x1, x2: x1-x2','lambda x1, x2: x1+x2',
                            'lambda x1, x2: (x1+x2)/2','lambda x1, x2: x1*x2',
                            'lambda x1, x2: x1/x2','lambda x1, x2: (x1+x2)/2',
                            'lambda x1, x2: (x1+x2)/2','lambda x1, x2: (x1+x2)/2',
                            'lambda x1, x2: x2/x1']),
        'soft_targets': choice([False, False, True]),
        'soft_val': 0.99,  # only if soft_targets = True, must be 0.5 < x <= 1.0
    }


def get_default_config(cor_zoom=True):
    if cor_zoom:
        zoom_rg = (1.1, 1.1)
    else:
        zoom_rg = (.1, .1)
    return {
        'name': 'default',
        # training
        'lr': 8e-5,
        'decay': 1e-6,
        'relu_type': 'relu',
        'epochs': 250,
        'full_cycls_per_epoch': 8,
        'batch_size': 32,
        'lr_patience': 15,
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
        'zoom_prob': .15,
        'zoom_rg': zoom_rg,
        'noise_prob': .4,
        'noise_rg': .02,
        # model
        'use_meta': False,
        'model_fn': 'model0',
        # preprocessing
        'preproc_strat': 'band3',
        'inc_angle_fill': -1,
        'band3_op': 'lambda x1, x2: (x1+x2)/2',
        'soft_targets': False,
        'soft_val': 0.99,  # only if soft_targets = True, must be 0.5 < x <= 1.0
    }


def runtime(start):
    end_time = time.time()
    sec_elapsed = end_time - start
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def single_run(config):
    start = time.time()

    print('Using following configuration:')
    print(json.dumps(config, indent=2))

    model_fn = getattr(models, config['model_fn'])
    model = model_fn(**config)
    model.summary()
    print('Model compiled after {}'.format(runtime(start)))

    tmp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")
    config['tmp'] = tmp
    config_name = '../config/config_{}_{}.json'.format(tmp, config['name'])
    print('Saving configuration file to: {}'.format(config_name), flush=True)
    with open(config_name, 'w') as f:
        json.dump(config, f, indent=4)

    labels, data, meta = create_dataset('train.json', True, **config)
    print('Data loaded after {}'.format(runtime(start)))

    dataset = (labels, data, meta)
    model = train(dataset, model, **config)
    print('Model trained after {}'.format(runtime(start)))

    idxs, test, test_meta = create_dataset('test.json', False, **config)
    dataset = (idxs, test, test_meta)
    evaluate(model, dataset, **config)
    print('Scriped successfully completed after {}'.format(runtime(start)))



if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1].endswith('json'):
            with open(sys.argv[1]) as f:
                config = json.load(f)
            print('Using loaded config from: {}'.format(sys.argv[1]))
        elif sys.argv[1].isnumeric():
            iters = int(sys.argv[1])
            for i in range(iters):
                print('Starting iteration {}/{}\n\n'.format(i, iters))
                single_run(gen_randomish_config())
                print('config iteration completed!!!')
                print('#'*40)
                print('\n\n\n\n\n\n\n\n',flush=True)

    else:
        single_run(get_default_config())
