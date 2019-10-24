# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""CIFAR-10 final evaluation"""

import logging
import sys,os
sys.path.append('./')
from experiments.run_context import RunContext
from experiments.run_context import Training_log_plot
import tensorflow as tf
import datasets
from mean_teacher.mean_teacher import mean_teacher
from mean_teacher import minibatching
# from mean_teacher import minibatching_cluster as minibatching
LOG = logging.getLogger('main')

flags = tf.app.flags
flags.DEFINE_integer('gpu', 0, 'GPU_number')
flags.DEFINE_string('n_labeled', '1000', 'labeled data')
flags.DEFINE_string('ckp', './results/cifar10_final_eval/2000/transient/checkpoint-150001.meta', 'path of loaded checkpoint')
flags.DEFINE_integer('test_only', 0,'test or train')
flags.DEFINE_integer('dataset_index', 0, 'datasets including Cifar10ZCA, SVHN, Audio10')
flags.DEFINE_integer('n_runs', 1, 'number of runs')
FLAGS = flags.FLAGS

datasets_name = ['Cifar10ZCA','SVHN','Audio10','Audio30']
assert FLAGS.dataset_index<= len(datasets_name), 'wrong dataset index'
data_loader = getattr(datasets, datasets_name[FLAGS.dataset_index])

def parameters():
    test_phase = True
    n_runs = FLAGS.n_runs
    n_labeled = FLAGS.n_labeled
    
    for n_labeled in [1000, 2000, 4000, 'all']:
        for data_seed in range(2000, 2000 + n_runs):
            yield {
                'test_phase': test_phase,
                'n_labeled': n_labeled,
                'data_seed': data_seed
            }

def run(test_phase, n_labeled, data_seed):
    minibatch_size = 100

    data = data_loader(n_labeled=n_labeled,
                    data_seed=data_seed,
                    test_phase=test_phase)

    print('{} is loaded with {} of training samples'.format(datasets_name[FLAGS.dataset_index],data['num_train']))

    if n_labeled == 'all':
        n_labeled_per_batch =  minibatch_size
        max_consistency_cost = minibatch_size
    else:
        # n_labeled_per_batch = 'vary'
        n_labeled_per_batch = 20
        max_consistency_cost = minibatch_size* int(n_labeled) / data['num_train']

    hyper_dcit = {'input_dim': data['input_dim'],
                'label_dim': data['label_dim'],
                'flip_horizontally':True,
                'max_consistency_cost': max_consistency_cost,
                'apply_consistency_to_labeled' : True,
                'adam_beta_2_during_rampup': 0.999,
                'ema_decay_during_rampup': 0.999,
                'normalize_input': False,
                'rampdown_length': 25000,
                'training_length': 150000,
                'test_only':FLAGS.test_only
                }
 
    tf.reset_default_graph()
    runner_name = os.path.basename(__file__).split(".")[0]
    file_name = '{}_{}'.format(runner_name,n_labeled)
    log_plot = Training_log_plot(file_name,data_seed)
    model = mean_teacher(RunContext(file_name, data_seed), hyper_dcit)

    training_batches = minibatching.training_batches(data.training,
                                                     minibatch_size,
                                                     n_labeled_per_batch)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(data.evaluation,
                                                                    minibatch_size)

    if FLAGS.test_only:
        model.restore(FLAGS.ckp)
        model.evaluate(evaluation_batches_fn)
    else:
        model.train(training_batches, evaluation_batches_fn)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    for run_params in parameters():
        run(**run_params)
