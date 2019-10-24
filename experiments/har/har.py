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
from mean_teacher.arguments import args
from mean_teacher.mean_teacher_base import mean_teacher_base as mean_teacher
from mean_teacher import minibatching


LOG = logging.getLogger('main')

data_loader = getattr(datasets, args.dataset)

def parameters(): 
    for n_labeled in args.n_labeled:
        for data_seed in range(args.init_run, args.init_run + args.n_runs):
            yield {
                'n_labeled': n_labeled,
                'data_seed': data_seed
            }

def run(n_labeled, data_seed):

    data = data_loader(n_labeled=n_labeled,
                    data_seed=data_seed,
                    test_phase=True,
                    dataset_detail = args.dataset_detail)

    print('{} is loaded with {} of training samples'.format(args.dataset,data['num_train']))

    if n_labeled == 'all':
        args.n_labeled_per_batch =  args.minibatch_size
        args.max_consistency_cost = args.minibatch_size
    else:
        if args.max_consistency_cost != 0 :
            args.max_consistency_cost = args.minibatch_size* int(n_labeled) / data['num_train']


    tf.reset_default_graph()
    runner_name = os.path.basename(__file__).split(".")[0]
    runner_name = args.save
    file_name = '{}_{}'.format(runner_name,n_labeled)
    log_plot = Training_log_plot(file_name,data_seed)
    model = mean_teacher(RunContext(file_name, data_seed), args,log_plot)

    training_batches = minibatching.training_batches(data.training,
                                                     args.minibatch_size,
                                                     args.n_labeled_per_batch)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(data.evaluation,
                                                                    args.minibatch_size)
    test_batches_fn = minibatching.evaluation_epoch_generator(data.evaluation,
                                                                    batch_size = 260)


    # import pdb; pdb.set_trace()
    if args.test_only:

        print('loading folers')
        root_path = "./results/"
        folders = os.listdir(root_path)
        assert args.ckp != '.','No ckp info was input'
        for i in range (len(folders)):
            folders[i] = os.path.join(root_path,folders[i])
            #print(folders[i])
        for folder in folders:
            if args.ckp in folder:
                print(folder)

                matrix = []
                for random_seed in os.listdir(folder):
                    ckp_path = os.path.join(folder,random_seed,'transient')
                    ckp = tf.train.latest_checkpoint(ckp_path)
                    print('restore checkpoint from {}'.format(ckp))
                    model.restore(ckp)

                    confuse_matrix = model.confusion_matrix(test_batches_fn)
                    print(acc_from_confuse(confuse_matrix))
                    acc_matrix = acc_from_confuse(confuse_matrix)
                    matrix.append(acc_matrix)

                save_confuse_matrix(matrix,ckp_path)

    else:
        model.train(training_batches, evaluation_batches_fn)

import numpy as np
def acc_from_confuse(matrix):
    num_sample_cls = np.sum(matrix,axis=1)
    correct = np.diag(matrix)
    accuracy = correct/num_sample_cls 
    return accuracy

def save_confuse_matrix(matrix,ckp_path):
    matrix = np.asarray(matrix)
    average = np.mean(matrix,axis=0)
    # var = np.std(average)
    csv_path = './results/csv/confuse_matrix/'+ckp_path.split('/')[2]+'_avg_conf.csv'
    # average.to_csv(csv_path)
    np.savetxt(csv_path,average)
    print('File saved as {}'.format(csv_path))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for run_params in parameters():
        run(**run_params)
