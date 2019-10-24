# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

from datetime import datetime
from collections import defaultdict
import threading
import time
import logging
import os

from pandas import DataFrame
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class Training_log_plot:
    
    def __init__(self,runner_file, run_idx, model_version=''):

        self.dir = "{root}/{runner_name}/{percentage}/{run_idx}".format(
            root='results',
            runner_name=runner_file,
            percentage = model_version,
            run_idx=run_idx
        )
        make_dir(self.dir)

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        self.train = {'step':[]}
        self.eval = {'step':[]}
        self.loss={'step':[],'value':[]}
        self.acc={'step':[],'value':[]}


    def write_log(self, log, refresh=False):
        # print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def add_loss(self, loss_log,step):
        self.loss['step'].append(step)
        self.loss['value'].append(loss_log)

    def add_acc(self, acc_log,step):
        self.acc['step'].append(step)
        self.acc['value'].append(acc_log)

    def add_train(self, log,step):
        self.train['step'].append(step)
        for key in log:
            try:
                self.train[key].append(log[key])
            except KeyError:
                self.train[key]=[log[key]]
 
    def add_eval(self, log,step):
        self.eval['step'].append(step)
        for key in log:
            try:
                self.eval[key].append(log[key])
            except KeyError:
                self.eval[key]=[log[key]]

    def save(self, is_best=False):
        # trainer.model.save(self.dir, epoch, is_best=is_best)
        # trainer.loss.save(self.dir)
        # trainer.loss.plot_loss(self.dir, epoch)

        # self.plot_loss()
        self.plot_acc()
        # self.plot_pdf(self.train,'Loss',[
        #     "train/error/1",
        #     "train/error/ema",
        #     "train/class_cost/1",
        #     "train/class_cost/ema",
        #     "train/cons_cost/mt",
        #     "train/total_cost/mt"])
        self.plot_pdf(self.eval,'Accuracy',[
            "eval/error/1",
            "eval/error/ema"])

    def save_train(self, is_best=False):
        # trainer.model.save(self.dir, epoch, is_best=is_best)
        # trainer.loss.save(self.dir)
        # trainer.loss.plot_loss(self.dir, epoch)

        self.plot_loss()
        # self.plot_acc()
        self.plot_pdf(self.train,'Loss',[
            "train/class_cost/1",
            "train/class_cost/ema",
            "train/cons_cost/mt"])

    def plot_acc(self):

        assert len(self.acc['step']) == len(self.acc['value']), 'Steps and acc data not match!'
        label = 'Err on {}'.format('test set')
        fig = plt.figure()
        plt.title(label)
        plt.plot( self.acc['step'], self.acc['value'], label=label)
        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('Error')
        plt.grid(True)
        plt.savefig('{}/test_.pdf'.format(self.dir ))
        plt.close(fig)

    def plot_loss(self):
        assert len(self.loss['step']) == len(self.loss['value']), 'Steps and acc data not match!'
        label = '{} Loss'.format('Train')
        fig = plt.figure()
        plt.title(label)
        plt.plot(self.loss['step'], self.loss['value'], label=label)
        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('{}/loss_.pdf'.format(self.dir))
        plt.close(fig)

    def plot_pdf(self,log,title,plot_list):
        fig = plt.figure()
        plt.title(title)
        for label in plot_list:
            assert len(log['step']) == len(log[label]), 'Step and data not match!'
            plt.plot(log['step'], log[label], label=label)
        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel(title)
        plt.grid(True)
        plt.savefig('{}/{}.pdf'.format(self.dir,title))
        plt.close(fig)


class TrainLog:
    """Saves training logs in Pandas msgpacks"""

    INCREMENTAL_UPDATE_TIME = 300

    def __init__(self, directory, name):
        self.log_file_path = "{}/{}.msgpack".format(directory, name)
        self._log = defaultdict(dict)
        self._log_lock = threading.RLock()
        self._last_update_time = time.time() - self.INCREMENTAL_UPDATE_TIME


    def record_single(self, step, column, value):
        self._record(step, {column: value})

    def record(self, step, col_val_dict):
        self._record(step, col_val_dict)

    def save(self):
        df = self._as_dataframe()
        df.to_msgpack(self.log_file_path, compress='zlib')

    def _record(self, step, col_val_dict):
        with self._log_lock:
            self._log[step].update(col_val_dict)
            # if time.time() - self._last_update_time >= self.INCREMENTAL_UPDATE_TIME:
                # self._last_update_time = time.time()
            self.save()

    def _as_dataframe(self):
        with self._log_lock:
            return DataFrame.from_dict(self._log, orient='index')


class RunContext:
    """Creates directories and files for the run"""

    def __init__(self, runner_file, run_idx, model_version=''):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        # runner_name = os.path.basename(runner_file).split(".")[0]
        runner_name = runner_file
        # self.result_dir = "{root}/{runner_name}/{date:%Y-%m-%d_%H:%M:%S}/{run_idx}".format(
        #     root='results',
        #     runner_name=runner_name,
        #     date=datetime.now(),
        #     run_idx=run_idx
        # )
        self.result_dir = "{root}/{runner_name}/{percentage}/{run_idx}".format(
            root='results',
            runner_name=runner_name,
            percentage = model_version,
            run_idx=run_idx
        )
        self.transient_dir = self.result_dir + "/transient"
        # os.makedirs(self.result_dir)
        # os.makedirs(self.transient_dir)
        make_dir(self.result_dir)
        make_dir(self.transient_dir)

    def create_train_log(self, name):
        return TrainLog(self.result_dir, name)

def make_dir(dir):
    try:
        os.makedirs(dir)  
    except OSError:
        pass 
