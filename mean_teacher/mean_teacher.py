# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"Mean teacher model"

import logging
import os
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib import metrics, slim
from tensorflow.contrib.metrics import streaming_mean

from . import nn
from .framework import assert_shape
from . import string_utils
from .loss import errors, classification_costs, consistency_costs, total_costs
from . import model
from .ramp import ramp_value
from . import ict
LOG = logging.getLogger('main')

import numpy as np
# import experiments.run_context.Text_log as Text_log
from .ramp import ramp_temperature

class mean_teacher(object):
    hyper = {

        # architecture of cnn
        'cnn': 'tower',
        'input_dim': (32,32,3),
        'label_dim':(),
        'sig':False, # sigmoid or softmax

        # Consistency hyperparameters
        'ema_consistency': True,
        'apply_consistency_to_labeled': True,
        'max_consistency_cost': 100.0,
        'ema_decay_during_rampup': 0.99,
        'ema_decay_after_rampup': 0.999,
        'consistency_trust': 0.0,

        # Optimizer hyperparameters
        'max_learning_rate': 0.003,
        'adam_beta_1_before_rampdown': 0.9,
        'adam_beta_1_after_rampdown': 0.5,
        'adam_beta_2_during_rampup': 0.99,
        'adam_beta_2_after_rampup': 0.999,
        'adam_epsilon': 1e-8,

        # Architecture hyperparameters
        'input_noise': 0.15,
        'student_dropout_probability': 0.5,
        'teacher_dropout_probability': 0.5,

        # Training schedule
        'rampup_length': 40000,
        'rampdown_length': 25000,
        'training_length': 150000,

        # Input augmentation
        'flip_horizontally': False,
        'translate': True,
 
        # Whether to scale each input image to mean=0 and std=1 per channel
        # Use False if input is already normalized in some other way
        'normalize_input': True,

        # Output schedule
        'print_span': 20,
        'evaluation_span': 500,

        # specially designed for audio dataset
        'bg_noise':False,
        'bg_noise_input': None,
        'bg_noise_level': 0,


        'optimizer': 'adam',
        'test_only': False,
        # 'ict': False
        # distilling temperature
        # 'max_temp': 10,
        'max_temp': 1,
        'min_temp': 1,
        'distilling':1
    }

    #pylint: disable=too-many-instance-attributes
    def __init__(self,run_context=None,  hyper_dict={},log_plot=None ):

        #inilization of hyper
        for i in hyper_dict:
            assert i in self.hyper, "Wrong hyper dict '{}'!".format(i)
            self.hyper[i] = hyper_dict[i]

        self.log_plot = log_plot
        # inilize bg noise input
        if self.hyper['bg_noise']:
            self.bg_noise_input = tf.convert_to_tensor(self.hyper['bg_noise_input'],dtype=tf.float32)
        else:
            self.bg_noise_input = tf.convert_to_tensor(np.zeros((32,32)),dtype=tf.float32)

        # inilization model
        print('{} is initliazed!'.format(self.hyper['cnn']))
        self.cnn = getattr(model,self.hyper['cnn'])

        if run_context is not None:
            self.training_log = run_context.create_train_log('training')
            self.validation_log = run_context.create_train_log('validation')
            self.checkpoint_path = os.path.join(run_context.transient_dir, 'checkpoint')
            self.tensorboard_path = os.path.join(run_context.result_dir, 'tensorboard')

        with tf.name_scope("placeholders"):
            self.images = tf.placeholder(dtype=tf.float32, shape=(None,) + self.hyper['input_dim'], name='images')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,) + self.hyper['label_dim'], name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        tf.add_to_collection("init_in_init", self.global_step)

        with tf.name_scope("ramps"):
            self.learning_rate, self.cons_coefficient, \
            self.adam_beta_1, self.adam_beta_2, \
            self.ema_decay = ramp_value(self.global_step,self.hyper)

            if self.hyper['max_temp']!=self.hyper['min_temp']:
                self.temperature = ramp_temperature(self.global_step,self.hyper)
            else:
                self.temperature = tf.to_float(self.hyper['min_temp'])

        (   self.class_logits_1,
            self.class_logits_ema
        ) = self.inference(
            self.images,
            is_training=self.is_training,
            ema_decay=self.ema_decay,
            input_noise=self.hyper['input_noise'],
            student_dropout_probability=self.hyper['student_dropout_probability'],
            teacher_dropout_probability=self.hyper['teacher_dropout_probability'],
            normalize_input=self.hyper['normalize_input'],
            flip_horizontally=self.hyper['flip_horizontally'],
            translate=self.hyper['translate'])

        with tf.name_scope("objectives"):
            self.mean_error_1, self.errors_1 = errors(self.class_logits_1, self.labels,sig = self.hyper['sig'])
            self.mean_error_ema, self.errors_ema = errors(self.class_logits_ema, self.labels,sig = self.hyper['sig'])

            self.mean_class_cost_1, self.class_costs_1 = classification_costs(
                self.class_logits_1, self.labels,sig = self.hyper['sig'])
            self.mean_class_cost_ema, self.class_costs_ema = classification_costs(
                self.class_logits_ema, self.labels,sig = self.hyper['sig'])

            labeled_consistency = self.hyper['apply_consistency_to_labeled']
            consistency_mask = tf.logical_or(tf.equal(self.labels, -1), labeled_consistency)
            self.mean_cons_cost_mt, self.cons_costs_mt = consistency_costs(
                self.class_logits_1, self.class_logits_ema, self.cons_coefficient, consistency_mask, 
                self.hyper['consistency_trust'],temperature=self.temperature,distilling=self.hyper['distilling'])

            self.mean_total_cost_mt, self.total_costs_mt = total_costs(
                self.class_costs_1, self.cons_costs_mt)

            self.cost_to_be_minimized = self.mean_total_cost_mt


        with tf.name_scope("train_step"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if self.hyper['optimizer']=='adam':
                    self.train_step_op = nn.adam_optimizer(self.cost_to_be_minimized,
                                                       self.global_step,
                                                       learning_rate=self.learning_rate,
                                                       beta1=self.adam_beta_1,
                                                       beta2=self.adam_beta_2,
                                                       epsilon=self.hyper['adam_epsilon'])
                elif self.hyper['optimizer']=='sgd':
                    self.train_step_op = nn.sgd_optimizer(self.cost_to_be_minimized,
                                                       self.global_step,
                                                       learning_rate=self.hyper['max_learning_rate'])
                else:
                    assert False, 'Wrong optimizer input!'


        self.training_metrics = {
            "learning_rate": self.learning_rate,
            "adam_beta_1": self.adam_beta_1,
            "adam_beta_2": self.adam_beta_2,
            "ema_decay": self.ema_decay,
            "cons_coefficient": self.cons_coefficient,
            "train/error/1": self.mean_error_1,
            "train/error/ema": self.mean_error_ema,
            "train/class_cost/1": self.mean_class_cost_1,
            "train/class_cost/ema": self.mean_class_cost_ema,
            "train/cons_cost/mt": self.mean_cons_cost_mt,
            "train/total_cost/mt": self.mean_total_cost_mt,
        }

        with tf.variable_scope("validation_metrics") as metrics_scope:
            self.metric_values, self.metric_update_ops = metrics.aggregate_metric_map({
                "eval/error/1": streaming_mean(self.errors_1),
                "eval/error/ema": streaming_mean(self.errors_ema),
                "eval/class_cost/1": streaming_mean(self.class_costs_1),
                "eval/class_cost/ema": streaming_mean(self.class_costs_ema),
            })
            metric_variables = slim.get_local_variables(scope=metrics_scope.name)
            self.metric_init_op = tf.variables_initializer(metric_variables)

        self.result_formatter = string_utils.DictFormatter(
            order=["eval/error/ema", "error/1", "class_cost/1", "cons_cost/mt"],
            default_format='{name}: {value:>10.6f}',
            separator=",  ")
        self.result_formatter.add_format('error', '{name}: {value:>6.1%}')

        with tf.name_scope("initializers"):
            init_init_variables = tf.get_collection("init_in_init")
            train_init_variables = [
                var for var in tf.global_variables() if var not in init_init_variables
            ]
            self.init_init_op = tf.variables_initializer(init_init_variables)
            self.train_init_op = tf.variables_initializer(train_init_variables)

        self.saver = tf.train.Saver()
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.run(self.init_init_op)
        self.save_tensorboard_graph()



    def train(self, training_batches, evaluation_batches_fn):
        self.run(self.train_init_op, self.feed_dict(next(training_batches)))
        LOG.info("Model variables initialized")
        self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()
        for batch in training_batches:
            # if self.hyper['bg_noise_input']!=None:
            #     batch = self.add_bg_noise(batch)
            results, step,temp, _ = self.run([self.training_metrics, self.global_step, self.temperature,
                    self.train_step_op], self.feed_dict(batch))

            if step % self.hyper['print_span']==0: 
                self.training_log.record(step, {**results })
                LOG.info(temp)
                log = "step %5d:   %s"%(step, self.result_formatter.format_dict(results))
                LOG.info(log)
                if self.log_plot:
                    self.log_plot.write_log(log)
                    self.log_plot.add_loss(results["train/total_cost/mt"],step)
                    self.log_plot.add_train(results,step)
                    self.log_plot.save_train()
            if step > self.hyper['training_length']:
                break
            if step % self.hyper['evaluation_span'] ==0 and step!=0:
                self.evaluate(evaluation_batches_fn)
                self.save_checkpoint()
            
        self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()

    def evaluate(self, evaluation_batches_fn):
        self.run(self.metric_init_op)
        for batch in evaluation_batches_fn():
            if self.hyper['test_only']:
                batch = self.input_ensemble(batch)
            self.run(self.metric_update_ops, feed_dict=self.feed_dict(batch, is_training=False))
        step = self.run(self.global_step)
        results = self.run(self.metric_values)
        self.validation_log.record(step, results)
        log = "step %5d:   %s"%(step, self.result_formatter.format_dict(results))
        LOG.info(log)
        # LOG.info("step %5d:   %s", step, self.result_formatter.format_dict(results))
        if self.log_plot:
            self.log_plot.write_log(log,refresh= True)
            self.log_plot.add_acc(results['eval/error/ema'],step)
            self.log_plot.add_eval(results,step)
            self.log_plot.save()

        if not self.hyper['test_only']: 
            self.validation_log.record(step, results)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def feed_dict(self, batch, is_training=True):
        return {
            self.images: batch['x'],
            self.labels: batch['y'],
            self.is_training: is_training
        }

    def input_ensemble(self,batch):
        tmp = batch['x'].copy() #(100, 32, 32, 3)
        # flip = tmp[:,:,::-1,:]
        flip = tmp[:,::-1,:,:]
        # translate = self.translate(tmp,0,5) + self.translate(tmp,0,-5)+ self.translate(tmp,-5,0)+ self.translate(tmp,5,0)
        # gaussian_noise = tmp + np.random.normal(size=tmp.shape) *0.15
        # ensemble = (flip + translate + gaussian_noise)/6
        # print('ensemble samples')
        # batch['x']=ensemble
        batch['x'] = (flip+tmp)/2
        return batch

    def translate(self,batch,offset_height,offset_width):
        import scipy
        tmp = scipy.ndimage.shift(batch,shift=(0,offset_height,offset_width,0),mode='reflect') 
        return tmp

    def restore(self,ckp):
        self.saver.restore(self.session,ckp)

    def save_checkpoint(self):
        path = self.saver.save(self.session, self.checkpoint_path, global_step=self.global_step)
        LOG.info("Saved checkpoint: %r", path)

    def save_tensorboard_graph(self):
        writer = tf.summary.FileWriter(self.tensorboard_path)
        writer.add_graph(self.session.graph)
        LOG.info("Saved tensorboard graph to %r", writer.get_logdir())

    def inference(self,inputs, is_training, ema_decay, input_noise, student_dropout_probability, teacher_dropout_probability,
                  normalize_input, flip_horizontally, translate):
        tower_args = dict(inputs=inputs,
                          is_training=is_training,
                          input_noise=input_noise,
                          normalize_input=normalize_input,
                          flip_horizontally=flip_horizontally,
                          translate=translate,
                          bg_noise = self.hyper['bg_noise'],
                          bg_noise_level =self.hyper['bg_noise_level'],
                          bg_noise_input =self.bg_noise_input )


        if 'resnet' in self.hyper['cnn']:
            with tf.variable_scope("student") as var_scope:
                class_logits_1 = self.cnn(**tower_args, dropout_probability=student_dropout_probability)
        else:
            with tf.variable_scope("initialization") as var_scope:
                _ = self.cnn(**tower_args, dropout_probability=student_dropout_probability, is_initialization=True)
            with tf.variable_scope(var_scope, reuse=True):
                class_logits_1 = self.cnn(**tower_args, dropout_probability=student_dropout_probability)

        # ema
        model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ema = tf.train.ExponentialMovingAverage(decay=ema_decay)

        with tf.control_dependencies(update_ops):
            ema_op = ema.apply(model_vars)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

        def ema_getter(getter, name, *args, **kwargs):
            var = getter(name, *args, **kwargs)
            if 'moving_mean' in var.name or 'moving_variance' in var.name:
                return var
            else:
                assert var in model_vars, "Unknown variable {}.".format(var)
                return ema.average(var)

        with tf.variable_scope(var_scope, custom_getter = ema_getter, reuse = True):
            class_logits_ema = self.cnn(**tower_args, dropout_probability=teacher_dropout_probability)
            class_logits_ema = tf.stop_gradient(class_logits_ema)

        return (class_logits_1,class_logits_ema)


    def __getitem__(self, key):
        return self.hyper[key]

    def __setitem__(self, key, value):
        self.hyper[key]=value