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
LOG = logging.getLogger('main')

import numpy as np
from .ramp import ramp_temperature
from mean_teacher.mean_teacher_base import mean_teacher_base

class mean_teacher_co(mean_teacher_base):

    def __init__(self,run_context=None,  hyper_dict={},log_plot=None ):
        self.stu_dist = tf.Variable(0.0,trainable=False)
        self.init_dist_op = tf.variables_initializer([self.stu_dist])
        super(mean_teacher_co, self).__init__(run_context, hyper_dict, log_plot)

    def session_start(self):
        self.result_formatter.order.extend(["error/ema", "class_cost/ema"])
        self.weight_distance()
        with tf.name_scope('stu_dist'):
            tf.summary.scalar('stu_dist',self.stu_dist)
        self.merged = tf.summary.merge_all()
        self.session = tf.Session(config=self.config)
        self.run(self.init_dist_op)
        self.run(self.init_init_op) 
        self.save_tensorboard_graph()

    def forward(self):

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

        # self.class_logits_ema = tf.stop_gradient(self.class_logits_ema)

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
                self.hyper['consistency_trust'], normalization = self.hyper['consistency_norm'],
                temperature = self.temperature, distilling = self.hyper['distilling'])

            self.mean_total_cost_mt, self.total_costs_mt = total_costs(
                self.class_costs_1, self.class_costs_ema, self.cons_costs_mt)

            self.cost_to_be_minimized = self.mean_total_cost_mt


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

        with tf.variable_scope('stu1') as var_scope1:
            _ = self.cnn(**tower_args, dropout_probability=student_dropout_probability, is_initialization=True)
        with tf.variable_scope(var_scope1, reuse=True):
            class_logits_1 = self.cnn(**tower_args, dropout_probability=student_dropout_probability) 
        with tf.variable_scope('stu2') as var_scope2:
            _ = self.cnn(**tower_args, dropout_probability=student_dropout_probability, is_initialization=True)
        with tf.variable_scope(var_scope2, reuse=True):
            class_logits_2 = self.cnn(**tower_args, dropout_probability=student_dropout_probability) 

        return (class_logits_1,class_logits_2)

    def weight_distance(self):

        stu1_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='stu1')
        stu2_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='stu2')
        assert len(stu1_var)==len(stu2_var)

        for i,j in zip(stu1_var, stu2_var):
            t1 = i.op.outputs[0]
            t2 = j.op.outputs[0]
            assert t1.shape==t2.shape
            self.stu_dist = self.stu_dist + tf.reduce_sum(tf.squared_difference(t1,t2))
