import tensorflow as tf
from tensorflow.contrib import slim
from .framework import assert_shape
from . import weight_norm as wn
from . import nn
from . import resnet_ops


def tower(inputs,
          is_training,
          dropout_probability,
          input_noise,
          normalize_input,
          flip_horizontally,
          translate,
          is_initialization=False,
          name=None,
          bg_noise=False,
          bg_noise_level=0,
          bg_noise_input=None,
         resize = False):
    with tf.name_scope(name, "tower"):
        default_conv_args = dict(
            padding='SAME',
            kernel_size=[3, 3],
            activation_fn=nn.lrelu,
            init=is_initialization
        )
        training_mode_funcs = [
            nn.random_translate, nn.flip_randomly, nn.gaussian_noise, slim.dropout,
            wn.fully_connected, wn.conv2d
        ]
        training_args = dict(
            is_training=is_training
        )

        with \
        slim.arg_scope([wn.conv2d], **default_conv_args), \
        slim.arg_scope(training_mode_funcs, **training_args):

            net = inputs
            assert_shape(net, [None, 32, 32, 3])

            net = tf.cond(tf.convert_to_tensor(normalize_input),
                          lambda: slim.layer_norm(net,
                                                  scale=False,
                                                  center=False,
                                                  scope='normalize_inputs'),
                          lambda: net)
            assert_shape(net, [None, 32, 32, 3])

            net = nn.flip_randomly(net,
                                   horizontally=flip_horizontally,
                                   vertically=False,
                                   name='random_flip')
            net = tf.cond(tf.convert_to_tensor(translate),
                          lambda: nn.random_translate(net, scale=2, name='random_translate'),
                          lambda: net)
            net = nn.gaussian_noise(net, scale=input_noise, name='gaussian_noise')

            net = wn.conv2d(net, 128, scope="conv_1_1")
            net = wn.conv2d(net, 128, scope="conv_1_2")
            net = wn.conv2d(net, 128, scope="conv_1_3")
            net = slim.max_pool2d(net, [2, 2], scope='max_pool_1')
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout_probability_1')
            assert_shape(net, [None, 16, 16, 128])

            net = wn.conv2d(net, 256, scope="conv_2_1")
            net = wn.conv2d(net, 256, scope="conv_2_2")
            net = wn.conv2d(net, 256, scope="conv_2_3")
            net = slim.max_pool2d(net, [2, 2], scope='max_pool_2')
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout_probability_2')
            assert_shape(net, [None, 8, 8, 256])

            net = wn.conv2d(net, 512, padding='VALID', scope="conv_3_1")
            assert_shape(net, [None, 6, 6, 512])
            net = wn.conv2d(net, 256, kernel_size=[1, 1], scope="conv_3_2")
            net = wn.conv2d(net, 128, kernel_size=[1, 1], scope="conv_3_3")
            net = slim.avg_pool2d(net, [6, 6], scope='avg_pool')
            assert_shape(net, [None, 1, 1, 128])

            net = slim.flatten(net)
            assert_shape(net, [None, 128])

            primary_logits = wn.fully_connected(net, 10, init=is_initialization)

            assert_shape(primary_logits, [None, 10])
            
            return primary_logits


def har_conv2d(inputs,
          is_training,
          dropout_probability,
          input_noise,
          normalize_input,
          flip_horizontally,
          translate,
          is_initialization=False,
          name=None,
          bg_noise=False,
          bg_noise_level=0,
          bg_noise_input=None,
          resize = False):
    with tf.name_scope(name, "tower"):
        default_conv_args = dict(
            padding='SAME',
            kernel_size=[3, 3],
            activation_fn=nn.lrelu,
            init=is_initialization
        )
        training_mode_funcs = [
            nn.random_translate, nn.flip_randomly, nn.gaussian_noise, slim.dropout,
            wn.fully_connected, wn.conv2d,
            nn.resize
        ]
        training_args = dict(
            is_training=is_training
        )

        with \
        slim.arg_scope([wn.conv2d], **default_conv_args), \
        slim.arg_scope(training_mode_funcs, **training_args):

            net = inputs
            assert_shape(net, [None, 9, 64, 1])

            net = tf.cond(tf.convert_to_tensor(normalize_input),
                          lambda: slim.layer_norm(net,
                                                  scale=False,
                                                  center=False,
                                                  scope='normalize_inputs'),
                          lambda: net)


            # net = tf.cond(tf.convert_to_tensor(translate),
            #               lambda: nn.random_translate(net, scale=[0,2], name='random_translate'),
            #               lambda: net)

            # net = tf.cond(tf.convert_to_tensor(resize),
            #               lambda: nn.resize(net,scale=1, name='resize_aug'),
            #               lambda: net)

            net = nn.gaussian_noise(net, scale=input_noise, name='gaussian_noise')

            net = wn.conv2d(net, 128, scope="conv_1_1")
            net = wn.conv2d(net, 128, scope="conv_1_2")
            # net = wn.conv2d(net, 128, scope="conv_1_3")
            net = slim.max_pool2d(net, [1, 2],stride=[1,2], scope='max_pool_1')
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout_probability_1')
            assert_shape(net, [None, 9, 32, 128])

            net = wn.conv2d(net, 256, scope="conv_2_1")
            net = wn.conv2d(net, 256, scope="conv_2_2")
            # net = wn.conv2d(net, 256, scope="conv_2_3")
            net = slim.max_pool2d(net, [1, 2], stride=[1,2],scope='max_pool_2')
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout_probability_2')
            assert_shape(net, [None, 9, 16, 256])

            net = wn.conv2d(net, 256, scope="conv_3_1")
            net = wn.conv2d(net, 256, scope="conv_3_2")
            # net = wn.conv2d(net, 256, scope="conv_2_3")
            net = slim.max_pool2d(net, [1, 2],stride=[1,2], scope='max_pool_3')
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout_probability_2') 
            assert_shape(net, [None, 9, 8, 256])


            net = wn.conv2d(net, 512, padding='VALID', scope="conv_4_1")
            assert_shape(net, [None, 7, 6, 512])
            net = wn.conv2d(net, 256, kernel_size=[1, 1], scope="conv_4_2")
            net = wn.conv2d(net, 128, kernel_size=[1, 1], scope="conv_4_3")
            net = slim.avg_pool2d(net, [7, 6], scope='avg_pool')
            assert_shape(net, [None, 1, 1, 128])

            net = slim.flatten(net)
            assert_shape(net, [None, 128])

            primary_logits = wn.fully_connected(net, 24, init=is_initialization)

            assert_shape(primary_logits, [None, 24])
            
            return primary_logits



def har_sp(inputs,
          is_training,
          dropout_probability,
          input_noise,
          normalize_input,
          flip_horizontally,
          translate,
          is_initialization=False,
          name=None,
          bg_noise=False,
          bg_noise_level=0,
          bg_noise_input=None):
    with tf.name_scope(name, "tower"):
        default_conv_args = dict(
            padding='SAME',
            kernel_size=[3, 3],
            activation_fn=nn.lrelu,
            init=is_initialization
        )
        training_mode_funcs = [
            nn.random_translate, nn.flip_randomly, nn.gaussian_noise, slim.dropout,
            wn.fully_connected, wn.conv2d
        ]
        training_args = dict(
            is_training=is_training
        )

        with \
        slim.arg_scope([wn.conv2d], **default_conv_args), \
        slim.arg_scope(training_mode_funcs, **training_args):

            net = inputs
            assert_shape(net, [None, 17, 16, 9])

            net = tf.cond(tf.convert_to_tensor(normalize_input),
                          lambda: slim.layer_norm(net,
                                                  scale=False,
                                                  center=False,
                                                  scope='normalize_inputs'),
                          lambda: net)


            # net = tf.cond(tf.convert_to_tensor(translate),
            #               lambda: nn.random_translate(net, scale=[0,2], name='random_translate'),
            #               lambda: net)
            net = nn.gaussian_noise(net, scale=input_noise, name='gaussian_noise')

            net = wn.conv2d(net, 128, scope="conv_1_1")
            net = wn.conv2d(net, 128, scope="conv_1_2")
            net = wn.conv2d(net, 128, scope="conv_1_3")
            net = slim.max_pool2d(net, [2, 2], scope='max_pool_1')
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout_probability_1')
            assert_shape(net, [None, 8, 8, 128])

            net = wn.conv2d(net, 256, scope="conv_2_1")
            net = wn.conv2d(net, 256, scope="conv_2_2")
            net = wn.conv2d(net, 256, scope="conv_2_3")
            net = slim.max_pool2d(net, [2, 2], scope='max_pool_2')
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout_probability_2')
            assert_shape(net, [None, 4, 4, 256])

            net = wn.conv2d(net, 512, scope="conv_3_1")
            assert_shape(net, [None, 4, 4, 512])
            net = wn.conv2d(net, 256, kernel_size=[1, 1], scope="conv_3_2")
            net = wn.conv2d(net, 128, kernel_size=[1, 1], scope="conv_3_3")
            net = slim.avg_pool2d(net, [4, 4], scope='avg_pool')
            assert_shape(net, [None, 1, 1, 128])

            net = slim.flatten(net)
            assert_shape(net, [None, 128])

            primary_logits = wn.fully_connected(net, 24, init=is_initialization)

            assert_shape(primary_logits, [None, 24])
            
            return primary_logits

# def har(inputs,
#           is_training,
#           dropout_probability,
#           input_noise,
#           normalize_input,
#           flip_horizontally,
#           translate,
#           is_initialization=False,
#           training = False,
#           name=None):
#     with tf.name_scope(name, "dac"):

#         training_mode_funcs = [
#             nn.step_noise, slim.dropout, wn.fully_connected,nn.gaussian_noise
#         ]
#         training_args = dict(
#             is_training=is_training
#         )

#         with slim.arg_scope(training_mode_funcs, **training_args):
#             net = inputs
#             assert_shape(net, [None, 9, 64,1])

#             net = nn.gaussian_noise(net, scale=input_noise['gaussian'], name='gaussian_noise')
#             net = nn.step_noise(net, step_size=input_noise, name='step_noise')

#             net = tf.cond(tf.convert_to_tensor(normalize_input),
#                           lambda: slim.layer_norm(net,
#                                                   scale=False,
#                                                   center=False,
#                                                   scope='normalize_inputs'),
#                           lambda: net)

#             net = wn.fully_connected(net, 32, init=is_initialization,activation_fn=nn.lrelu,scope="dense_1")
#             net = wn.fully_connected(net, 128, init=is_initialization,activation_fn=nn.lrelu,scope="dense_2")
#             net = slim.dropout(net, 1 - dropout_probability, scope='dropout')
#             primary_logits = wn.fully_connected(net, 1, init=is_initialization,scope="dense_out")

#             return primary_logits


def dac(inputs,
          is_training,
          dropout_probability,
          input_noise,
          normalize_input,
          flip_horizontally,
          translate,
          is_initialization=False,
          training = False,
          name=None):
    with tf.name_scope(name, "dac"):

        training_mode_funcs = [
            nn.step_noise, slim.dropout, wn.fully_connected,nn.gaussian_noise
        ]
        training_args = dict(
            is_training=is_training
        )

        with slim.arg_scope(training_mode_funcs, **training_args):
            net = inputs
            assert_shape(net, [None, 13])

            # net = nn.gaussian_noise(net, scale=input_noise['gaussian'], name='gaussian_noise')
            net = nn.step_noise(net, step_size=input_noise, name='step_noise')

            net = tf.cond(tf.convert_to_tensor(normalize_input),
                          lambda: slim.layer_norm(net,
                                                  scale=False,
                                                  center=False,
                                                  scope='normalize_inputs'),
                          lambda: net)

            net = wn.fully_connected(net, 32, init=is_initialization,activation_fn=nn.lrelu,scope="dense_1")
            net = wn.fully_connected(net, 128, init=is_initialization,activation_fn=nn.lrelu,scope="dense_2")
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout')
            primary_logits = wn.fully_connected(net, 1, init=is_initialization,scope="dense_out")

            return primary_logits

def solar(inputs,
          is_training,
          dropout_probability,
          input_noise,
          normalize_input,
          flip_horizontally,
          translate,
          is_initialization=False,
          training = False,
          name=None):
    with tf.name_scope(name, "solar"):

        training_mode_funcs = [
            nn.step_noise, slim.dropout, wn.fully_connected,nn.gaussian_noise
        ]
        training_args = dict(
            is_training=is_training
        )

        with slim.arg_scope(training_mode_funcs, **training_args):
            net = inputs
            assert_shape(net, [None, 4])

            # net = nn.gaussian_noise(net, scale=input_noise['gaussian'], name='gaussian_noise')
            # net = nn.step_noise(net, step_size=input_noise, name='step_noise')

            net = tf.cond(tf.convert_to_tensor(normalize_input),
                          lambda: slim.layer_norm(net,
                                                  scale=False,
                                                  center=False,
                                                  scope='normalize_inputs'),
                          lambda: net)

            net = wn.fully_connected(net, 32, init=is_initialization,activation_fn=nn.lrelu,scope="dense_1")
            net = wn.fully_connected(net, 128, init=is_initialization,activation_fn=nn.lrelu,scope="dense_2")
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout')
            primary_logits = wn.fully_connected(net, 1, init=is_initialization,scope="dense_out")

            return primary_logits

def solar8(inputs,
          is_training,
          dropout_probability,
          input_noise,
          normalize_input,
          flip_horizontally,
          translate,
          is_initialization=False,
          training = False,
          name=None):
    with tf.name_scope(name, "solar8"):

        training_mode_funcs = [
            nn.step_noise_solar, slim.dropout, wn.fully_connected,nn.gaussian_noise
        ]
        training_args = dict(
            is_training=is_training
        )

        with slim.arg_scope(training_mode_funcs, **training_args):
            net = inputs
            assert_shape(net, [None, 8])

            # net = nn.gaussian_noise(net, scale=input_noise['gaussian'], name='gaussian_noise')
            net = nn.step_noise_solar(net, step_size=input_noise, name='step_noise')

            net = tf.cond(tf.convert_to_tensor(normalize_input),
                          lambda: slim.layer_norm(net,
                                                  scale=False,
                                                  center=False,
                                                  scope='normalize_inputs'),
                          lambda: net)

            net = wn.fully_connected(net, 16, init=is_initialization,activation_fn=nn.lrelu,scope="dense_1")
            net = wn.fully_connected(net, 32, init=is_initialization,activation_fn=nn.lrelu,scope="dense_2")
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout')
            primary_logits = wn.fully_connected(net, 1, init=is_initialization,scope="dense_out")

            return primary_logits


def solar24(inputs,
          is_training,
          dropout_probability,
          input_noise,
          normalize_input,
          flip_horizontally,
          translate,
          is_initialization=False,
          training = False,
          name=None):
    with tf.name_scope(name, "solar24"):

        training_mode_funcs = [
            nn.step_noise_solar, nn.step_noise_solar24,slim.dropout, wn.fully_connected,nn.gaussian_noise
        ]
        training_args = dict(
            is_training=is_training
        )

        with slim.arg_scope(training_mode_funcs, **training_args):
            net = inputs
            assert_shape(net, [None, 24])

            # net = nn.gaussian_noise(net, scale=input_noise['gaussian'], name='gaussian_noise')
            net = nn.step_noise_solar24(net, step_size=input_noise, name='step_noise')

            net = tf.cond(tf.convert_to_tensor(normalize_input),
                          lambda: slim.layer_norm(net,
                                                  scale=False,
                                                  center=False,
                                                  scope='normalize_inputs'),
                          lambda: net)

            net = wn.fully_connected(net, 16, init=is_initialization,activation_fn=nn.lrelu,scope="dense_1")
            net = wn.fully_connected(net, 32, init=is_initialization,activation_fn=nn.lrelu,scope="dense_2")
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout')
            primary_logits = wn.fully_connected(net, 1, init=is_initialization,scope="dense_out")

            return primary_logits