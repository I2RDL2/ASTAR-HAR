import tensorflow as tf

def  ramp_temperature(global_step, hyper):
    global_step = tf.to_float(global_step)
    rampdown_length = tf.to_float(hyper['rampdown_length'])
    training_length = tf.to_float(hyper['training_length'])

    x = tf.div(global_step,training_length)
    # def ramp():
    #     phase = 1.0 - tf.maximum(0.0, training_length - global_step) / rampdown_length
    #     return tf.exp(-9 * phase * phase)
    result = (hyper['max_temp']-hyper['min_temp'])* tf.exp(-15 * x * x)+hyper['min_temp']
    # result = 
    return tf.identity(result, name="temp_rampdown")


def ramp_value(global_step, hyper):

    sigmoid_rampup_value = sigmoid_rampup(global_step, hyper['rampup_length'])
    sigmoid_rampdown_value = sigmoid_rampdown(global_step,
                                              hyper['rampdown_length'],
                                              hyper['training_length'])
    learning_rate = tf.multiply(sigmoid_rampup_value * sigmoid_rampdown_value,
                                     hyper['max_learning_rate'],
                                     name='learning_rate')
    adam_beta_1 = tf.add(sigmoid_rampdown_value * hyper['adam_beta_1_before_rampdown'],
                              (1 - sigmoid_rampdown_value) * hyper['adam_beta_1_after_rampdown'],
                              name='adam_beta_1')
    cons_coefficient = tf.multiply(sigmoid_rampup_value,
                                        hyper['max_consistency_cost'],
                                        name='consistency_coefficient')

    step_rampup_value = step_rampup(global_step, hyper['rampup_length'])
    adam_beta_2 = tf.add((1 - step_rampup_value) * hyper['adam_beta_2_during_rampup'],
                              step_rampup_value * hyper['adam_beta_2_after_rampup'],
                              name='adam_beta_2')
    ema_decay = tf.add((1 - step_rampup_value) * hyper['ema_decay_during_rampup'],
                            step_rampup_value * hyper['ema_decay_after_rampup'],
                            name='ema_decay')
    return learning_rate,cons_coefficient, adam_beta_1,adam_beta_2, ema_decay


def step_rampup(global_step, rampup_length):
    result = tf.cond(global_step < rampup_length,
                     lambda: tf.constant(0.0),
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="step_rampup")


def sigmoid_rampup(global_step, rampup_length):
    global_step = tf.to_float(global_step)
    rampup_length = tf.to_float(rampup_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, global_step) / rampup_length
        return tf.exp(-5.0 * phase * phase)

    result = tf.cond(global_step < rampup_length, ramp, lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampup")


def sigmoid_rampdown(global_step, rampdown_length, training_length):
    global_step = tf.to_float(global_step)
    rampdown_length = tf.to_float(rampdown_length)
    training_length = tf.to_float(training_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, training_length - global_step) / rampdown_length
        return tf.exp(-12.5 * phase * phase)

    result = tf.cond(global_step >= training_length - rampdown_length,
                     ramp,
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampdown")