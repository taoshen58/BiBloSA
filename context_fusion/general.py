from __future__ import absolute_import, division, print_function
import tensorflow as tf
from functools import reduce
from operator import mul

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


def get_last_state(rnn_out_put, mask): # correct
    '''
    get_last_state of rnn output
    :param rnn_out_put: [d1,d2,dn-1,max_len,d]
    :param mask: [d1,d2,dn-1,max_len]
    :return: [d1,d2,dn-1,d]
    '''
    rnn_out_put_flatten = flatten(rnn_out_put, 2)# [X, ml, d]
    mask_flatten = flatten(mask,1) # [X,ml]
    idxs = tf.reduce_sum(tf.cast(mask_flatten,tf.int32),-1) - 1 # [X]
    indices = tf.stack([tf.range(tf.shape(idxs)[0]), idxs], axis=-1) #[X] => [X,2]
    flatten_res = tf.expand_dims(tf.gather_nd(rnn_out_put_flatten, indices),-2 )# #[x,d]->[x,1,d]
    return tf.squeeze(reconstruct(flatten_res,rnn_out_put,2),-2) #[d1,d2,dn-1,1,d] ->[d1,d2,dn-1,d]


def expand_tile(tensor,pattern,tile_num = None, scope=None): # todo: add more func
    with tf.name_scope(scope or 'expand_tile'):
        assert isinstance(pattern,(tuple,list))
        assert isinstance(tile_num,(tuple,list)) or tile_num is None
        assert len(pattern) == len(tile_num) or tile_num is None
        idx_pattern = list([(dim, p) for dim, p in enumerate(pattern)])
        for dim,p in idx_pattern:
            if p == 'x':
                tensor = tf.expand_dims(tensor,dim)
    return tf.tile(tensor,tile_num) if tile_num is not None else tensor


def get_initializer(matrix):
    def _initializer(shape, dtype=None, partition_info=None, **kwargs): return matrix
    return _initializer


def mask(val, mask, name=None):
    if name is None:
        name = 'mask'
    return tf.multiply(val, tf.cast(mask, 'float'), name=name)


def mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.multiply(val, tf.cast(val_mask, tf.float32), name=name or 'mask_for_high_rank')


def exp_mask(val, mask, name=None):
    """Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Args:
        val: values to be masked
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor

    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    if name is None:
        name = "exp_mask"
    return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name=name)


def exp_mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.add(val, (1 - tf.cast(val_mask, tf.float32)) * VERY_NEGATIVE_NUMBER,
                  name=name or 'exp_mask_for_high_rank')


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep, dim_reduced_keep=None):
    dim_reduced_keep = dim_reduced_keep or keep

    ref_shape = ref.get_shape().as_list() # original shape
    tensor_shape = tensor.get_shape().as_list() # current shape
    ref_stop = len(ref_shape) - keep # flatten dims list
    tensor_start = len(tensor_shape) - dim_reduced_keep  # start
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)] #
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))] #
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out


def add_wd(wd, scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    with tf.name_scope("weight_decay"):
        for var in variables:
            counter+=1
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                       name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
            tf.add_to_collection('losses', weight_decay)
        return counter


def add_wd_without_bias(wd, scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    with tf.name_scope("weight_decay"):
        for var in variables:
            if len(var.get_shape().as_list()) <= 1: continue
            counter += 1
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                       name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
            tf.add_to_collection('losses', weight_decay)
        return counter


def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    for var in variables:
        if len(var.get_shape().as_list()) <= 1: continue
        tf.add_to_collection('reg_vars', var)
        counter += 1
    return counter


def add_var_reg(var):
    tf.add_to_collection('reg_vars', var)


def add_wd_for_var(var, wd):
    with tf.name_scope("weight_decay"):
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                   name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
        tf.add_to_collection('losses', weight_decay)



# ------------- selu ----------------
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils


# (1) scale inputs to zero mean and unit variance


# (2) use SELUs
def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


# (3) initialize weights with stddev sqrt(1/n)
# e.g. use:
initializer = layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')


# (4) use this dropout
def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        alpha.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = math_ops.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * math_ops.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))

