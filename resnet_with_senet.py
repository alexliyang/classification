# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import collections


slim = tf.contrib.slim

reduction_ratio = 16

def global_average_pooling(x, stride=1):
    width=np.shape(x)[1]
    height=np.shape(x)[2]
    pool_size= [width, height]
    
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride)

def fully_connected(x, units, scope = None, layer_name='fully_connected') :
    with tf.variable_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def squeeze_excitation_layer(input_x, out_dim, ratio, scope = None, layer_name='SE_C'):
    with tf.variable_scope(scope, layer_name) :
        squeeze = global_average_pooling(input_x)

        excitation = fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = tf.nn.relu(excitation)
        excitation = fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = tf.nn.sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        scale = input_x * excitation
        return scale

def subsample(inputs, factor, scope=None):
    """
    上采样
    """
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    """
    使用SAME padding进行卷积
    """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                         padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           rate=rate, padding='VALID', scope=scope)
        
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """
    Block声明，确定Block的名称，使用的bottleneck函数名及其相应的参数
    """
    
@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None,
                       store_non_strided_activations=False,
                       outputs_collections=None):
#  """
#  堆叠Blocks
#  """

    current_stride = 1
    rate = 1

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            block_stride = 1
            for i, unit in enumerate(block.args):
                if store_non_strided_activations and i == len(block.args) - 1:
                    # Move stride from the block's last unit to the end of the block.
                    block_stride = unit.get('stride', 1)
                    unit = dict(unit, stride=1)
        
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)
        
                    else:
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)
                        if output_stride is not None and current_stride > output_stride:
                            raise ValueError('The target output_stride cannot be reached.')
    
            # Collect activations at the block's end before performing subsampling.
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    
            # Subsampling of the block's output activations.
            if output_stride is not None and current_stride == output_stride:
                rate *= block_stride
            else:
                net = subsample(net, block_stride)
                current_stride *= block_stride
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return net
    
@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None,
               use_bounded_activations=False,
               use_se_module = False):
#    """
#    bottleneck函数实现，残差学习子结构
#    """
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(
                                inputs,
                                depth, [1, 1],
                                stride=stride,
                                activation_fn=tf.nn.relu6 if use_bounded_activations else None,
                                scope='shortcut')
        
        residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                                scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                                            rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               activation_fn=None, scope='conv3')
        
        '对残差分支使用Squeeze and Excitation模块'
        if use_se_module:
            channel = int(np.shape(residual)[-1])
            residual = squeeze_excitation_layer(residual, out_dim=channel, ratio=reduction_ratio, scope=scope, layer_name='SE_C')            

        if use_bounded_activations:
          
            residual = tf.clip_by_value(residual, -6.0, 6.0)
            output = tf.nn.relu6(shortcut + residual)
        else:
            output = tf.nn.relu(shortcut + residual)        
        
        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)

def resnet(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              reuse=None,
              scope=None,
              weight_decay = 0.0001,
              use_batch_norm = True,
              batch_norm_decay=0.997,
              batch_norm_epsilon=1e-5,
              batch_norm_scale=True):
    """
    搭建resnet网络结构
    """
  
    batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
    }
    
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             stack_blocks_dense],
                            outputs_collections=end_points_collection),\
             slim.arg_scope([slim.conv2d],
                          activation_fn=tf.nn.relu,
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                          biases_initializer=tf.constant_initializer(0.01),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          normalizer_fn=slim.batch_norm if use_batch_norm else None,
                          normalizer_params=batch_norm_params),\
            slim.arg_scope([slim.max_pool2d], padding='SAME'):
            
                with slim.arg_scope([slim.batch_norm], **batch_norm_params), \
                     slim.arg_scope([slim.batch_norm],is_training=is_training):
                    
                    net = inputs
                    if include_root_block:
                        if output_stride is not None:
                            if output_stride % 4 != 0:
                                raise ValueError('The output_stride needs to be a multiple of 4.')
                            output_stride /= 4
                        net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                    net = stack_blocks_dense(net, blocks, output_stride)
                    if global_pool:
                        net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                    if num_classes is not None:
                        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                        normalizer_fn=None, scope='logits')
                        if spatial_squeeze:
                            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        
                    return net


def resnet_block(scope, base_depth, num_units, stride,use_se_module = False):
    """
    根据输入参数组建Block
    """
    return Block(scope, bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1,
        'use_se_module':use_se_module
    }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride,
        'use_se_module':use_se_module
    }])


def resnet_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_v1_50',
                 use_se_module = False,
                 weight_decay = 0.0001,
                 use_batch_norm = True,
                 batch_norm_decay=0.997,
                 batch_norm_epsilon=1e-5,
                 batch_norm_scale=True):
    """
    搭建resnet50网络
    """
    blocks = [
        resnet_block('block1', base_depth=64, num_units=3, stride=2, use_se_module = use_se_module),
        resnet_block('block2', base_depth=128, num_units=4, stride=2, use_se_module = use_se_module),
        resnet_block('block3', base_depth=256, num_units=6, stride=2, use_se_module = use_se_module),
        resnet_block('block4', base_depth=512, num_units=3, stride=1, use_se_module = use_se_module),
    ]
    return resnet(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     reuse=reuse, scope=scope, 
                     weight_decay = weight_decay, use_batch_norm = use_batch_norm,
                     batch_norm_decay=batch_norm_decay, batch_norm_epsilon=batch_norm_epsilon, batch_norm_scale=batch_norm_scale)


def resnet_101(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v1_101',
                  use_se_module = False,
                  weight_decay = 0.0001,
                  use_batch_norm = True,
                  batch_norm_decay=0.997,
                  batch_norm_epsilon=1e-5,
                  batch_norm_scale=True):
    """
    搭建resnet101网络
    """
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=3, stride=2, use_se_module = use_se_module),
        resnet_v1_block('block2', base_depth=128, num_units=4, stride=2, use_se_module = use_se_module),
        resnet_v1_block('block3', base_depth=256, num_units=23, stride=2, use_se_module = use_se_module),
        resnet_v1_block('block4', base_depth=512, num_units=3, stride=1, use_se_module = use_se_module),
    ]
    return resnet(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     reuse=reuse, scope=scope, 
                     weight_decay = weight_decay, use_batch_norm = use_batch_norm,
                     batch_norm_decay=batch_norm_decay, batch_norm_epsilon=batch_norm_epsilon, batch_norm_scale=batch_norm_scale)