# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


@slim.add_arg_scope
def global_avg_pool2d(inputs, scope=None):
    
    with tf.variable_scope(scope, 'xx', [inputs]) as sc:
        axis = [1, 2]
        net = tf.reduce_mean(inputs, axis=axis, keep_dims=True)
        
        return net


@slim.add_arg_scope
def block_layer_conv(inputs, num_filters, kernel_size, stride=1, dropout_rate=None,
          scope=None):
    
    with tf.variable_scope(scope, 'xx', [inputs]) as sc:
        net = slim.batch_norm(inputs)
        net = tf.nn.relu(net)
        net = slim.conv2d(net, num_filters, kernel_size)
    
        if dropout_rate:
            net = tf.nn.dropout(net)
    
    return net


@slim.add_arg_scope
def conv_block(inputs, num_filters,  scope=None):
    
    with tf.variable_scope(scope, 'conv_blockx', [inputs]) as sc:
        net = inputs
        net = block_layer_conv(net, num_filters*4, 1, scope='x1')
        net = block_layer_conv(net, num_filters, 3, scope='x2')
        net = tf.concat([inputs, net], axis=3)
    
    return net


@slim.add_arg_scope
def dense_block(inputs, num_layers, num_filters, growth_rate,
                 grow_num_filters=True, scope=None):

    with tf.variable_scope(scope, 'dense_blockx', [inputs]) as sc:
        net = inputs
        for i in range(num_layers):
            branch = i + 1
            net = conv_block(net, growth_rate, scope='conv_block'+str(branch))
        
            if grow_num_filters:
                num_filters += growth_rate

    return net, num_filters


@slim.add_arg_scope
def transition_block(inputs, num_filters, compression=1.0,
                      scope=None):

    num_filters = int(num_filters * compression)
    with tf.variable_scope(scope, 'transition_blockx', [inputs]) as sc:
        net = inputs
        net = block_layer_conv(net, num_filters, 1, scope='blk')
    
        net = slim.avg_pool2d(net, 2)


    return net, num_filters


def densenet(inputs,
             num_classes=80,
             reduction=None,
             growth_rate=None,
             num_filters=None,
             num_layers=None,
             dropout_rate=None,
             is_training=True,
             reuse=None,
             scope=None,
             weight_decay = 0.0001,
             use_batch_norm = True,
             batch_norm_decay=0.997,
             batch_norm_epsilon=1e-5,
             batch_norm_scale=True):
    
    assert reduction is not None
    assert growth_rate is not None
    assert num_filters is not None
    assert num_layers is not None

    compression = 1.0 - reduction
    num_dense_blocks = len(num_layers)

    batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
    }    
      
    with tf.variable_scope(scope, 'densenetxxx', [inputs, num_classes],
                         reuse=reuse) as sc:
        
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                         is_training=is_training), \
             slim.arg_scope([block_layer_conv], dropout_rate=dropout_rate), \
             slim.arg_scope([slim.conv2d],
                         weights_regularizer=slim.l2_regularizer(weight_decay),
                         activation_fn=None,
                         biases_initializer=None), \
             slim.arg_scope([slim.batch_norm], **batch_norm_params):
            
            net = inputs

            # initial convolution
            net = slim.conv2d(net, num_filters, 7, stride=2, scope='conv1')
            net = slim.batch_norm(net)
            net = tf.nn.relu(net)
            net = slim.max_pool2d(net, 3, stride=2, padding='SAME')
            
            # blocks
            for i in range(num_dense_blocks - 1):
                # dense blocks
                net, num_filters = dense_block(net, num_layers[i], num_filters,
                                                growth_rate,
                                                scope='dense_block' + str(i+1))
        
                # Add transition_block
                net, num_filters = transition_block(net, num_filters,
                                                     compression=compression,
                                                     scope='transition_block' + str(i+1))

            net, num_filters = dense_block(
                              net, num_layers[-1], num_filters,
                              growth_rate,
                              scope='dense_block' + str(num_dense_blocks))

            # final blocks
            with tf.variable_scope('final_block', [inputs]):
                net = slim.batch_norm(net)
                net = tf.nn.relu(net)
                net = global_avg_pool2d(net, scope='global_avg_pool')

            net = slim.conv2d(net, num_classes, 1,
                        biases_initializer=tf.zeros_initializer(),
                        scope='logits')


            return net


def densenet121(inputs, 
                num_classes=80,  
                is_training=True, 
                reuse=None, 
                scope = 'densenet121',
                dropout_keep_prob=None,
                weight_decay = 0.0001,
                use_batch_norm = True,
                batch_norm_decay=0.997,
                batch_norm_epsilon=1e-5,
                batch_norm_scale=True):
    
    return densenet(inputs,
                  num_classes=num_classes, 
                  reduction=0.5,
                  growth_rate=32,
                  num_filters=64,
                  num_layers=[6,12,24,16],
                  is_training=is_training,
                  reuse=reuse,
                  dropout_rate = dropout_keep_prob,
                  scope=scope,
                  weight_decay = weight_decay,
                  use_batch_norm = use_batch_norm,
                  batch_norm_decay=batch_norm_decay,
                  batch_norm_epsilon=batch_norm_epsilon,
                  batch_norm_scale=batch_norm_scale)


def densenet169(inputs, 
                num_classes=80,  
                is_training=True, 
                reuse=None,
                scope='densenet169',
                dropout_keep_prob=None,
                weight_decay = 0.0001,
                use_batch_norm = True,
                batch_norm_decay=0.997,
                batch_norm_epsilon=1e-5,
                batch_norm_scale=True):
    
    return densenet(inputs,
                  num_classes=num_classes, 
                  reduction=0.5,
                  growth_rate=32,
                  num_filters=64,
                  num_layers=[6,12,32,32],
                  is_training=is_training,
                  reuse=reuse,
                  dropout_rate = dropout_keep_prob,
                  scope=scope,
                  weight_decay = weight_decay,
                  use_batch_norm = use_batch_norm,
                  batch_norm_decay=batch_norm_decay,
                  batch_norm_epsilon=batch_norm_epsilon,
                  batch_norm_scale=batch_norm_scale)

