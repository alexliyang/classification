#!/usr/bin/python

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import scene_input
import numpy as np
import time
import argparse
import cv2
import json
import os
import img2tf
from audioop import cross

BATCH_SIZE = 32
IMAGE_SIZE = 224
IMAGE_CHANNEL = 3
NUM_CLASS = 80


def vgg_16(inputs,
           num_classes=80,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           reuse = False,
           weight_decay = 0.0001,
           use_batch_norm = False,
           batch_norm_decay=0.997,
           batch_norm_epsilon=1e-5,
           batch_norm_scale=True):


    batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
    }
    
    with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with    slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      biases_initializer=tf.constant_initializer(0.01)),\
                slim.arg_scope([slim.conv2d],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      normalizer_fn=slim.batch_norm if use_batch_norm else None,
                      normalizer_params=batch_norm_params),\
                slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')
                # Use conv2d instead of fully_connected layers.
                net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
                #net = slim.conv2d(net, 1024, [7, 7], padding='VALID', scope='fc6')
                if dropout_keep_prob:
                    net = slim.dropout(net, dropout_keep_prob, scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                #net = slim.conv2d(net, 1024, [1, 1], scope='fc7')
                if dropout_keep_prob:
                    net = slim.dropout(net, dropout_keep_prob, scope='dropout7')
                net = slim.conv2d(net, num_classes, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='fc8')
                
                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    
                return net
  